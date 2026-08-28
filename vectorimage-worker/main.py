import os
import io
import re
import cv2
import json
import time
import base64
import secrets
import tempfile
import subprocess
import numpy as np
import xml.etree.ElementTree as ET

from typing import Any, Dict, Optional, List

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse


app = FastAPI(title="VectorImage Worker", version="0.2.0")

API_TOKEN = os.environ.get("VECTOR_WORKER_TOKEN", "").strip()


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def make_error(
    status_code: int,
    error_code: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "requestId": request_id,
            "errorCode": error_code,
            "message": message,
            "details": details or {}
        }
    )


def verify_token(authorization: Optional[str]):
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="Worker token is not configured")

    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid worker token")


def np_image_to_base64_png(img: np.ndarray) -> str:
    """
    Converts a numpy image (gray or BGR) to a base64 PNG data URL.
    """
    if img is None:
        return ""

    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""

    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def find_largest_contour_mask(binary_img: np.ndarray) -> np.ndarray:
    """
    Finds the largest external contour and returns a filled mask.
    If no contour is found, returns a full-white mask.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(binary_img)

    if not contours:
        mask[:] = 255
        return mask

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def build_subject_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Very simple subject-isolation approach:
    - Convert to gray
    - Blur
    - Otsu threshold
    - Keep largest connected component
    - Close / open to stabilize mask

    This works reasonably for a light archaeological object on a dark cloth background.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Since the object is lighter than the dark background,
    # regular THRESH_BINARY + OTSU often works well.
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel_close = np.ones((9, 9), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)

    mask = find_largest_contour_mask(cleaned)

    # Slight dilation to preserve near-edge details
    dilate_kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    return mask


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def compute_edges(
    gray_enhanced: np.ndarray,
    edge_threshold: int,
    detect_internal_lines: bool,
    line_sensitivity: int
) -> np.ndarray:
    """
    Produces an edge map.
    Combines:
    - Canny edges
    - Optional adaptive threshold for internal details
    """
    blurred = cv2.GaussianBlur(gray_enhanced, (3, 3), 0)

    lower = max(5, int(edge_threshold * 0.5))
    upper = min(255, int(edge_threshold * 1.8))

    canny = cv2.Canny(blurred, lower, upper)

    if not detect_internal_lines:
        return canny

    # Adaptive threshold can help recover internal painted or carved lines.
    block_size = 31
    c_value = max(2, int(12 - (line_sensitivity / 15)))
    adaptive = cv2.adaptiveThreshold(
        gray_enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )

    combined = cv2.bitwise_or(canny, adaptive)
    return combined


def clean_edge_map(
    edges: np.ndarray,
    subject_mask: np.ndarray,
    ignore_background_texture: bool,
    preserve_small_details: bool,
    line_sensitivity: int,
    min_path_length: int
) -> np.ndarray:
    """
    Cleans the edge map using:
    - optional subject masking
    - morphological closing
    - contour filtering by path length
    """
    work = edges.copy()

    if ignore_background_texture:
        work = cv2.bitwise_and(work, work, mask=subject_mask)

    # Kernel size based on sensitivity
    if line_sensitivity >= 80:
        close_ksize = 5
    elif line_sensitivity >= 50:
        close_ksize = 3
    else:
        close_ksize = 2

    kernel = np.ones((close_ksize, close_ksize), np.uint8)

    # Join nearby line fragments
    cleaned = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Optional slight opening to remove speckles
    if not preserve_small_details:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Filter very short contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered = np.zeros_like(cleaned)

    for cnt in contours:
        length = cv2.arcLength(cnt, False)
        if length >= min_path_length:
            cv2.drawContours(filtered, [cnt], -1, 255, thickness=1)

    return filtered


def run_potrace(bitmap_black_on_white: np.ndarray, simplification: float, preserve_small_details: bool):
    """
    Runs Potrace on a bitmap image and returns SVG string.
    Potrace expects black foreground on white background.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "trace.pgm")
        output_path = os.path.join(tmpdir, "trace.svg")

        ok = cv2.imwrite(input_path, bitmap_black_on_white)
        if not ok:
            raise RuntimeError("Failed to write temporary PGM file for Potrace")

        turdsize = 2 if preserve_small_details else 8
        opttolerance = str(max(0.0, min(1.5, simplification)))

        cmd = [
            "potrace",
            input_path,
            "-s",
            "-o", output_path,
            "--turdsize", str(turdsize),
            "--alphamax", "1.0",
            "--opttolerance", opttolerance
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Potrace failed")

        with open(output_path, "r", encoding="utf-8") as f:
            svg = f.read()

        return svg


def extract_svg_paths(svg: str) -> List[Dict[str, Any]]:
    """
    Extracts <path> elements from the SVG.
    Potrace often places paths inside a <g transform="...">.
    We preserve basic attributes and also return group transform when present.
    """
    try:
        root = ET.fromstring(svg)
    except Exception:
        return []

    # SVG namespace handling
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].strip("{")
        ns = {"svg": ns_uri}
        path_query = ".//svg:path"
        group_query = ".//svg:g"
    else:
        ns = {}
        path_query = ".//path"
        group_query = ".//g"

    group_transform = None
    for g in root.findall(group_query, ns):
        transform = g.attrib.get("transform")
        if transform:
            group_transform = transform
            break

    paths = []
    for i, path_el in enumerate(root.findall(path_query, ns)):
        d = path_el.attrib.get("d", "").strip()
        if not d:
            continue

        paths.append({
            "id": path_el.attrib.get("id", f"path_{i+1}"),
            "d": d,
            "fill": path_el.attrib.get("fill", "none"),
            "stroke": path_el.attrib.get("stroke", "#000000"),
            "transform": path_el.attrib.get("transform", group_transform)
        })

    return paths


def estimate_anchor_count_from_paths(paths: List[Dict[str, Any]]) -> int:
    """
    Rough estimate based on SVG commands in path 'd'.
    """
    anchor_estimate = 0
    pattern = re.compile(r"[MLCQASTHVZmlcqasthvz]")
    for p in paths:
        d = p.get("d", "")
        anchor_estimate += len(pattern.findall(d))
    return anchor_estimate


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/health")
def health():
    try:
        result = subprocess.run(
            ["potrace", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        potrace_ok = result.returncode == 0
        potrace_version = (result.stdout or result.stderr or "").strip()
    except Exception:
        potrace_ok = False
        potrace_version = "Unavailable"

    return {
        "status": "ok" if potrace_ok else "degraded",
        "service": "vectorimage-worker",
        "provider": "opencv-potrace",
        "opencv": True,
        "potrace": potrace_ok,
        "potraceVersion": potrace_version,
        "version": "0.2.0"
    }


@app.post("/vectorize")
async def vectorize(
    image: UploadFile = File(...),
    settings: str = Form("{}"),
    authorization: Optional[str] = Header(default=None)
):
    verify_token(authorization)

    started = time.time()
    request_id = f"vec_{secrets.token_hex(6)}"

    try:
        config = json.loads(settings)
    except Exception:
        config = {}

    # Settings with defaults
    edge_threshold = int(config.get("edgeThreshold", 80))
    line_sensitivity = int(config.get("lineSensitivity", 60))
    preserve_small_details = bool(config.get("preserveSmallDetails", True))
    detect_internal_lines = bool(config.get("detectInternalLines", True))
    ignore_background_texture = bool(config.get("ignoreBackgroundTexture", True))
    min_path_length = int(config.get("minPathLength", 18))
    path_simplification = float(config.get("pathSimplification", 0.2))
    return_diagnostics = bool(config.get("returnDiagnostics", True))

    # Read uploaded file
    content = await image.read()
    if not content:
        return make_error(
            422,
            "EMPTY_IMAGE",
            "Uploaded image is empty",
            request_id=request_id
        )

    np_buffer = np.frombuffer(content, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return make_error(
            422,
            "IMAGE_DECODE_FAILED",
            "Image could not be decoded",
            request_id=request_id
        )

    height, width = img_bgr.shape[:2]

    try:
        # ------------------------------
        # Diagnostic Stage 1: Original
        # ------------------------------
        original_preview = img_bgr.copy()

        # ------------------------------
        # Subject Mask
        # ------------------------------
        subject_mask = build_subject_mask(img_bgr)

        # ------------------------------
        # Gray + Enhanced
        # ------------------------------
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = apply_clahe(gray)

        # ------------------------------
        # Edge detection
        # ------------------------------
        edges = compute_edges(
            enhanced,
            edge_threshold=edge_threshold,
            detect_internal_lines=detect_internal_lines,
            line_sensitivity=line_sensitivity
        )

        # ------------------------------
        # Cleanup
        # ------------------------------
        cleaned_edges = clean_edge_map(
            edges=edges,
            subject_mask=subject_mask,
            ignore_background_texture=ignore_background_texture,
            preserve_small_details=preserve_small_details,
            line_sensitivity=line_sensitivity,
            min_path_length=min_path_length
        )

        # Potrace expects black foreground on white background
        binary_for_potrace = 255 - cleaned_edges

        # ------------------------------
        # Vectorization
        # ------------------------------
        svg = run_potrace(
            bitmap_black_on_white=binary_for_potrace,
            simplification=path_simplification,
            preserve_small_details=preserve_small_details
        )

        paths = extract_svg_paths(svg)
        path_count = len(paths)
        anchor_count = estimate_anchor_count_from_paths(paths)

        if path_count == 0:
            return make_error(
                422,
                "NO_PATHS_GENERATED",
                "No usable SVG paths were generated",
                request_id=request_id,
                details={
                    "provider": "opencv-potrace",
                    "width": width,
                    "height": height
                }
            )

        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "original": np_image_to_base64_png(original_preview),
                "subjectMask": np_image_to_base64_png(subject_mask),
                "enhancedGray": np_image_to_base64_png(enhanced),
                "edges": np_image_to_base64_png(edges),
                "cleanedEdges": np_image_to_base64_png(cleaned_edges),
                "binaryForPotrace": np_image_to_base64_png(binary_for_potrace),
            }

        processing_ms = int((time.time() - started) * 1000)

        # Some extra quality-related diagnostics
        nonzero_cleaned = int(np.count_nonzero(cleaned_edges))
        nonzero_mask = int(np.count_nonzero(subject_mask))
        fragmentation_ratio = 0.0
        if path_count > 0:
            short_frag_estimate = sum(1 for p in paths if len(p.get("d", "")) < 60)
            fragmentation_ratio = short_frag_estimate / path_count

        warnings = []
        if fragmentation_ratio > 0.4:
            warnings.append("Trace quality low — excessive fragmentation detected.")
        if nonzero_cleaned < 200:
            warnings.append("Very little edge information survived preprocessing.")

        return {
            "success": True,
            "requestId": request_id,
            "provider": "opencv-potrace",
            "width": width,
            "height": height,
            "viewBox": f"0 0 {width} {height}",
            "svg": svg,
            "paths": paths,
            "statistics": {
                "pathCount": path_count,
                "anchorCountEstimate": anchor_count,
                "processingTimeMs": processing_ms,
                "svgBytes": len(svg.encode("utf-8")),
                "cleanedEdgePixels": nonzero_cleaned,
                "subjectMaskPixels": nonzero_mask,
                "fragmentationRatio": fragmentation_ratio
            },
            "warnings": warnings,
            "diagnostics": diagnostics,
            "settingsUsed": {
                "edgeThreshold": edge_threshold,
                "lineSensitivity": line_sensitivity,
                "preserveSmallDetails": preserve_small_details,
                "detectInternalLines": detect_internal_lines,
                "ignoreBackgroundTexture": ignore_background_texture,
                "minPathLength": min_path_length,
                "pathSimplification": path_simplification,
                "returnDiagnostics": return_diagnostics
            }
        }

    except subprocess.TimeoutExpired:
        return make_error(
            504,
            "POTRACE_TIMEOUT",
            "Potrace processing exceeded the allowed time",
            request_id=request_id
        )

    except Exception as e:
        return make_error(
            500,
            "VECTORIZATION_FAILED",
            str(e),
            request_id=request_id
        )