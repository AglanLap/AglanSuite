import os
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

from typing import Any, Dict, Optional, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="VectorImage Worker",
    version="0.3.0"
)

API_TOKEN = os.environ.get("VECTOR_WORKER_TOKEN", "").strip()


# ============================================================
# ERROR / AUTH
# ============================================================

def make_error(
    status_code: int,
    error_code: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "requestId": request_id,
            "errorCode": error_code,
            "message": message,
            "details": details or {},
        },
    )


def verify_token(authorization: Optional[str]):
    if not API_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Worker token is not configured",
        )

    expected = f"Bearer {API_TOKEN}"

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid worker token",
        )


# ============================================================
# IMAGE HELPERS
# ============================================================

def np_image_to_base64_png(img: np.ndarray) -> str:
    if img is None:
        return ""

    success, buffer = cv2.imencode(".png", img)

    if not success:
        return ""

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


# ============================================================
# SUBJECT MASK
# ============================================================

def largest_component_mask(binary_img: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    mask = np.zeros_like(binary_img)

    if not contours:
        mask[:] = 255
        return mask

    largest = max(
        contours,
        key=cv2.contourArea,
    )

    cv2.drawContours(
        mask,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return mask


def build_subject_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Optimized mainly for archaeological object
    photographed against a darker background.
    """

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (11, 11),
        0,
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel_close = np.ones(
        (13, 13),
        np.uint8,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=2,
    )

    mask = largest_component_mask(binary)

    # Preserve details very close to the object edge.
    kernel_expand = np.ones(
        (9, 9),
        np.uint8,
    )

    mask = cv2.dilate(
        mask,
        kernel_expand,
        iterations=1,
    )

    return mask


# ============================================================
# POTRACE
# ============================================================

def run_potrace(
    bitmap_black_on_white: np.ndarray,
    simplification: float = 0.15,
    turdsize: int = 5,
) -> str:

    with tempfile.TemporaryDirectory() as tmpdir:

        input_path = os.path.join(
            tmpdir,
            "trace.pgm",
        )

        output_path = os.path.join(
            tmpdir,
            "trace.svg",
        )

        ok = cv2.imwrite(
            input_path,
            bitmap_black_on_white,
        )

        if not ok:
            raise RuntimeError(
                "Failed to write temporary bitmap for Potrace"
            )

        cmd = [
            "potrace",
            input_path,
            "-s",
            "-o",
            output_path,
            "--turdsize",
            str(turdsize),
            "--alphamax",
            "1.0",
            "--opttolerance",
            str(simplification),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or "Potrace failed"
            )

        with open(
            output_path,
            "r",
            encoding="utf-8",
        ) as f:
            return f.read()


def extract_svg_paths(svg: str) -> List[Dict[str, Any]]:
    """
    Preserve Potrace's transform.
    Base44 already supports path transforms.
    """

    try:
        root = ET.fromstring(svg)
    except Exception:
        return []

    namespace = ""

    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0] + "}"

    parent_transform = None
    parent_fill = None
    parent_stroke = None

    for group in root.iter(
        f"{namespace}g"
    ):
        parent_transform = group.attrib.get(
            "transform",
            parent_transform,
        )

        parent_fill = group.attrib.get(
            "fill",
            parent_fill,
        )

        parent_stroke = group.attrib.get(
            "stroke",
            parent_stroke,
        )

    output = []

    i = 1

    for element in root.iter(
        f"{namespace}path"
    ):

        d = element.attrib.get(
            "d",
            "",
        ).strip()

        if not d:
            continue

        output.append(
            {
                "id": element.attrib.get(
                    "id",
                    f"outer_{i}",
                ),
                "d": d,
                "fill": element.attrib.get(
                    "fill",
                    parent_fill or "#000000",
                ),
                "stroke": element.attrib.get(
                    "stroke",
                    parent_stroke or "none",
                ),
                "strokeWidth": element.attrib.get(
                    "stroke-width",
                    None,
                ),
                "transform": element.attrib.get(
                    "transform",
                    parent_transform,
                ),
                "group": "01_Outer_Contour",
                "type": "contour",
            }
        )

        i += 1

    return output


# ============================================================
# OUTER CONTOUR PASS
# ============================================================

def create_outer_contour_bitmap(
    subject_mask: np.ndarray,
) -> np.ndarray:
    """
    Potrace gets a filled subject region.
    White background + black foreground.
    """

    return 255 - subject_mask


# ============================================================
# STRUCTURAL PASS
# ============================================================

def structural_edge_pass(
    enhanced: np.ndarray,
    subject_mask: np.ndarray,
    edge_threshold: int,
    max_gap: int,
) -> np.ndarray:

    blurred = cv2.GaussianBlur(
        enhanced,
        (5, 5),
        0,
    )

    lower = max(
        10,
        int(edge_threshold * 0.45),
    )

    upper = min(
        255,
        int(edge_threshold * 1.6),
    )

    canny = cv2.Canny(
        blurred,
        lower,
        upper,
    )

    canny = cv2.bitwise_and(
        canny,
        canny,
        mask=subject_mask,
    )

    # Horizontal line reconnection.
    horizontal_size = max(
        3,
        min(25, max_gap),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_size, 1),
    )

    horizontal = cv2.morphologyEx(
        canny,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
        iterations=1,
    )

    # Vertical line reconnection.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, horizontal_size),
    )

    vertical = cv2.morphologyEx(
        canny,
        cv2.MORPH_CLOSE,
        vertical_kernel,
        iterations=1,
    )

    combined = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    # Small general gap closing.
    general_kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        general_kernel,
        iterations=1,
    )

    return combined


# ============================================================
# FINE DETAIL PASS
# ============================================================

def fine_detail_pass(
    enhanced: np.ndarray,
    subject_mask: np.ndarray,
    line_sensitivity: int,
    include_texture: bool,
) -> np.ndarray:

    # High sensitivity = lower threshold requirements.
    c_value = max(
        2,
        int(
            14
            - (line_sensitivity / 10)
        ),
    )

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        c_value,
    )

    adaptive = cv2.bitwise_and(
        adaptive,
        adaptive,
        mask=subject_mask,
    )

    if not include_texture:
        # Remove tiny speckle noise.
        kernel = np.ones(
            (2, 2),
            np.uint8,
        )

        adaptive = cv2.morphologyEx(
            adaptive,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

    return adaptive


# ============================================================
# SKELETONIZATION
# ============================================================

def skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    Uses OpenCV contrib thinning.
    """

    binary = (
        binary > 0
    ).astype(np.uint8) * 255

    try:
        result = cv2.ximgproc.thinning(
            binary,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        )

        return result

    except Exception:
        # Fallback morphological skeleton.
        skeleton = np.zeros_like(
            binary
        )

        working = binary.copy()

        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3),
        )

        while True:
            eroded = cv2.erode(
                working,
                kernel,
            )

            opened = cv2.dilate(
                eroded,
                kernel,
            )

            temp = cv2.subtract(
                working,
                opened,
            )

            skeleton = cv2.bitwise_or(
                skeleton,
                temp,
            )

            working = eroded.copy()

            if cv2.countNonZero(
                working
            ) == 0:
                break

        return skeleton


# ============================================================
# SKELETON → CENTERLINE GRAPH
# ============================================================

NEIGHBORS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def pixel_neighbors(
    y: int,
    x: int,
    pixels: set,
) -> List[Tuple[int, int]]:

    result = []

    for dy, dx in NEIGHBORS:

        p = (
            y + dy,
            x + dx,
        )

        if p in pixels:
            result.append(p)

    return result


def skeleton_to_polylines(
    skeleton: np.ndarray,
    min_length: int,
) -> List[List[Tuple[int, int]]]:
    """
    Converts a 1-pixel skeleton into centerline
    polylines.

    Each output point = (x, y).
    """

    ys, xs = np.where(
        skeleton > 0
    )

    pixels = set(
        zip(
            ys.tolist(),
            xs.tolist(),
        )
    )

    if not pixels:
        return []

    degree = {}

    for p in pixels:
        degree[p] = len(
            pixel_neighbors(
                p[0],
                p[1],
                pixels,
            )
        )

    important = {
        p
        for p, d in degree.items()
        if d != 2
    }

    visited_edges = set()
    lines = []

    def edge_key(a, b):
        return tuple(
            sorted(
                [a, b]
            )
        )

    # Trace paths starting from endpoints/junctions.
    for start in important:

        for neighbor in pixel_neighbors(
            start[0],
            start[1],
            pixels,
        ):

            key = edge_key(
                start,
                neighbor,
            )

            if key in visited_edges:
                continue

            line = [
                start,
                neighbor,
            ]

            visited_edges.add(key)

            prev = start
            current = neighbor

            while True:

                if current in important and current != start:
                    break

                candidates = [
                    n
                    for n in pixel_neighbors(
                        current[0],
                        current[1],
                        pixels,
                    )
                    if n != prev
                ]

                if not candidates:
                    break

                next_pixel = candidates[0]

                key = edge_key(
                    current,
                    next_pixel,
                )

                if key in visited_edges:
                    break

                visited_edges.add(key)

                line.append(
                    next_pixel
                )

                prev = current
                current = next_pixel

            if len(line) >= min_length:
                lines.append(
                    [
                        (x, y)
                        for y, x in line
                    ]
                )

    # Handle loops with no endpoints.
    for start in pixels:

        remaining = []

        for neighbor in pixel_neighbors(
            start[0],
            start[1],
            pixels,
        ):
            key = edge_key(
                start,
                neighbor,
            )

            if key not in visited_edges:
                remaining.append(
                    neighbor
                )

        if not remaining:
            continue

        neighbor = remaining[0]

        line = [
            start,
            neighbor,
        ]

        visited_edges.add(
            edge_key(
                start,
                neighbor,
            )
        )

        prev = start
        current = neighbor

        while True:

            candidates = [
                n
                for n in pixel_neighbors(
                    current[0],
                    current[1],
                    pixels,
                )
                if n != prev
            ]

            unvisited = []

            for n in candidates:

                key = edge_key(
                    current,
                    n,
                )

                if key not in visited_edges:
                    unvisited.append(
                        n
                    )

            if not unvisited:
                break

            next_pixel = unvisited[0]

            visited_edges.add(
                edge_key(
                    current,
                    next_pixel,
                )
            )

            line.append(
                next_pixel
            )

            prev = current
            current = next_pixel

            if current == start:
                break

        if len(line) >= min_length:
            lines.append(
                [
                    (x, y)
                    for y, x in line
                ]
            )

    return lines


# ============================================================
# POLYLINE SIMPLIFICATION
# ============================================================

def simplify_polyline(
    points: List[Tuple[int, int]],
    epsilon: float,
) -> List[Tuple[int, int]]:

    if len(points) < 3:
        return points

    contour = np.array(
        points,
        dtype=np.float32,
    ).reshape(
        (-1, 1, 2)
    )

    approximated = cv2.approxPolyDP(
        contour,
        epsilon,
        False,
    )

    return [
        (
            int(p[0][0]),
            int(p[0][1]),
        )
        for p in approximated
    ]


# ============================================================
# CENTERLINE → SVG
# ============================================================

def polyline_to_svg_path(
    points: List[Tuple[int, int]],
) -> str:

    if len(points) < 2:
        return ""

    pieces = [
        f"M{points[0][0]} {points[0][1]}"
    ]

    for x, y in points[1:]:
        pieces.append(
            f"L{x} {y}"
        )

    return " ".join(
        pieces
    )


def polylines_to_path_records(
    lines: List[List[Tuple[int, int]]],
    prefix: str,
    group: str,
    simplification: float,
    stroke_width: float = 1.0,
) -> List[Dict[str, Any]]:

    paths = []

    counter = 1

    epsilon = max(
        0.5,
        simplification * 4.0,
    )

    for line in lines:

        simplified = simplify_polyline(
            line,
            epsilon,
        )

        if len(simplified) < 2:
            continue

        d = polyline_to_svg_path(
            simplified
        )

        if not d:
            continue

        paths.append(
            {
                "id": f"{prefix}_{counter}",
                "d": d,
                "fill": "none",
                "stroke": "#000000",
                "strokeWidth": stroke_width,
                "strokeLinecap": "round",
                "strokeLinejoin": "round",
                "transform": None,
                "group": group,
                "type": "centerline",
            }
        )

        counter += 1

    return paths


# ============================================================
# SVG BUILDING
# ============================================================

def path_record_to_svg(
    path: Dict[str, Any],
) -> str:

    attrs = [
        f'id="{path["id"]}"',
        f'd="{path["d"]}"',
    ]

    fill = path.get(
        "fill"
    )

    stroke = path.get(
        "stroke"
    )

    stroke_width = path.get(
        "strokeWidth"
    )

    transform = path.get(
        "transform"
    )

    if fill is not None:
        attrs.append(
            f'fill="{fill}"'
        )

    if stroke is not None:
        attrs.append(
            f'stroke="{stroke}"'
        )

    if stroke_width is not None:
        attrs.append(
            f'stroke-width="{stroke_width}"'
        )

    if path.get(
        "strokeLinecap"
    ):
        attrs.append(
            f'stroke-linecap="{path["strokeLinecap"]}"'
        )

    if path.get(
        "strokeLinejoin"
    ):
        attrs.append(
            f'stroke-linejoin="{path["strokeLinejoin"]}"'
        )

    if transform:
        attrs.append(
            f'transform="{transform}"'
        )

    return (
        "<path "
        + " ".join(attrs)
        + " />"
    )


def build_group_svg(
    group_id: str,
    paths: List[Dict[str, Any]],
) -> str:

    inner = "\n".join(
        path_record_to_svg(p)
        for p in paths
    )

    return (
        f'<g id="{group_id}">\n'
        f"{inner}\n"
        "</g>"
    )


def build_final_svg(
    width: int,
    height: int,
    outer_paths: List[Dict[str, Any]],
    structural_paths: List[Dict[str, Any]],
    fine_paths: List[Dict[str, Any]],
) -> str:

    groups = [
        build_group_svg(
            "01_Outer_Contour",
            outer_paths,
        ),
        build_group_svg(
            "02_Structural_Lines",
            structural_paths,
        ),
        build_group_svg(
            "03_Fine_Detail",
            fine_paths,
        ),
    ]

    return f"""<svg
xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">
{chr(10).join(groups)}
</svg>"""


# ============================================================
# STATISTICS
# ============================================================

SVG_COMMAND_PATTERN = re.compile(
    r"[MLCQASTHVZmlcqasthvz]"
)


def estimate_anchor_count(
    paths: List[Dict[str, Any]],
) -> int:

    total = 0

    for p in paths:
        total += len(
            SVG_COMMAND_PATTERN.findall(
                p.get(
                    "d",
                    "",
                )
            )
        )

    return total


def approximate_path_length(
    path: Dict[str, Any],
) -> float:
    """
    Approximation based on numeric coordinate pairs.
    Enough for diagnostics.
    """

    values = re.findall(
        r"-?\d+(?:\.\d+)?",
        path.get(
            "d",
            "",
        ),
    )

    if len(values) < 4:
        return 0.0

    nums = [
        float(x)
        for x in values
    ]

    points = []

    for i in range(
        0,
        len(nums) - 1,
        2,
    ):
        points.append(
            (
                nums[i],
                nums[i + 1],
            )
        )

    total = 0.0

    for i in range(
        1,
        len(points),
    ):

        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        total += (
            (x2 - x1) ** 2
            + (y2 - y1) ** 2
        ) ** 0.5

    return total


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    try:
        result = subprocess.run(
            [
                "potrace",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        potrace_ok = (
            result.returncode == 0
        )

        potrace_version = (
            result.stdout
            or result.stderr
            or ""
        ).strip()

    except Exception:

        potrace_ok = False
        potrace_version = "Unavailable"

    thinning_ok = bool(
        hasattr(
            cv2,
            "ximgproc",
        )
    )

    return {
        "status": (
            "ok"
            if potrace_ok
            else "degraded"
        ),
        "service": "vectorimage-worker",
        "provider": "opencv-potrace",
        "opencv": True,
        "opencvContrib": thinning_ok,
        "potrace": potrace_ok,
        "potraceVersion": potrace_version,
        "pipeline": "hybrid-multipass-centerline",
        "version": "0.3.0",
    }


# ============================================================
# VECTORIZE
# ============================================================

@app.post("/vectorize")
async def vectorize(
    image: UploadFile = File(...),
    settings: str = Form("{}"),
    authorization: Optional[str] = Header(
        default=None
    ),
):

    verify_token(
        authorization
    )

    started = time.time()

    request_id = (
        "vec_"
        + secrets.token_hex(6)
    )

    try:
        config = json.loads(
            settings
        )

    except Exception:
        config = {}

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    edge_threshold = int(
        config.get(
            "edgeThreshold",
            55,
        )
    )

    line_sensitivity = int(
        config.get(
            "lineSensitivity",
            85,
        )
    )

    preserve_small_details = bool(
        config.get(
            "preserveSmallDetails",
            True,
        )
    )

    detect_internal_lines = bool(
        config.get(
            "detectInternalLines",
            True,
        )
    )

    ignore_background_texture = bool(
        config.get(
            "ignoreBackgroundTexture",
            True,
        )
    )

    include_texture = bool(
        config.get(
            "includeTexture",
            config.get(
                "includeCracksTexture",
                False,
            ),
        )
    )

    min_path_length = int(
        config.get(
            "minPathLength",
            config.get(
                "minimumMeaningfulLineLength",
                8,
            ),
        )
    )

    max_gap = int(
        config.get(
            "maxGapReconnect",
            config.get(
                "maximumGapToReconnect",
                10,
            ),
        )
    )

    path_simplification = float(
        config.get(
            "pathSimplification",
            0.08,
        )
    )

    return_diagnostics = bool(
        config.get(
            "returnDiagnostics",
            True,
        )
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    content = await image.read()

    if not content:
        return make_error(
            422,
            "EMPTY_IMAGE",
            "Uploaded image is empty",
            request_id,
        )

    np_buffer = np.frombuffer(
        content,
        dtype=np.uint8,
    )

    img_bgr = cv2.imdecode(
        np_buffer,
        cv2.IMREAD_COLOR,
    )

    if img_bgr is None:
        return make_error(
            422,
            "IMAGE_DECODE_FAILED",
            "Image could not be decoded",
            request_id,
        )

    height, width = (
        img_bgr.shape[:2]
    )

    try:

        # ----------------------------------------------------
        # STAGE 1 — MASK
        # ----------------------------------------------------

        subject_mask = build_subject_mask(
            img_bgr
        )

        if not ignore_background_texture:
            subject_mask[:] = 255

        # ----------------------------------------------------
        # STAGE 2 — CONTRAST
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        enhanced = apply_clahe(
            gray
        )

        # ----------------------------------------------------
        # PASS 1 — OUTER CONTOUR
        # ----------------------------------------------------

        outer_bitmap = (
            create_outer_contour_bitmap(
                subject_mask
            )
        )

        outer_svg_raw = run_potrace(
            outer_bitmap,
            simplification=max(
                0.05,
                path_simplification,
            ),
            turdsize=20,
        )

        outer_paths = extract_svg_paths(
            outer_svg_raw
        )

        # ----------------------------------------------------
        # PASS 2 — STRUCTURAL
        # ----------------------------------------------------

        structural_edges = (
            structural_edge_pass(
                enhanced,
                subject_mask,
                edge_threshold,
                max_gap,
            )
        )

        structural_skeleton = skeletonize(
            structural_edges
        )

        structural_lines = skeleton_to_polylines(
            structural_skeleton,
            max(
                4,
                min_path_length,
            ),
        )

        structural_paths = (
            polylines_to_path_records(
                structural_lines,
                prefix="struct",
                group="02_Structural_Lines",
                simplification=path_simplification,
                stroke_width=1.0,
            )
        )

        # ----------------------------------------------------
        # PASS 3 — FINE DETAIL
        # ----------------------------------------------------

        fine_paths = []

        fine_edges = np.zeros_like(
            gray
        )

        fine_skeleton = np.zeros_like(
            gray
        )

        if detect_internal_lines:

            fine_edges = fine_detail_pass(
                enhanced,
                subject_mask,
                line_sensitivity,
                include_texture,
            )

            fine_skeleton = skeletonize(
                fine_edges
            )

            fine_min = (
                max(
                    3,
                    min_path_length // 2,
                )
                if preserve_small_details
                else max(
                    6,
                    min_path_length,
                )
            )

            fine_lines = skeleton_to_polylines(
                fine_skeleton,
                fine_min,
            )

            fine_paths = (
                polylines_to_path_records(
                    fine_lines,
                    prefix="fine",
                    group="03_Fine_Detail",
                    simplification=max(
                        0.02,
                        path_simplification * 0.7,
                    ),
                    stroke_width=0.8,
                )
            )

        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        all_paths = (
            outer_paths
            + structural_paths
            + fine_paths
        )

        if not all_paths:
            return make_error(
                422,
                "NO_PATHS_GENERATED",
                "No usable SVG paths were generated",
                request_id,
            )

        svg = build_final_svg(
            width,
            height,
            outer_paths,
            structural_paths,
            fine_paths,
        )

        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------

        anchor_count = estimate_anchor_count(
            all_paths
        )

        path_lengths = [
            approximate_path_length(p)
            for p in (
                structural_paths
                + fine_paths
            )
        ]

        valid_lengths = [
            x
            for x in path_lengths
            if x > 0
        ]

        average_path_length = (
            sum(valid_lengths)
            / len(valid_lengths)
            if valid_lengths
            else 0.0
        )

        short_fragment_count = sum(
            1
            for x in valid_lengths
            if x < 20
        )

        processing_ms = int(
            (
                time.time()
                - started
            )
            * 1000
        )

        warnings = []

        if len(all_paths) > 5000:
            warnings.append(
                "Trace contains a very high number of paths."
            )

        if (
            short_fragment_count
            > max(
                100,
                len(valid_lengths) * 0.35,
            )
        ):
            warnings.append(
                "Trace contains many short fragments."
            )

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        diagnostics = {}

        if return_diagnostics:

            merged_binary = cv2.bitwise_or(
                structural_skeleton,
                fine_skeleton,
            )

            diagnostics = {
                "subjectMask":
                    np_image_to_base64_png(
                        subject_mask
                    ),

                "enhancedGray":
                    np_image_to_base64_png(
                        enhanced
                    ),

                "outerContourBitmap":
                    np_image_to_base64_png(
                        outer_bitmap
                    ),

                "structuralEdges":
                    np_image_to_base64_png(
                        structural_edges
                    ),

                "structuralSkeleton":
                    np_image_to_base64_png(
                        structural_skeleton
                    ),

                "fineDetailEdges":
                    np_image_to_base64_png(
                        fine_edges
                    ),

                "fineDetailSkeleton":
                    np_image_to_base64_png(
                        fine_skeleton
                    ),

                "mergedBinary":
                    np_image_to_base64_png(
                        merged_binary
                    ),
            }

        # ----------------------------------------------------
        # GROUPS
        # ----------------------------------------------------

        groups = [
            {
                "id": "01_Outer_Contour",
                "label": "Outer Contour",
                "paths": outer_paths,
            },
            {
                "id": "02_Structural_Lines",
                "label": "Structural Lines",
                "paths": structural_paths,
            },
            {
                "id": "03_Fine_Detail",
                "label": "Fine Detail",
                "paths": fine_paths,
            },
        ]

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "success": True,
            "requestId": request_id,

            # Keep backward compatibility
            "provider": "opencv-potrace",

            "pipeline":
                "hybrid-multipass-centerline",

            "workerVersion": "0.3.0",

            "width": width,
            "height": height,

            "viewBox":
                f"0 0 {width} {height}",

            "svg": svg,

            # Flat path array remains available
            "paths": all_paths,

            # New grouped output
            "groups": groups,

            "statistics": {
                "pathCount":
                    len(all_paths),

                "anchorCount":
                    anchor_count,

                "anchorCountEstimate":
                    anchor_count,

                "outerContourPathCount":
                    len(outer_paths),

                "structuralPathCount":
                    len(structural_paths),

                "fineDetailPathCount":
                    len(fine_paths),

                "averagePathLength":
                    round(
                        average_path_length,
                        2,
                    ),

                "shortFragmentCount":
                    short_fragment_count,

                "processingTimeMs":
                    processing_ms,

                "svgBytes":
                    len(
                        svg.encode(
                            "utf-8"
                        )
                    ),
            },

            "warnings": warnings,

            "diagnostics": diagnostics,

            "settingsUsed": {
                "edgeThreshold":
                    edge_threshold,

                "lineSensitivity":
                    line_sensitivity,

                "preserveSmallDetails":
                    preserve_small_details,

                "detectInternalLines":
                    detect_internal_lines,

                "ignoreBackgroundTexture":
                    ignore_background_texture,

                "includeTexture":
                    include_texture,

                "minPathLength":
                    min_path_length,

                "maxGapReconnect":
                    max_gap,

                "pathSimplification":
                    path_simplification,

                "returnDiagnostics":
                    return_diagnostics,
            },
        }

    except subprocess.TimeoutExpired:

        return make_error(
            504,
            "POTRACE_TIMEOUT",
            "Potrace processing exceeded the allowed time",
            request_id,
        )

    except Exception as exc:

        return make_error(
            500,
            "VECTORIZATION_FAILED",
            str(exc),
            request_id,
        )