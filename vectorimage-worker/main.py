import os
import json
import time
import math
import base64
import secrets
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="VectorImage Worker",
    version="0.5.0",
)

API_TOKEN = os.environ.get(
    "VECTOR_WORKER_TOKEN",
    "",
).strip()

PIPELINE_NAME = "archaeological-line-detector-v3"

GROUP_OUTER = "01_Outer_Contour"
GROUP_STRUCTURAL = "02_Structural_Lines"
GROUP_FINE = "03_Fine_Detail"


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


def verify_token(
    authorization: Optional[str],
):
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
# GENERAL HELPERS
# ============================================================

def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def image_to_base64_png(
    image: np.ndarray,
) -> str:
    if image is None:
        return ""

    ok, buffer = cv2.imencode(
        ".png",
        image,
    )

    if not ok:
        return ""

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


def apply_clahe(
    gray: np.ndarray,
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 8),
    )

    return clahe.apply(
        gray
    )


# ============================================================
# SUBJECT MASK
# ============================================================

def build_subject_mask(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Designed for an archaeological object
    photographed against a predominantly dark
    background.

    Uses Otsu thresholding and keeps the largest
    foreground component.
    """

    gray = cv2.cvtColor(
        image_bgr,
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
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    mask = np.zeros_like(
        gray
    )

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

    # Slight dilation avoids clipping edge details.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
        iterations=1,
    )

    return mask


# ============================================================
# OUTER CONTOUR
# ============================================================

def extract_outer_contour(
    subject_mask: np.ndarray,
    simplification: float,
) -> List[Dict[str, Any]]:

    contours, _ = cv2.findContours(
        subject_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return []

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    epsilon = max(
        1.0,
        simplification * 12.0,
    )

    simplified = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    points = [
        (
            int(p[0][0]),
            int(p[0][1]),
        )
        for p in simplified
    ]

    if len(points) < 3:
        return []

    commands = [
        f"M{points[0][0]} {points[0][1]}"
    ]

    for x, y in points[1:]:
        commands.append(
            f"L{x} {y}"
        )

    commands.append("Z")

    return [
        {
            "id": "outer_1",
            "d": " ".join(commands),
            "fill": "none",
            "stroke": "#000000",
            "strokeWidth": 1.4,
            "strokeLinecap": "round",
            "strokeLinejoin": "round",
            "vectorEffect": "non-scaling-stroke",
            "transform": None,
            "group": GROUP_OUTER,
            "type": "outer-contour",
            "orientation": None,
            "lengthPx": 0,
        }
    ]


# ============================================================
# LINE GEOMETRY
# ============================================================

def line_length(
    line: Tuple[float, float, float, float],
) -> float:
    x1, y1, x2, y2 = line

    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


def line_angle_degrees(
    line: Tuple[float, float, float, float],
) -> float:
    x1, y1, x2, y2 = line

    angle = math.degrees(
        math.atan2(
            y2 - y1,
            x2 - x1,
        )
    )

    angle = abs(angle)

    if angle > 180:
        angle %= 180

    if angle > 90:
        angle = 180 - angle

    return angle


def classify_orientation(
    line: Tuple[float, float, float, float],
    tolerance: float = 12.0,
) -> str:
    angle = line_angle_degrees(
        line
    )

    if angle <= tolerance:
        return "horizontal"

    if angle >= (
        90.0 - tolerance
    ):
        return "vertical"

    return "oblique"


def normalize_line_direction(
    line: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = line

    if (
        x1 > x2
        or (
            x1 == x2
            and y1 > y2
        )
    ):
        return (
            x2,
            y2,
            x1,
            y1,
        )

    return line


# ============================================================
# FAST LINE DETECTOR
# ============================================================

def create_fast_line_detector(
    min_length: int,
    edge_threshold: int,
):
    """
    OpenCV contrib Fast Line Detector.

    Signature:
    length_threshold
    distance_threshold
    canny_th1
    canny_th2
    canny_aperture_size
    do_merge
    """

    low = float(
        clamp(
            edge_threshold * 0.55,
            10,
            180,
        )
    )

    high = float(
        clamp(
            edge_threshold * 1.4,
            low + 1,
            255,
        )
    )

    return cv2.ximgproc.createFastLineDetector(
        max(
            8,
            int(min_length),
        ),
        1.41421356,
        low,
        high,
        3,
        False,
    )


def detect_line_segments(
    enhanced: np.ndarray,
    subject_mask: np.ndarray,
    min_length: int,
    edge_threshold: int,
) -> List[
    Tuple[
        float,
        float,
        float,
        float,
    ]
]:

    # Edge-preserving denoise.
    filtered = cv2.bilateralFilter(
        enhanced,
        d=7,
        sigmaColor=35,
        sigmaSpace=35,
    )

    # Hide background before line detection.
    working = filtered.copy()

    working[
        subject_mask == 0
    ] = 255

    detector = create_fast_line_detector(
        min_length=min_length,
        edge_threshold=edge_threshold,
    )

    detected = detector.detect(
        working
    )

    if detected is None:
        return []

    output = []

    for record in detected:
        x1, y1, x2, y2 = (
            record[0]
        )

        line = (
            float(x1),
            float(y1),
            float(x2),
            float(y2),
        )

        if line_length(
            line
        ) < min_length:
            continue

        # Midpoint must lie inside subject mask.
        mx = int(
            round(
                (x1 + x2) / 2
            )
        )

        my = int(
            round(
                (y1 + y2) / 2
            )
        )

        if (
            mx < 0
            or my < 0
            or mx >= subject_mask.shape[1]
            or my >= subject_mask.shape[0]
        ):
            continue

        if subject_mask[
            my,
            mx,
        ] == 0:
            continue

        output.append(
            normalize_line_direction(
                line
            )
        )

    return output


# ============================================================
# LINE MERGING
# ============================================================

def merge_horizontal_lines(
    lines: List[
        Tuple[
            float,
            float,
            float,
            float,
        ]
    ],
    y_tolerance: float,
    gap_tolerance: float,
) -> List[
    Tuple[
        float,
        float,
        float,
        float,
    ]
]:

    if not lines:
        return []

    normalized = []

    for x1, y1, x2, y2 in lines:

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        average_y = (
            y1 + y2
        ) / 2.0

        normalized.append(
            (
                x1,
                average_y,
                x2,
                average_y,
            )
        )

    normalized.sort(
        key=lambda item: (
            item[1],
            item[0],
        )
    )

    merged = []

    current = list(
        normalized[0]
    )

    for line in normalized[1:]:

        x1, y, x2, _ = line

        cx1, cy, cx2, _ = current

        same_row = (
            abs(
                y - cy
            )
            <= y_tolerance
        )

        close_gap = (
            x1 - cx2
            <= gap_tolerance
        )

        overlaps = (
            x1 <= cx2
        )

        if (
            same_row
            and (
                close_gap
                or overlaps
            )
        ):
            current[2] = max(
                cx2,
                x2,
            )

            current[1] = (
                cy + y
            ) / 2.0

            current[3] = (
                current[1]
            )

        else:
            merged.append(
                tuple(
                    current
                )
            )

            current = list(
                line
            )

    merged.append(
        tuple(
            current
        )
    )

    return merged


def merge_vertical_lines(
    lines: List[
        Tuple[
            float,
            float,
            float,
            float,
        ]
    ],
    x_tolerance: float,
    gap_tolerance: float,
) -> List[
    Tuple[
        float,
        float,
        float,
        float,
    ]
]:

    if not lines:
        return []

    normalized = []

    for x1, y1, x2, y2 in lines:

        if y1 > y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        average_x = (
            x1 + x2
        ) / 2.0

        normalized.append(
            (
                average_x,
                y1,
                average_x,
                y2,
            )
        )

    normalized.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    merged = []

    current = list(
        normalized[0]
    )

    for line in normalized[1:]:

        x, y1, _, y2 = line

        cx, cy1, _, cy2 = current

        same_column = (
            abs(
                x - cx
            )
            <= x_tolerance
        )

        close_gap = (
            y1 - cy2
            <= gap_tolerance
        )

        overlaps = (
            y1 <= cy2
        )

        if (
            same_column
            and (
                close_gap
                or overlaps
            )
        ):
            current[3] = max(
                cy2,
                y2,
            )

            current[0] = (
                cx + x
            ) / 2.0

            current[2] = (
                current[0]
            )

        else:
            merged.append(
                tuple(
                    current
                )
            )

            current = list(
                line
            )

    merged.append(
        tuple(
            current
        )
    )

    return merged


# ============================================================
# OBLIQUE FILTERING
# ============================================================

def filter_oblique_lines(
    lines: List[
        Tuple[
            float,
            float,
            float,
            float,
        ]
    ],
    min_length: float,
) -> List[
    Tuple[
        float,
        float,
        float,
        float,
    ]
]:
    """
    Keep only relatively long oblique structures.

    This aggressively suppresses texture.
    """

    output = []

    for line in lines:

        if line_length(
            line
        ) >= min_length:
            output.append(
                line
            )

    return output


# ============================================================
# DEDUPLICATION
# ============================================================

def quantized_line_key(
    line,
    precision: int = 3,
):
    x1, y1, x2, y2 = line

    return (
        round(
            x1 / precision
        ),
        round(
            y1 / precision
        ),
        round(
            x2 / precision
        ),
        round(
            y2 / precision
        ),
    )


def deduplicate_lines(
    lines,
):
    seen = set()
    result = []

    for line in lines:

        key = quantized_line_key(
            line
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            line
        )

    return result


# ============================================================
# LINES → SVG PATHS
# ============================================================

def line_to_path_record(
    line,
    path_id: str,
    orientation: str,
    stroke_width: float,
):
    x1, y1, x2, y2 = line

    d = (
        f"M{round(x1, 2)} {round(y1, 2)} "
        f"L{round(x2, 2)} {round(y2, 2)}"
    )

    return {
        "id": path_id,
        "d": d,
        "fill": "none",
        "stroke": "#000000",
        "strokeWidth": stroke_width,
        "strokeLinecap": "round",
        "strokeLinejoin": "round",
        "vectorEffect": "non-scaling-stroke",
        "transform": None,
        "group": GROUP_STRUCTURAL,
        "type": "structural",
        "orientation": orientation,
        "lengthPx": round(
            line_length(
                line
            ),
            2,
        ),
    }


def structural_paths_from_lines(
    horizontal,
    vertical,
    oblique,
):
    paths = []

    counter = 1

    for line in horizontal:
        paths.append(
            line_to_path_record(
                line=line,
                path_id=(
                    f"struct_h_{counter}"
                ),
                orientation="horizontal",
                stroke_width=1.0,
            )
        )
        counter += 1

    counter = 1

    for line in vertical:
        paths.append(
            line_to_path_record(
                line=line,
                path_id=(
                    f"struct_v_{counter}"
                ),
                orientation="vertical",
                stroke_width=1.0,
            )
        )
        counter += 1

    counter = 1

    for line in oblique:
        paths.append(
            line_to_path_record(
                line=line,
                path_id=(
                    f"struct_o_{counter}"
                ),
                orientation="oblique",
                stroke_width=0.85,
            )
        )
        counter += 1

    return paths


# ============================================================
# OPTIONAL FINE DETAIL
# ============================================================

def extract_fine_detail_paths(
    enhanced: np.ndarray,
    subject_mask: np.ndarray,
    min_line_length: int,
    line_sensitivity: int,
    max_paths: int = 1200,
) -> List[Dict[str, Any]]:
    """
    Fine detail is deliberately conservative.

    It uses a second Fast Line Detector pass with
    lower minimum length, but caps the number of
    returned paths.

    This is much safer than adaptive-thresholding
    every local photographic texture.
    """

    sensitivity = clamp(
        line_sensitivity,
        0,
        100,
    )

    threshold = int(
        75
        - (
            sensitivity
            * 0.35
        )
    )

    threshold = clamp(
        threshold,
        30,
        70,
    )

    detail_min = max(
        10,
        int(
            min_line_length * 0.7
        ),
    )

    lines = detect_line_segments(
        enhanced=enhanced,
        subject_mask=subject_mask,
        min_length=detail_min,
        edge_threshold=threshold,
    )

    # Avoid duplicating the shortest photographic
    # texture fragments.
    lines = [
        line
        for line in lines
        if line_length(
            line
        ) >= detail_min
    ]

    # Prefer longest features.
    lines.sort(
        key=line_length,
        reverse=True,
    )

    lines = lines[
        :max_paths
    ]

    paths = []

    for index, line in enumerate(
        lines,
        start=1,
    ):
        x1, y1, x2, y2 = line

        paths.append(
            {
                "id": (
                    f"fine_{index}"
                ),
                "d": (
                    f"M{round(x1, 2)} {round(y1, 2)} "
                    f"L{round(x2, 2)} {round(y2, 2)}"
                ),
                "fill": "none",
                "stroke": "#000000",
                "strokeWidth": 0.7,
                "strokeLinecap": "round",
                "strokeLinejoin": "round",
                "vectorEffect": "non-scaling-stroke",
                "transform": None,
                "group": GROUP_FINE,
                "type": "fine-detail",
                "orientation": (
                    classify_orientation(
                        line
                    )
                ),
                "lengthPx": round(
                    line_length(
                        line
                    ),
                    2,
                ),
            }
        )

    return paths


# ============================================================
# SVG GENERATION
# ============================================================

def path_to_svg(
    path: Dict[str, Any],
) -> str:
    return (
        f'<path '
        f'id="{path["id"]}" '
        f'd="{path["d"]}" '
        f'fill="none" '
        f'stroke="{path.get("stroke", "#000000")}" '
        f'stroke-width="{path.get("strokeWidth", 1.0)}" '
        f'stroke-linecap="round" '
        f'stroke-linejoin="round" '
        f'vector-effect="non-scaling-stroke" '
        f'/>'
    )


def svg_group(
    group_id: str,
    paths: List[Dict[str, Any]],
) -> str:

    body = "\n".join(
        path_to_svg(
            path
        )
        for path in paths
    )

    return (
        f'<g id="{group_id}">\n'
        f"{body}\n"
        f"</g>"
    )


def build_svg(
    width: int,
    height: int,
    outer_paths,
    structural_paths,
    fine_paths,
):
    return f"""<svg
xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">
{svg_group(GROUP_OUTER, outer_paths)}
{svg_group(GROUP_STRUCTURAL, structural_paths)}
{svg_group(GROUP_FINE, fine_paths)}
</svg>"""


# ============================================================
# STATISTICS
# ============================================================

def count_anchors(
    paths,
):
    total = 0

    for path in paths:
        d = path.get(
            "d",
            "",
        )

        total += (
            d.count("M")
            + d.count("L")
        )

    return total


def path_statistics(
    paths,
):
    lengths = [
        float(
            path.get(
                "lengthPx",
                0,
            )
        )
        for path in paths
    ]

    lengths = [
        value
        for value in lengths
        if value > 0
    ]

    if not lengths:
        return (
            0.0,
            0,
            0.0,
        )

    average = (
        sum(lengths)
        / len(lengths)
    )

    short_count = sum(
        1
        for value in lengths
        if value < 20
    )

    fragmentation = (
        short_count
        / len(lengths)
    )

    return (
        round(
            average,
            2,
        ),
        short_count,
        round(
            fragmentation,
            4,
        ),
    )


# ============================================================
# DIAGNOSTIC IMAGE
# ============================================================

def draw_detected_lines(
    gray: np.ndarray,
    horizontal,
    vertical,
    oblique,
):
    preview = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    for line in horizontal:
        x1, y1, x2, y2 = map(
            int,
            line,
        )

        cv2.line(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            1,
        )

    for line in vertical:
        x1, y1, x2, y2 = map(
            int,
            line,
        )

        cv2.line(
            preview,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            1,
        )

    for line in oblique:
        x1, y1, x2, y2 = map(
            int,
            line,
        )

        cv2.line(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            1,
        )

    return preview


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    try:
        process = subprocess.run(
            [
                "potrace",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        potrace_ok = (
            process.returncode == 0
        )

        potrace_version = (
            process.stdout
            or process.stderr
            or ""
        ).strip()

    except Exception:

        potrace_ok = False
        potrace_version = "Unavailable"

    contrib_ok = bool(
        hasattr(
            cv2,
            "ximgproc",
        )
        and hasattr(
            cv2.ximgproc,
            "createFastLineDetector",
        )
    )

    return {
        "status": "ok",
        "service": "vectorimage-worker",
        "provider": "opencv-potrace",
        "opencv": True,
        "opencvVersion": cv2.__version__,
        "opencvContrib": contrib_ok,
        "fastLineDetector": contrib_ok,
        "potrace": potrace_ok,
        "potraceVersion": potrace_version,
        "pipeline": PIPELINE_NAME,
        "version": "0.5.0",
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

    request_id = (
        "vec_"
        + secrets.token_hex(6)
    )

    started = time.time()

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    try:
        config = json.loads(
            settings
        )
    except Exception:
        config = {}

    edge_threshold = int(
        config.get(
            "edgeThreshold",
            60,
        )
    )

    line_sensitivity = int(
        config.get(
            "lineSensitivity",
            80,
        )
    )

    min_path_length = int(
        config.get(
            "minPathLength",
            config.get(
                "minimumMeaningfulLineLength",
                30,
            ),
        )
    )

    max_gap = int(
        config.get(
            "maxGapReconnect",
            config.get(
                "maximumGapToReconnect",
                12,
            ),
        )
    )

    path_simplification = float(
        config.get(
            "pathSimplification",
            0.08,
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
            False,
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

    return_diagnostics = bool(
        config.get(
            "returnDiagnostics",
            True,
        )
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    content = await image.read()

    if not content:
        return make_error(
            422,
            "EMPTY_IMAGE",
            "Uploaded image is empty",
            request_id,
        )

    buffer = np.frombuffer(
        content,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        return make_error(
            422,
            "IMAGE_DECODE_FAILED",
            "Image could not be decoded",
            request_id,
        )

    height, width = (
        image_bgr.shape[:2]
    )

    try:

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        enhanced = apply_clahe(
            gray
        )

        subject_mask = build_subject_mask(
            image_bgr
        )

        if not ignore_background_texture:
            subject_mask[:] = 255

        # ----------------------------------------------------
        # PASS 1 — OUTER CONTOUR
        # ----------------------------------------------------

        outer_paths = extract_outer_contour(
            subject_mask,
            simplification=(
                path_simplification
            ),
        )

        # ----------------------------------------------------
        # PASS 2 — STRUCTURAL LINES
        # ----------------------------------------------------

        structural_min_length = max(
            20,
            min_path_length,
        )

        detected_lines = detect_line_segments(
            enhanced=enhanced,
            subject_mask=subject_mask,
            min_length=(
                structural_min_length
            ),
            edge_threshold=(
                edge_threshold
            ),
        )

        horizontal_raw = []
        vertical_raw = []
        oblique_raw = []

        for line in detected_lines:

            orientation = (
                classify_orientation(
                    line,
                    tolerance=12.0,
                )
            )

            if orientation == "horizontal":
                horizontal_raw.append(
                    line
                )

            elif orientation == "vertical":
                vertical_raw.append(
                    line
                )

            else:
                oblique_raw.append(
                    line
                )

        horizontal = merge_horizontal_lines(
            horizontal_raw,
            y_tolerance=4.0,
            gap_tolerance=max(
                8,
                max_gap,
            ),
        )

        vertical = merge_vertical_lines(
            vertical_raw,
            x_tolerance=4.0,
            gap_tolerance=max(
                8,
                max_gap,
            ),
        )

        # Oblique lines must be significantly
        # longer than normal structural segments.
        oblique = filter_oblique_lines(
            oblique_raw,
            min_length=max(
                45,
                structural_min_length * 1.5,
            ),
        )

        horizontal = deduplicate_lines(
            horizontal
        )

        vertical = deduplicate_lines(
            vertical
        )

        oblique = deduplicate_lines(
            oblique
        )

        structural_paths = (
            structural_paths_from_lines(
                horizontal,
                vertical,
                oblique,
            )
        )

        # ----------------------------------------------------
        # PASS 3 — OPTIONAL FINE DETAIL
        # ----------------------------------------------------

        fine_paths = []

        if detect_internal_lines:

            fine_paths = (
                extract_fine_detail_paths(
                    enhanced=enhanced,
                    subject_mask=(
                        subject_mask
                    ),
                    min_line_length=max(
                        12,
                        min_path_length,
                    ),
                    line_sensitivity=(
                        line_sensitivity
                    ),
                    max_paths=(
                        1200
                        if preserve_small_details
                        else 500
                    ),
                )
            )

        # Texture is deliberately NOT automatically
        # added merely because it exists in the photo.
        #
        # This flag is kept for API compatibility.
        if include_texture:
            pass

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
                "No usable vector paths were generated",
                request_id,
            )

        svg = build_svg(
            width=width,
            height=height,
            outer_paths=(
                outer_paths
            ),
            structural_paths=(
                structural_paths
            ),
            fine_paths=(
                fine_paths
            ),
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        anchors = count_anchors(
            all_paths
        )

        (
            average_length,
            short_fragment_count,
            fragmentation_ratio,
        ) = path_statistics(
            structural_paths
            + fine_paths
        )

        processing_ms = int(
            (
                time.time()
                - started
            )
            * 1000
        )

        warnings = []

        if len(all_paths) > 2500:
            warnings.append(
                "Trace contains a high number of paths."
            )

        if fragmentation_ratio > 0.25:
            warnings.append(
                "Trace contains many short fragments."
            )

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        diagnostics = {}

        if return_diagnostics:

            line_preview = draw_detected_lines(
                enhanced,
                horizontal,
                vertical,
                oblique,
            )

            diagnostics = {
                "subjectMask":
                    image_to_base64_png(
                        subject_mask
                    ),

                "enhancedGray":
                    image_to_base64_png(
                        enhanced
                    ),

                "structuralLinePreview":
                    image_to_base64_png(
                        line_preview
                    ),
            }

        # ----------------------------------------------------
        # GROUPS
        # ----------------------------------------------------

        groups = [
            {
                "id": GROUP_OUTER,
                "label": "Outer Contour",
                "paths": outer_paths,
            },

            {
                "id": GROUP_STRUCTURAL,
                "label": "Structural Lines",
                "paths": structural_paths,
                "subgroups": {
                    "horizontal": [
                        path
                        for path
                        in structural_paths
                        if path.get(
                            "orientation"
                        )
                        == "horizontal"
                    ],

                    "vertical": [
                        path
                        for path
                        in structural_paths
                        if path.get(
                            "orientation"
                        )
                        == "vertical"
                    ],

                    "oblique": [
                        path
                        for path
                        in structural_paths
                        if path.get(
                            "orientation"
                        )
                        == "oblique"
                    ],
                },
            },

            {
                "id": GROUP_FINE,
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

            # Keep Base44 contract stable.
            "provider": "opencv-potrace",

            "pipeline": PIPELINE_NAME,
            "workerVersion": "0.5.0",

            "width": width,
            "height": height,

            "viewBox":
                f"0 0 {width} {height}",

            "svg": svg,

            "paths": all_paths,

            "groups": groups,

            "statistics": {
                "pathCount":
                    len(all_paths),

                "anchorCount":
                    anchors,

                "anchorCountEstimate":
                    anchors,

                "outerContourPathCount":
                    len(
                        outer_paths
                    ),

                "structuralPathCount":
                    len(
                        structural_paths
                    ),

                "horizontalStructuralPathCount":
                    len(
                        horizontal
                    ),

                "verticalStructuralPathCount":
                    len(
                        vertical
                    ),

                "obliqueStructuralPathCount":
                    len(
                        oblique
                    ),

                "fineDetailPathCount":
                    len(
                        fine_paths
                    ),

                "rawDetectedLineCount":
                    len(
                        detected_lines
                    ),

                "averagePathLength":
                    average_length,

                "shortFragmentCount":
                    short_fragment_count,

                "fragmentationRatio":
                    fragmentation_ratio,

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

                "minPathLength":
                    min_path_length,

                "maxGapReconnect":
                    max_gap,

                "pathSimplification":
                    path_simplification,

                "preserveSmallDetails":
                    preserve_small_details,

                "detectInternalLines":
                    detect_internal_lines,

                "ignoreBackgroundTexture":
                    ignore_background_texture,

                "includeTexture":
                    include_texture,

                "returnDiagnostics":
                    return_diagnostics,
            },
        }

    except Exception as exc:

        return make_error(
            500,
            "VECTORIZATION_FAILED",
            str(exc),
            request_id,
        )
