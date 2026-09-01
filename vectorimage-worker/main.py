import os
import json
import time
import math
import base64
import secrets
import subprocess
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse


# ============================================================
# APPLICATION
# ============================================================

VERSION = "0.7.1"
PIPELINE_NAME = "archaeological-roi-tracer-v1.1"

app = FastAPI(
    title="VectorImage Worker",
    version=VERSION,
)

API_TOKEN = os.environ.get(
    "VECTOR_WORKER_TOKEN",
    "",
).strip()


# ============================================================
# GROUPS
# ============================================================

GROUP_OUTER = "01_Outer_Contour"
GROUP_GLOBAL = "02_Global_Structural"
GROUP_ROI_PREFIX = "ROI"


# ============================================================
# TYPES
# ============================================================

Line = Tuple[float, float, float, float]


# ============================================================
# ROI PRESETS
# ============================================================

DEFAULT_PRESETS = {

    # NEW v0.7.1:
    # processed by trace_painted_grid(), not generic FLD.
    "painted_grid": {
        "detector": "painted_grid",

        "edgeThreshold": 42,

        # Morphological line extraction
        "horizontalKernelPx": 25,
        "verticalKernelPx": 11,

        "horizontalThicknessPx": 3,
        "verticalThicknessPx": 3,

        "minimumHorizontalLength": 18,
        "minimumVerticalLength": 7,

        "minimumComponentArea": 10,

        "horizontalGap": 20,
        "verticalGap": 7,

        "horizontalTolerance": 5,
        "verticalTolerance": 4,

        # Reject obviously non-grid components
        "maximumObliqueAngle": 12,

        "irregularDetail": False,
        "irregularCap": 0,
    },

    "ornament": {
        "detector": "generic",

        "edgeThreshold": 38,
        "minLineLength": 8,
        "maxGap": 8,

        "horizontalTolerance": 5,
        "verticalTolerance": 5,

        "irregularDetail": True,
        "irregularCap": 350,
    },

    "wood_texture": {
        "detector": "generic",

        "edgeThreshold": 62,
        "minLineLength": 18,
        "maxGap": 8,

        "horizontalTolerance": 5,
        "verticalTolerance": 5,

        "irregularDetail": True,
        "irregularCap": 250,
    },

    "general_detail": {
        "detector": "generic",

        "edgeThreshold": 48,
        "minLineLength": 14,
        "maxGap": 12,

        "horizontalTolerance": 5,
        "verticalTolerance": 5,

        "irregularDetail": True,
        "irregularCap": 250,
    },
}


# ============================================================
# AUTH / ERRORS
# ============================================================

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


def make_error(
    status_code: int,
    code: str,
    message: str,
    request_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "requestId": request_id,
            "errorCode": code,
            "message": message,
            "details": details or {},
        },
    )


# ============================================================
# BASIC HELPERS
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

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
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
        cv2.FILLED,
    )

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

def outer_contour_path(
    mask: np.ndarray,
) -> Optional[Dict[str, Any]]:

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return None

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    simplified = cv2.approxPolyDP(
        contour,
        1.5,
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
        return None

    commands = [
        f"M{points[0][0]} {points[0][1]}"
    ]

    for x, y in points[1:]:
        commands.append(
            f"L{x} {y}"
        )

    commands.append("Z")

    return {
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


# ============================================================
# LINE GEOMETRY
# ============================================================

def line_length(
    line: Line,
) -> float:

    x1, y1, x2, y2 = line

    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


def line_angle(
    line: Line,
) -> float:

    x1, y1, x2, y2 = line

    angle = math.degrees(
        math.atan2(
            y2 - y1,
            x2 - x1,
        )
    )

    angle %= 180.0

    if angle > 90:
        angle = 180 - angle

    return angle


def classify_orientation(
    line: Line,
    horizontal_tolerance: float = 6,
    vertical_tolerance: float = 6,
) -> str:

    angle = line_angle(
        line
    )

    if angle <= horizontal_tolerance:
        return "horizontal"

    if angle >= (
        90 - vertical_tolerance
    ):
        return "vertical"

    return "oblique"


def normalize_line(
    line: Line,
) -> Line:

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
# Generic presets only
# ============================================================

def detect_lines(
    image_gray: np.ndarray,
    mask: np.ndarray,
    min_length: int,
    edge_threshold: int,
) -> List[Line]:

    working = cv2.bilateralFilter(
        image_gray,
        7,
        35,
        35,
    )

    working = working.copy()

    working[
        mask == 0
    ] = 255

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

    detector = (
        cv2.ximgproc
        .createFastLineDetector(
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
    )

    detected = detector.detect(
        working
    )

    if detected is None:
        return []

    lines = []

    for row in detected:

        x1, y1, x2, y2 = row[0]

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

        mx = int(
            round(
                (x1 + x2) / 2.0
            )
        )

        my = int(
            round(
                (y1 + y2) / 2.0
            )
        )

        if not (
            0 <= mx < mask.shape[1]
            and
            0 <= my < mask.shape[0]
        ):
            continue

        if mask[
            my,
            mx,
        ] == 0:
            continue

        lines.append(
            normalize_line(
                line
            )
        )

    return lines


# ============================================================
# HORIZONTAL MERGING
# ============================================================

def merge_horizontal(
    lines: List[Line],
    y_tolerance: float,
    gap: float,
) -> List[Line]:

    candidates = []

    for x1, y1, x2, y2 in lines:

        if x1 > x2:
            x1, x2 = x2, x1

        y = (
            y1 + y2
        ) / 2.0

        candidates.append(
            (
                x1,
                y,
                x2,
                y,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[1],
            item[0],
        )
    )

    if not candidates:
        return []

    result = []

    current = list(
        candidates[0]
    )

    for item in candidates[1:]:

        x1, y, x2, _ = item

        same_row = (
            abs(
                y - current[1]
            )
            <= y_tolerance
        )

        line_gap = (
            x1 - current[2]
        )

        if (
            same_row
            and
            line_gap <= gap
        ):

            current[2] = max(
                current[2],
                x2,
            )

            current[1] = (
                current[1]
                + y
            ) / 2.0

            current[3] = current[1]

        else:

            result.append(
                tuple(
                    current
                )
            )

            current = list(
                item
            )

    result.append(
        tuple(
            current
        )
    )

    return result


# ============================================================
# VERTICAL MERGING
# ============================================================

def merge_vertical(
    lines: List[Line],
    x_tolerance: float,
    gap: float,
) -> List[Line]:

    candidates = []

    for x1, y1, x2, y2 in lines:

        if y1 > y2:
            y1, y2 = y2, y1

        x = (
            x1 + x2
        ) / 2.0

        candidates.append(
            (
                x,
                y1,
                x,
                y2,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    if not candidates:
        return []

    result = []

    current = list(
        candidates[0]
    )

    for item in candidates[1:]:

        x, y1, _, y2 = item

        same_column = (
            abs(
                x - current[0]
            )
            <= x_tolerance
        )

        line_gap = (
            y1 - current[3]
        )

        if (
            same_column
            and
            line_gap <= gap
        ):

            current[3] = max(
                current[3],
                y2,
            )

            current[0] = (
                current[0]
                + x
            ) / 2.0

            current[2] = current[0]

        else:

            result.append(
                tuple(
                    current
                )
            )

            current = list(
                item
            )

    result.append(
        tuple(
            current
        )
    )

    return result


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_lines(
    lines: List[Line],
    precision: int = 3,
) -> List[Line]:

    seen = set()
    output = []

    for line in lines:

        x1, y1, x2, y2 = line

        key = (
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

        if key in seen:
            continue

        seen.add(
            key
        )

        output.append(
            line
        )

    return output


# ============================================================
# PATH CREATION
# ============================================================

def line_to_path(
    line: Line,
    path_id: str,
    group: str,
    path_type: str,
    stroke_width: float = 0.8,
) -> Dict[str, Any]:

    x1, y1, x2, y2 = line

    return {
        "id": path_id,

        "d": (
            f"M{round(x1, 2)} {round(y1, 2)} "
            f"L{round(x2, 2)} {round(y2, 2)}"
        ),

        "fill": "none",
        "stroke": "#000000",

        "strokeWidth": stroke_width,

        "strokeLinecap": "round",
        "strokeLinejoin": "round",
        "vectorEffect": "non-scaling-stroke",

        "transform": None,

        "group": group,
        "type": path_type,

        "orientation":
            classify_orientation(
                line
            ),

        "lengthPx":
            round(
                line_length(
                    line
                ),
                2,
            ),
    }


# ============================================================
# GENERIC REGION TRACE
# ============================================================

def trace_region_lines(
    image_gray: np.ndarray,
    mask: np.ndarray,
    preset: Dict[str, Any],
    offset_x: int = 0,
    offset_y: int = 0,
    group: str = GROUP_GLOBAL,
) -> Tuple[
    List[Dict[str, Any]],
    List[Line],
]:

    raw = detect_lines(
        image_gray,
        mask,

        min_length=int(
            preset[
                "minLineLength"
            ]
        ),

        edge_threshold=int(
            preset[
                "edgeThreshold"
            ]
        ),
    )

    horizontal_raw = []
    vertical_raw = []
    oblique_raw = []

    for line in raw:

        orientation = (
            classify_orientation(
                line,

                horizontal_tolerance=float(
                    preset[
                        "horizontalTolerance"
                    ]
                ),

                vertical_tolerance=float(
                    preset[
                        "verticalTolerance"
                    ]
                ),
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

            if line_length(
                line
            ) >= (
                float(
                    preset[
                        "minLineLength"
                    ]
                )
                * 1.5
            ):

                oblique_raw.append(
                    line
                )

    horizontal = merge_horizontal(
        horizontal_raw,
        float(
            preset[
                "horizontalTolerance"
            ]
        ),
        float(
            preset[
                "maxGap"
            ]
        ),
    )

    vertical = merge_vertical(
        vertical_raw,
        float(
            preset[
                "verticalTolerance"
            ]
        ),
        float(
            preset[
                "maxGap"
            ]
        ),
    )

    horizontal = deduplicate_lines(
        horizontal
    )

    vertical = deduplicate_lines(
        vertical
    )

    oblique = deduplicate_lines(
        oblique_raw
    )

    local_lines = (
        horizontal
        + vertical
        + oblique
    )

    paths = []
    global_lines = []

    for i, line in enumerate(
        local_lines,
        start=1,
    ):

        x1, y1, x2, y2 = line

        global_line = (
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
        )

        global_lines.append(
            global_line
        )

        paths.append(
            line_to_path(
                global_line,
                f"{group}_line_{i}",
                group,
                "line",
                0.8,
            )
        )

    return (
        paths,
        global_lines,
    )


# ============================================================
# NEW v0.7.1
# PAINTED GRID EXTRACTION
# ============================================================

def adaptive_kernel_size(
    requested: int,
    roi_dimension: int,
    minimum: int,
    maximum_fraction: float,
) -> int:
    """
    Prevent fixed morphology kernels becoming absurdly
    large/small on differently sized ROIs.
    """

    maximum = max(
        minimum,
        int(
            roi_dimension
            * maximum_fraction
        ),
    )

    value = int(
        clamp(
            requested,
            minimum,
            maximum,
        )
    )

    # Odd values are generally safer for morphology.
    if value % 2 == 0:
        value += 1

    return value


def extract_component_lines(
    binary: np.ndarray,
    orientation: str,
    min_length: int,
    min_area: int,
) -> List[Line]:

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            (
                binary > 0
            ).astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    lines = []

    for label in range(
        1,
        count,
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP,
            ]
        )

        w = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH,
            ]
        )

        h = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        if orientation == "horizontal":

            if w < min_length:
                continue

            # Reject compact blobs.
            if w < (
                h * 1.5
            ):
                continue

            cy = float(
                centroids[
                    label
                ][1]
            )

            line = (
                float(x),
                cy,
                float(
                    x + w - 1
                ),
                cy,
            )

        else:

            if h < min_length:
                continue

            if h < (
                w * 1.3
            ):
                continue

            cx = float(
                centroids[
                    label
                ][0]
            )

            line = (
                cx,
                float(y),
                cx,
                float(
                    y + h - 1
                ),
            )

        lines.append(
            line
        )

    return lines


def trace_painted_grid(
    image_gray: np.ndarray,
    mask: np.ndarray,
    preset: Dict[str, Any],
    offset_x: int,
    offset_y: int,
    group: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Line],
    Dict[str, Any],
]:

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    working = cv2.bilateralFilter(
        image_gray,
        7,
        30,
        30,
    )

    working = apply_clahe(
        working
    )

    # Background to black because we use bright-feature
    # morphology inside the ROI.
    working_masked = np.zeros_like(
        working
    )

    working_masked[
        mask > 0
    ] = working[
        mask > 0
    ]


    # --------------------------------------------------------
    # ENHANCE LOCAL LIGHT FEATURES
    # --------------------------------------------------------

    # White top-hat enhances relatively bright painted bands
    # against a darker painted field.
    illumination_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (15, 15),
        )
    )

    top_hat = cv2.morphologyEx(
        working_masked,
        cv2.MORPH_TOPHAT,
        illumination_kernel,
    )

    # Blend enhanced local contrast back with original.
    enhanced = cv2.addWeighted(
        working_masked,
        0.45,
        top_hat,
        1.4,
        0,
    )


    # --------------------------------------------------------
    # ADAPTIVE KERNEL SIZES
    # --------------------------------------------------------

    roi_height, roi_width = (
        image_gray.shape[:2]
    )

    horizontal_kernel_width = (
        adaptive_kernel_size(
            int(
                preset.get(
                    "horizontalKernelPx",
                    25,
                )
            ),
            roi_width,
            minimum=9,
            maximum_fraction=0.10,
        )
    )

    horizontal_kernel_height = max(
        1,
        int(
            preset.get(
                "horizontalThicknessPx",
                3,
            )
        ),
    )

    vertical_kernel_height = (
        adaptive_kernel_size(
            int(
                preset.get(
                    "verticalKernelPx",
                    11,
                )
            ),
            roi_height,
            minimum=5,
            maximum_fraction=0.08,
        )
    )

    vertical_kernel_width = max(
        1,
        int(
            preset.get(
                "verticalThicknessPx",
                3,
            )
        ),
    )


    # --------------------------------------------------------
    # THRESHOLD BRIGHT STRUCTURE
    # --------------------------------------------------------

    _, bright_binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    bright_binary = cv2.bitwise_and(
        bright_binary,
        bright_binary,
        mask=mask,
    )


    # --------------------------------------------------------
    # HORIZONTAL MORPHOLOGY
    # --------------------------------------------------------

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_kernel_width,
                horizontal_kernel_height,
            ),
        )
    )

    horizontal_mask = (
        cv2.morphologyEx(
            bright_binary,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )
    )

    horizontal_close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(
                    3,
                    int(
                        preset.get(
                            "horizontalGap",
                            20,
                        )
                        // 2
                    ),
                ),
                1,
            ),
        )
    )

    horizontal_mask = (
        cv2.morphologyEx(
            horizontal_mask,
            cv2.MORPH_CLOSE,
            horizontal_close_kernel,
        )
    )


    # --------------------------------------------------------
    # VERTICAL MORPHOLOGY
    # --------------------------------------------------------

    vertical_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                vertical_kernel_width,
                vertical_kernel_height,
            ),
        )
    )

    vertical_mask = (
        cv2.morphologyEx(
            bright_binary,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )
    )

    vertical_close_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(
                    3,
                    int(
                        preset.get(
                            "verticalGap",
                            7,
                        )
                    ),
                ),
            ),
        )
    )

    vertical_mask = (
        cv2.morphologyEx(
            vertical_mask,
            cv2.MORPH_CLOSE,
            vertical_close_kernel,
        )
    )


    # --------------------------------------------------------
    # COMPONENT → CENTERLINE
    # --------------------------------------------------------

    minimum_component_area = int(
        preset.get(
            "minimumComponentArea",
            10,
        )
    )

    horizontal_lines = (
        extract_component_lines(
            horizontal_mask,
            "horizontal",

            min_length=int(
                preset.get(
                    "minimumHorizontalLength",
                    18,
                )
            ),

            min_area=(
                minimum_component_area
            ),
        )
    )

    vertical_lines = (
        extract_component_lines(
            vertical_mask,
            "vertical",

            min_length=int(
                preset.get(
                    "minimumVerticalLength",
                    7,
                )
            ),

            min_area=max(
                5,
                minimum_component_area
                // 2,
            ),
        )
    )


    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    horizontal_lines = merge_horizontal(
        horizontal_lines,

        y_tolerance=float(
            preset.get(
                "horizontalTolerance",
                5,
            )
        ),

        gap=float(
            preset.get(
                "horizontalGap",
                20,
            )
        ),
    )

    vertical_lines = merge_vertical(
        vertical_lines,

        x_tolerance=float(
            preset.get(
                "verticalTolerance",
                4,
            )
        ),

        gap=float(
            preset.get(
                "verticalGap",
                7,
            )
        ),
    )

    horizontal_lines = deduplicate_lines(
        horizontal_lines
    )

    vertical_lines = deduplicate_lines(
        vertical_lines
    )


    # --------------------------------------------------------
    # GLOBAL COORDINATES
    # --------------------------------------------------------

    local_lines = (
        horizontal_lines
        + vertical_lines
    )

    global_lines = []
    paths = []

    horizontal_count = 0
    vertical_count = 0

    for index, line in enumerate(
        local_lines,
        start=1,
    ):

        x1, y1, x2, y2 = line

        global_line = (
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
        )

        global_lines.append(
            global_line
        )

        orientation = (
            classify_orientation(
                global_line
            )
        )

        if orientation == "horizontal":
            horizontal_count += 1
            stroke_width = 0.9

        else:
            vertical_count += 1
            stroke_width = 0.75

        paths.append(
            line_to_path(
                global_line,

                f"{group}_grid_{index}",

                group,

                "painted-grid",

                stroke_width,
            )
        )


    diagnostics = {

        "detector":
            "painted_grid_morphology",

        "horizontalCount":
            horizontal_count,

        "verticalCount":
            vertical_count,

        "horizontalKernel":
            [
                horizontal_kernel_width,
                horizontal_kernel_height,
            ],

        "verticalKernel":
            [
                vertical_kernel_width,
                vertical_kernel_height,
            ],

        "brightBinary":
            image_to_base64_png(
                bright_binary
            ),

        "horizontalMask":
            image_to_base64_png(
                horizontal_mask
            ),

        "verticalMask":
            image_to_base64_png(
                vertical_mask
            ),
    }

    return (
        paths,
        global_lines,
        diagnostics,
    )


# ============================================================
# SKELETONIZATION
# ============================================================

def skeletonize(
    binary: np.ndarray,
) -> np.ndarray:

    binary = (
        binary > 0
    ).astype(
        np.uint8
    ) * 255

    if (
        hasattr(
            cv2,
            "ximgproc",
        )
        and hasattr(
            cv2.ximgproc,
            "thinning",
        )
    ):

        return cv2.ximgproc.thinning(
            binary,

            thinningType=(
                cv2.ximgproc
                .THINNING_ZHANGSUEN
            ),
        )

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

        residue = cv2.subtract(
            working,
            opened,
        )

        skeleton = cv2.bitwise_or(
            skeleton,
            residue,
        )

        working = eroded

        if cv2.countNonZero(
            working
        ) == 0:
            break

    return skeleton


# ============================================================
# IRREGULAR DETAIL
# ============================================================

def irregular_paths(
    image_gray: np.ndarray,
    mask: np.ndarray,
    existing_lines: List[Line],
    offset_x: int,
    offset_y: int,
    group: str,
    max_paths: int,
) -> List[Dict[str, Any]]:

    if max_paths <= 0:
        return []

    blurred = cv2.GaussianBlur(
        image_gray,
        (3, 3),
        0,
    )

    edges = cv2.Canny(
        blurred,
        25,
        75,
    )

    edges = cv2.bitwise_and(
        edges,
        edges,
        mask=mask,
    )

    line_mask = np.zeros_like(
        edges
    )

    for line in existing_lines:

        x1, y1, x2, y2 = line

        lx1 = int(
            round(
                x1 - offset_x
            )
        )

        ly1 = int(
            round(
                y1 - offset_y
            )
        )

        lx2 = int(
            round(
                x2 - offset_x
            )
        )

        ly2 = int(
            round(
                y2 - offset_y
            )
        )

        cv2.line(
            line_mask,
            (
                lx1,
                ly1,
            ),
            (
                lx2,
                ly2,
            ),
            255,
            5,
        )

    line_mask = cv2.dilate(
        line_mask,
        np.ones(
            (3, 3),
            np.uint8,
        ),
        iterations=1,
    )

    edges = cv2.bitwise_and(
        edges,
        cv2.bitwise_not(
            line_mask
        ),
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (
                edges > 0
            ).astype(
                np.uint8
            ),
            8,
        )
    )

    paths = []
    counter = 1

    for label in range(
        1,
        count,
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < 12:
            continue

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP,
            ]
        )

        w = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH,
            ]
        )

        h = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        if w <= 1 or h <= 1:
            continue

        crop = (
            labels[
                y:y + h,
                x:x + w,
            ]
            == label
        ).astype(
            np.uint8
        ) * 255

        skeleton = skeletonize(
            crop
        )

        contours, _ = cv2.findContours(
            skeleton,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        for contour in contours:

            if len(
                contour
            ) < 7:
                continue

            approx = cv2.approxPolyDP(
                contour,
                1.0,
                False,
            )

            points = [
                (
                    int(
                        p[0][0]
                    )
                    + x
                    + offset_x,

                    int(
                        p[0][1]
                    )
                    + y
                    + offset_y,
                )
                for p in approx
            ]

            if len(points) < 2:
                continue

            commands = [
                f"M{points[0][0]} {points[0][1]}"
            ]

            length = 0.0

            for index in range(
                1,
                len(points),
            ):

                px, py = points[
                    index
                ]

                previous_x, previous_y = points[
                    index - 1
                ]

                commands.append(
                    f"L{px} {py}"
                )

                length += math.hypot(
                    px - previous_x,
                    py - previous_y,
                )

            if length < 8:
                continue

            paths.append(
                {
                    "id":
                        f"{group}_irregular_{counter}",

                    "d":
                        " ".join(
                            commands
                        ),

                    "fill":
                        "none",

                    "stroke":
                        "#000000",

                    "strokeWidth":
                        0.7,

                    "strokeLinecap":
                        "round",

                    "strokeLinejoin":
                        "round",

                    "vectorEffect":
                        "non-scaling-stroke",

                    "transform":
                        None,

                    "group":
                        group,

                    "type":
                        "irregular-detail",

                    "orientation":
                        "irregular",

                    "lengthPx":
                        round(
                            length,
                            2,
                        ),
                }
            )

            counter += 1

            if len(
                paths
            ) >= max_paths:

                return paths

    return paths


# ============================================================
# ROI NORMALIZATION
# ============================================================

def normalize_roi(
    roi: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:

    x = int(
        roi.get(
            "x",
            0,
        )
    )

    y = int(
        roi.get(
            "y",
            0,
        )
    )

    width = int(
        roi.get(
            "width",
            0,
        )
    )

    height = int(
        roi.get(
            "height",
            0,
        )
    )

    x = clamp(
        x,
        0,
        image_width - 1,
    )

    y = clamp(
        y,
        0,
        image_height - 1,
    )

    width = clamp(
        width,
        1,
        image_width - x,
    )

    height = clamp(
        height,
        1,
        image_height - y,
    )

    preset = str(
        roi.get(
            "preset",
            "general_detail",
        )
    )

    if preset not in DEFAULT_PRESETS:

        preset = "general_detail"

    mode = str(
        roi.get(
            "mode",
            "replace",
        )
    ).lower()

    if mode not in (
        "replace",
        "add",
    ):

        mode = "replace"

    return {
        **roi,

        "id":
            str(
                roi.get(
                    "id",
                    "",
                )
            ),

        "x":
            x,

        "y":
            y,

        "width":
            width,

        "height":
            height,

        "preset":
            preset,

        "mode":
            mode,
    }


# ============================================================
# PATH / ROI TEST
# ============================================================

def path_inside_roi(
    path: Dict[str, Any],
    roi: Dict[str, Any],
) -> bool:

    values = re.findall(
        r"-?\d+(?:\.\d+)?",
        path.get(
            "d",
            "",
        ),
    )

    if len(values) < 2:
        return False

    numbers = [
        float(v)
        for v in values
    ]

    xs = numbers[
        0::2
    ]

    ys = numbers[
        1::2
    ]

    if not xs or not ys:
        return False

    center_x = (
        min(xs)
        + max(xs)
    ) / 2.0

    center_y = (
        min(ys)
        + max(ys)
    ) / 2.0

    return (
        roi["x"]
        <= center_x
        <= (
            roi["x"]
            + roi["width"]
        )
        and
        roi["y"]
        <= center_y
        <= (
            roi["y"]
            + roi["height"]
        )
    )


# ============================================================
# SVG
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
        f'stroke-width="{path.get("strokeWidth", 0.8)}" '
        f'stroke-linecap="round" '
        f'stroke-linejoin="round" '
        f'vector-effect="non-scaling-stroke" '
        f'/>'
    )


def group_to_svg(
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
    grouped_paths: Dict[
        str,
        List[
            Dict[
                str,
                Any,
            ]
        ],
    ],
) -> str:

    groups = []

    for group_id, paths in grouped_paths.items():

        groups.append(
            group_to_svg(
                group_id,
                paths,
            )
        )

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

def count_anchors(
    paths: List[Dict[str, Any]],
) -> int:

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
    paths: List[Dict[str, Any]],
):

    lengths = [
        float(
            path.get(
                "lengthPx",
                0,
            )
        )
        for path in paths
        if float(
            path.get(
                "lengthPx",
                0,
            )
        ) > 0
    ]

    if not lengths:

        return (
            0.0,
            0,
            0.0,
        )

    average = (
        sum(
            lengths
        )
        / len(
            lengths
        )
    )

    short_count = sum(
        1
        for value in lengths
        if value < 20
    )

    fragmentation = (
        short_count
        / len(
            lengths
        )
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
# HEALTH
# ============================================================

@app.get("/health")
def health():

    fast_line_ok = bool(
        hasattr(
            cv2,
            "ximgproc",
        )
        and hasattr(
            cv2.ximgproc,
            "createFastLineDetector",
        )
    )

    thinning_ok = bool(
        hasattr(
            cv2,
            "ximgproc",
        )
        and hasattr(
            cv2.ximgproc,
            "thinning",
        )
    )

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

    return {
        "status":
            "ok",

        "service":
            "vectorimage-worker",

        "provider":
            "opencv-potrace",

        "opencv":
            True,

        "opencvVersion":
            cv2.__version__,

        "opencvContrib":
            fast_line_ok
            and thinning_ok,

        "fastLineDetector":
            fast_line_ok,

        "thinning":
            thinning_ok,

        "potrace":
            potrace_ok,

        "potraceVersion":
            potrace_version,

        "pipeline":
            PIPELINE_NAME,

        "version":
            VERSION,

        "roiSupport":
            True,

        "paintedGridDetector":
            "morphology-v1",

        "roiPresets":
            list(
                DEFAULT_PRESETS.keys()
            ),
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

        # ====================================================
        # PREPROCESS
        # ====================================================

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        enhanced = apply_clahe(
            gray
        )

        subject_mask = (
            build_subject_mask(
                image_bgr
            )
        )

        ignore_background = bool(
            config.get(
                "ignoreBackgroundTexture",
                True,
            )
        )

        if not ignore_background:

            subject_mask[:] = 255


        # ====================================================
        # GROUP STORAGE
        # ====================================================

        grouped_paths = {}


        # ====================================================
        # OUTER CONTOUR
        # ====================================================

        outer = outer_contour_path(
            subject_mask
        )

        grouped_paths[
            GROUP_OUTER
        ] = (
            [outer]
            if outer
            else []
        )


        # ====================================================
        # GLOBAL TRACE
        # ====================================================

        global_preset = {

            "detector":
                "generic",

            "edgeThreshold":
                int(
                    config.get(
                        "edgeThreshold",
                        48,
                    )
                ),

            "minLineLength":
                int(
                    config.get(
                        "minPathLength",
                        20,
                    )
                ),

            "maxGap":
                int(
                    config.get(
                        "maxGapReconnect",
                        16,
                    )
                ),

            "horizontalTolerance":
                6,

            "verticalTolerance":
                6,

            "irregularDetail":
                False,

            "irregularCap":
                0,
        }

        (
            global_paths,
            global_geometry,
        ) = trace_region_lines(
            enhanced,
            subject_mask,
            global_preset,

            offset_x=0,
            offset_y=0,

            group=GROUP_GLOBAL,
        )

        grouped_paths[
            GROUP_GLOBAL
        ] = global_paths


        # ====================================================
        # ROI INPUT
        # ====================================================

        roi_input = config.get(
            "rois",
            [],
        )

        if not isinstance(
            roi_input,
            list,
        ):

            roi_input = []

        normalized_rois = []

        for roi_raw in roi_input:

            if not isinstance(
                roi_raw,
                dict,
            ):
                continue

            normalized_rois.append(
                normalize_roi(
                    roi_raw,
                    width,
                    height,
                )
            )


        # ====================================================
        # PROCESS ROIs
        # ====================================================

        roi_reports = []

        roi_algorithm_diagnostics = {}

        for index, roi in enumerate(
            normalized_rois,
            start=1,
        ):

            preset_name = roi[
                "preset"
            ]

            preset = dict(
                DEFAULT_PRESETS[
                    preset_name
                ]
            )

            roi_settings = roi.get(
                "settings",
                {},
            )

            if isinstance(
                roi_settings,
                dict,
            ):

                preset.update(
                    roi_settings
                )

            x = roi["x"]
            y = roi["y"]

            roi_width = roi[
                "width"
            ]

            roi_height = roi[
                "height"
            ]

            crop_gray = enhanced[
                y:y + roi_height,
                x:x + roi_width,
            ]

            crop_mask = subject_mask[
                y:y + roi_height,
                x:x + roi_width,
            ]

            group_id = (
                f"{GROUP_ROI_PREFIX}_"
                f"{index:02d}_"
                f"{preset_name}"
            )


            # =================================================
            # SPECIALIZED DISPATCH
            # =================================================

            detector_type = preset.get(
                "detector",
                "generic",
            )

            if detector_type == "painted_grid":

                (
                    roi_paths,
                    roi_geometry,
                    grid_diagnostics,
                ) = trace_painted_grid(
                    image_gray=crop_gray,

                    mask=crop_mask,

                    preset=preset,

                    offset_x=x,
                    offset_y=y,

                    group=group_id,
                )

                roi_algorithm_diagnostics[
                    group_id
                ] = grid_diagnostics

                irregular_count = 0

            else:

                (
                    roi_paths,
                    roi_geometry,
                ) = trace_region_lines(
                    crop_gray,
                    crop_mask,
                    preset,

                    offset_x=x,
                    offset_y=y,

                    group=group_id,
                )

                irregular_count = 0

                if bool(
                    preset.get(
                        "irregularDetail",
                        False,
                    )
                ):

                    irregular = (
                        irregular_paths(
                            image_gray=crop_gray,

                            mask=crop_mask,

                            existing_lines=(
                                roi_geometry
                            ),

                            offset_x=x,
                            offset_y=y,

                            group=group_id,

                            max_paths=int(
                                preset.get(
                                    "irregularCap",
                                    250,
                                )
                            ),
                        )
                    )

                    irregular_count = len(
                        irregular
                    )

                    roi_paths.extend(
                        irregular
                    )


            # =================================================
            # REPLACE GLOBAL GEOMETRY
            # =================================================

            if roi[
                "mode"
            ] == "replace":

                grouped_paths[
                    GROUP_GLOBAL
                ] = [
                    path
                    for path
                    in grouped_paths[
                        GROUP_GLOBAL
                    ]
                    if not path_inside_roi(
                        path,
                        roi,
                    )
                ]


            grouped_paths[
                group_id
            ] = roi_paths


            roi_reports.append(
                {
                    "id":
                        (
                            roi.get(
                                "id"
                            )
                            or
                            f"roi-{index}"
                        ),

                    "groupId":
                        group_id,

                    "preset":
                        preset_name,

                    "detector":
                        detector_type,

                    "mode":
                        roi[
                            "mode"
                        ],

                    "x":
                        x,

                    "y":
                        y,

                    "width":
                        roi_width,

                    "height":
                        roi_height,

                    "bounds": {
                        "x":
                            x,

                        "y":
                            y,

                        "width":
                            roi_width,

                        "height":
                            roi_height,
                    },

                    "pathCount":
                        len(
                            roi_paths
                        ),

                    "irregularPathCount":
                        irregular_count,
                }
            )


        # ====================================================
        # FINAL COLLECTION
        # ====================================================

        all_paths = []
        groups = []

        for group_id, paths in (
            grouped_paths.items()
        ):

            all_paths.extend(
                paths
            )

            groups.append(
                {
                    "id":
                        group_id,

                    "label":
                        group_id,

                    "paths":
                        paths,

                    "pathCount":
                        len(
                            paths
                        ),
                }
            )


        # ====================================================
        # SVG
        # ====================================================

        svg = build_svg(
            width,
            height,
            grouped_paths,
        )


        # ====================================================
        # STATS
        # ====================================================

        anchors = count_anchors(
            all_paths
        )

        (
            average_path_length,
            short_fragment_count,
            fragmentation_ratio,
        ) = path_statistics(
            all_paths
        )

        processing_ms = int(
            (
                time.time()
                - started
            )
            * 1000
        )

        global_final_count = len(
            grouped_paths.get(
                GROUP_GLOBAL,
                [],
            )
        )

        roi_path_total = sum(
            report[
                "pathCount"
            ]
            for report in roi_reports
        )

        warnings = []

        if len(
            all_paths
        ) > 3000:

            warnings.append(
                "Trace contains a high number of paths."
            )

        if fragmentation_ratio > 0.25:

            warnings.append(
                "Trace contains many short fragments."
            )


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        return_diagnostics = bool(
            config.get(
                "returnDiagnostics",
                True,
            )
        )

        diagnostics = {}

        if return_diagnostics:

            diagnostics = {

                "subjectMask":
                    image_to_base64_png(
                        subject_mask
                    ),

                "roiRequestCount":
                    len(
                        roi_input
                    ),

                "roiProcessedCount":
                    len(
                        roi_reports
                    ),

                "returnedGroupIds":
                    [
                        group[
                            "id"
                        ]
                        for group in groups
                    ],

                # NEW v0.7.1
                "roiAlgorithms":
                    roi_algorithm_diagnostics,
            }


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "success":
                True,

            "requestId":
                request_id,

            "provider":
                "opencv-potrace",

            "pipeline":
                PIPELINE_NAME,

            "workerVersion":
                VERSION,

            "version":
                VERSION,

            "width":
                width,

            "height":
                height,

            "viewBox":
                f"0 0 {width} {height}",

            "svg":
                svg,

            "paths":
                all_paths,

            "groups":
                groups,

            "rois":
                roi_reports,

            "statistics": {

                "pathCount":
                    len(
                        all_paths
                    ),

                "anchorCount":
                    anchors,

                "anchorCountEstimate":
                    anchors,

                "globalPathCount":
                    global_final_count,

                "roiCount":
                    len(
                        roi_reports
                    ),

                "roiPathCount":
                    roi_path_total,

                "averagePathLength":
                    average_path_length,

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

            "warnings":
                warnings,

            "diagnostics":
                diagnostics,

            "settingsUsed":
                config,
        }


    except Exception as exc:

        return make_error(
            500,
            "VECTORIZATION_FAILED",
            str(exc),
            request_id,
        )
