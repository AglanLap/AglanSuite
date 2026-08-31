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
    version="0.6.0",
)

API_TOKEN = os.environ.get(
    "VECTOR_WORKER_TOKEN",
    "",
).strip()

PIPELINE_NAME = "archaeological-hybrid-detail-v4"


# ============================================================
# GROUPS
# ============================================================

GROUP_OUTER = "01_Outer_Contour"
GROUP_STRUCTURAL = "02_Structural_Lines"
GROUP_FINE = "03_Fine_Line_Detail"
GROUP_IRREGULAR = "04_Irregular_Detail"


# ============================================================
# DEFAULT LIMITS
# ============================================================

DEFAULT_FINE_DETAIL_CAP = 700
DEFAULT_IRREGULAR_DETAIL_CAP = 450

FINE_MAX_DISTANCE_PX = 5.0
FINE_MAX_ANGLE_DIFF_DEG = 8.0
FINE_MIN_OVERLAP_RATIO = 0.60


# ============================================================
# TYPES
# ============================================================

Line = Tuple[
    float,
    float,
    float,
    float,
]


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
    line: Line,
) -> float:

    x1, y1, x2, y2 = line

    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


def line_angle_raw(
    line: Line,
) -> float:

    x1, y1, x2, y2 = line

    angle = math.degrees(
        math.atan2(
            y2 - y1,
            x2 - x1,
        )
    )

    return (
        angle % 180.0
    )


def line_angle_difference(
    a: Line,
    b: Line,
) -> float:

    angle_a = line_angle_raw(
        a
    )

    angle_b = line_angle_raw(
        b
    )

    difference = abs(
        angle_a
        - angle_b
    )

    return min(
        difference,
        180.0 - difference,
    )


def line_angle_degrees(
    line: Line,
) -> float:

    angle = line_angle_raw(
        line
    )

    if angle > 90:
        angle = 180 - angle

    return angle


def classify_orientation(
    line: Line,
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


def line_midpoint(
    line: Line,
) -> Tuple[float, float]:

    x1, y1, x2, y2 = line

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


# ============================================================
# FAST LINE DETECTOR
# ============================================================

def create_fast_line_detector(
    min_length: int,
    edge_threshold: int,
):

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
) -> List[Line]:

    filtered = cv2.bilateralFilter(
        enhanced,
        7,
        35,
        35,
    )

    working = filtered.copy()

    working[
        subject_mask == 0
    ] = 255

    detector = create_fast_line_detector(
        min_length,
        edge_threshold,
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
# STRUCTURAL MERGING
# ============================================================

def merge_horizontal_lines(
    lines: List[Line],
    y_tolerance: float,
    gap_tolerance: float,
) -> List[Line]:

    if not lines:
        return []

    normalized = []

    for x1, y1, x2, y2 in lines:

        if x1 > x2:
            x1, x2 = x2, x1

        y = (
            y1 + y2
        ) / 2.0

        normalized.append(
            (
                x1,
                y,
                x2,
                y,
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

        gap = (
            x1 - cx2
        )

        if (
            same_row
            and gap <= gap_tolerance
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
    lines: List[Line],
    x_tolerance: float,
    gap_tolerance: float,
) -> List[Line]:

    if not lines:
        return []

    normalized = []

    for x1, y1, x2, y2 in lines:

        if y1 > y2:
            y1, y2 = y2, y1

        x = (
            x1 + x2
        ) / 2.0

        normalized.append(
            (
                x,
                y1,
                x,
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

        gap = (
            y1 - cy2
        )

        if (
            same_column
            and gap <= gap_tolerance
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
# STRUCTURAL DUPLICATE TEST
# ============================================================

def point_to_line_distance(
    point: Tuple[float, float],
    line: Line,
) -> float:

    px, py = point

    x1, y1, x2, y2 = line

    dx = x2 - x1
    dy = y2 - y1

    denominator = math.hypot(
        dx,
        dy,
    )

    if denominator < 1e-6:

        return math.hypot(
            px - x1,
            py - y1,
        )

    numerator = abs(
        dy * px
        - dx * py
        + x2 * y1
        - y2 * x1
    )

    return (
        numerator
        / denominator
    )


def projection_interval(
    line: Line,
    origin: Tuple[float, float],
    axis: Tuple[float, float],
):

    x1, y1, x2, y2 = line

    ox, oy = origin
    ux, uy = axis

    p1 = (
        (x1 - ox) * ux
        + (y1 - oy) * uy
    )

    p2 = (
        (x2 - ox) * ux
        + (y2 - oy) * uy
    )

    return (
        min(
            p1,
            p2,
        ),
        max(
            p1,
            p2,
        ),
    )


def line_overlap_ratio(
    fine_line: Line,
    structural_line: Line,
) -> float:

    sx1, sy1, sx2, sy2 = (
        structural_line
    )

    dx = sx2 - sx1
    dy = sy2 - sy1

    structural_length = math.hypot(
        dx,
        dy,
    )

    if structural_length < 1e-6:
        return 0.0

    ux = (
        dx
        / structural_length
    )

    uy = (
        dy
        / structural_length
    )

    origin = (
        sx1,
        sy1,
    )

    fine_min, fine_max = (
        projection_interval(
            fine_line,
            origin,
            (
                ux,
                uy,
            ),
        )
    )

    struct_min, struct_max = (
        projection_interval(
            structural_line,
            origin,
            (
                ux,
                uy,
            ),
        )
    )

    overlap = max(
        0,
        min(
            fine_max,
            struct_max,
        )
        - max(
            fine_min,
            struct_min,
        ),
    )

    fine_length = max(
        1e-6,
        fine_max
        - fine_min,
    )

    return clamp(
        overlap
        / fine_length,
        0,
        1,
    )


def is_duplicate_of_structural(
    fine_line: Line,
    structural_lines: List[Line],
) -> bool:

    midpoint = line_midpoint(
        fine_line
    )

    for structural_line in structural_lines:

        if (
            line_angle_difference(
                fine_line,
                structural_line,
            )
            > FINE_MAX_ANGLE_DIFF_DEG
        ):
            continue

        if (
            point_to_line_distance(
                midpoint,
                structural_line,
            )
            > FINE_MAX_DISTANCE_PX
        ):
            continue

        if (
            line_overlap_ratio(
                fine_line,
                structural_line,
            )
            >= FINE_MIN_OVERLAP_RATIO
        ):
            return True

    return False


# ============================================================
# LINES → PATH RECORDS
# ============================================================

def line_to_path_record(
    line: Line,
    path_id: str,
    group: str,
    path_type: str,
    orientation: str,
    stroke_width: float,
):

    x1, y1, x2, y2 = line

    return {
        "id": path_id,

        "d": (
            f"M{round(x1, 2)} "
            f"{round(y1, 2)} "
            f"L{round(x2, 2)} "
            f"{round(y2, 2)}"
        ),

        "fill": "none",
        "stroke": "#000000",

        "strokeWidth":
            stroke_width,

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
            path_type,

        "orientation":
            orientation,

        "lengthPx":
            round(
                line_length(
                    line
                ),
                2,
            ),
    }


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

    # Fallback
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
# SKELETON → POLYLINES
# ============================================================

NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def pixel_neighbors(
    point,
    pixels,
):

    y, x = point

    output = []

    for dy, dx in NEIGHBORS:

        candidate = (
            y + dy,
            x + dx,
        )

        if candidate in pixels:

            output.append(
                candidate
            )

    return output


def skeleton_to_polylines(
    skeleton: np.ndarray,
    min_pixels: int,
) -> List[List[Tuple[int, int]]]:

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

    degree = {
        p: len(
            pixel_neighbors(
                p,
                pixels,
            )
        )
        for p in pixels
    }

    endpoints = {
        p
        for p, d
        in degree.items()
        if d != 2
    }

    visited = set()
    lines = []

    def edge_key(
        a,
        b,
    ):
        return tuple(
            sorted(
                (
                    a,
                    b,
                )
            )
        )

    def save(
        sequence,
    ):

        if len(
            sequence
        ) < min_pixels:
            return

        lines.append(
            [
                (
                    x,
                    y,
                )
                for y, x
                in sequence
            ]
        )

    for start in endpoints:

        for neighbor in pixel_neighbors(
            start,
            pixels,
        ):

            key = edge_key(
                start,
                neighbor,
            )

            if key in visited:
                continue

            visited.add(
                key
            )

            sequence = [
                start,
                neighbor,
            ]

            previous = start
            current = neighbor

            while True:

                if (
                    current in endpoints
                    and current != start
                ):
                    break

                candidates = [
                    p
                    for p in pixel_neighbors(
                        current,
                        pixels,
                    )
                    if p != previous
                ]

                next_point = None

                for candidate in candidates:

                    key = edge_key(
                        current,
                        candidate,
                    )

                    if key not in visited:

                        next_point = (
                            candidate
                        )

                        break

                if next_point is None:
                    break

                visited.add(
                    edge_key(
                        current,
                        next_point,
                    )
                )

                sequence.append(
                    next_point
                )

                previous = current
                current = next_point

            save(
                sequence
            )

    return lines


# ============================================================
# IRREGULAR DETAIL DETECTOR
# ============================================================

def create_irregular_detail_mask(
    enhanced: np.ndarray,
    subject_mask: np.ndarray,
    straight_lines: List[Line],
    edge_threshold: int,
) -> np.ndarray:

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0,
    )

    low = clamp(
        int(
            edge_threshold
            * 0.45
        ),
        8,
        160,
    )

    high = clamp(
        int(
            edge_threshold
            * 1.35
        ),
        low + 1,
        255,
    )

    edges = cv2.Canny(
        blurred,
        low,
        high,
    )

    edges = cv2.bitwise_and(
        edges,
        edges,
        mask=subject_mask,
    )

    # Build mask of already detected straight geometry.
    straight_mask = np.zeros_like(
        edges
    )

    for line in straight_lines:

        x1, y1, x2, y2 = map(
            int,
            line,
        )

        cv2.line(
            straight_mask,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            255,
            5,
        )

    straight_mask = cv2.dilate(
        straight_mask,
        np.ones(
            (3, 3),
            np.uint8,
        ),
        iterations=1,
    )

    residual = cv2.bitwise_and(
        edges,
        cv2.bitwise_not(
            straight_mask
        ),
    )

    residual = cv2.morphologyEx(
        residual,
        cv2.MORPH_CLOSE,
        np.ones(
            (2, 2),
            np.uint8,
        ),
        iterations=1,
    )

    return residual


def extract_irregular_paths(
    residual_mask: np.ndarray,
    min_component_area: int,
    min_line_pixels: int,
    simplification: float,
    max_paths: int,
) -> Tuple[
    List[Dict[str, Any]],
    np.ndarray,
]:

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (
                residual_mask > 0
            ).astype(
                np.uint8
            ),
            8,
        )
    )

    paths = []

    combined_skeleton = (
        np.zeros_like(
            residual_mask
        )
    )

    path_counter = 1

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

        if area < min_component_area:
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

        width = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        # Suppress tiny compact photographic spots.
        longest = max(
            width,
            height,
        )

        shortest = max(
            1,
            min(
                width,
                height,
            ),
        )

        elongation = (
            longest
            / shortest
        )

        box_area = max(
            1,
            width
            * height,
        )

        density = (
            area
            / box_area
        )

        if (
            elongation < 1.3
            and density > 0.65
            and area < 60
        ):
            continue

        padding = 2

        x0 = max(
            0,
            x - padding,
        )

        y0 = max(
            0,
            y - padding,
        )

        x1 = min(
            residual_mask.shape[1],
            x + width + padding,
        )

        y1 = min(
            residual_mask.shape[0],
            y + height + padding,
        )

        component = (
            labels[
                y0:y1,
                x0:x1,
            ]
            == label
        ).astype(
            np.uint8
        ) * 255

        skeleton = skeletonize(
            component
        )

        combined_skeleton[
            y0:y1,
            x0:x1,
        ] = cv2.bitwise_or(
            combined_skeleton[
                y0:y1,
                x0:x1,
            ],
            skeleton,
        )

        polylines = (
            skeleton_to_polylines(
                skeleton,
                min_pixels=(
                    min_line_pixels
                ),
            )
        )

        for polyline in polylines:

            if len(
                paths
            ) >= max_paths:
                break

            global_points = [
                (
                    px + x0,
                    py + y0,
                )
                for px, py
                in polyline
            ]

            contour = np.asarray(
                global_points,
                dtype=np.float32,
            ).reshape(
                -1,
                1,
                2,
            )

            epsilon = max(
                0.5,
                simplification
                * 4.0,
            )

            simplified = cv2.approxPolyDP(
                contour,
                epsilon,
                False,
            )

            points = [
                (
                    int(
                        p[0][0]
                    ),
                    int(
                        p[0][1]
                    ),
                )
                for p in simplified
            ]

            if len(
                points
            ) < 2:
                continue

            commands = [
                f"M{points[0][0]} {points[0][1]}"
            ]

            for px, py in points[1:]:

                commands.append(
                    f"L{px} {py}"
                )

            length = 0.0

            for i in range(
                1,
                len(points),
            ):

                x_a, y_a = points[
                    i - 1
                ]

                x_b, y_b = points[
                    i
                ]

                length += math.hypot(
                    x_b - x_a,
                    y_b - y_a,
                )

            if (
                length
                < min_line_pixels
            ):
                continue

            paths.append(
                {
                    "id": (
                        f"irregular_"
                        f"{path_counter}"
                    ),

                    "d": " ".join(
                        commands
                    ),

                    "fill": "none",

                    "stroke":
                        "#000000",

                    "strokeWidth":
                        0.75,

                    "strokeLinecap":
                        "round",

                    "strokeLinejoin":
                        "round",

                    "vectorEffect":
                        "non-scaling-stroke",

                    "transform":
                        None,

                    "group":
                        GROUP_IRREGULAR,

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

            path_counter += 1

        if len(
            paths
        ) >= max_paths:
            break

    return (
        paths,
        combined_skeleton,
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
    width,
    height,
    outer,
    structural,
    fine,
    irregular,
):

    return f"""<svg
xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">
{svg_group(GROUP_OUTER, outer)}
{svg_group(GROUP_STRUCTURAL, structural)}
{svg_group(GROUP_FINE, fine)}
{svg_group(GROUP_IRREGULAR, irregular)}
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
        if float(
            path.get(
                "lengthPx",
                0,
            )
        ) > 0
    ]

    if not lengths:
        return (
            0,
            0,
            0,
        )

    average = (
        sum(
            lengths
        )
        / len(
            lengths
        )
    )

    short = sum(
        1
        for value in lengths
        if value < 20
    )

    ratio = (
        short
        / len(
            lengths
        )
    )

    return (
        round(
            average,
            2,
        ),
        short,
        round(
            ratio,
            4,
        ),
    )


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

        potrace_version = (
            "Unavailable"
        )

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
            "0.6.0",

        "passes": [
            GROUP_OUTER,
            GROUP_STRUCTURAL,
            GROUP_FINE,
            GROUP_IRREGULAR,
        ],
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

    try:

        config = json.loads(
            settings
        )

    except Exception:

        config = {}

    edge_threshold = int(
        config.get(
            "edgeThreshold",
            48,
        )
    )

    line_sensitivity = int(
        config.get(
            "lineSensitivity",
            88,
        )
    )

    min_path_length = int(
        config.get(
            "minPathLength",
            20,
        )
    )

    max_gap = int(
        config.get(
            "maxGapReconnect",
            16,
        )
    )

    path_simplification = float(
        config.get(
            "pathSimplification",
            0.05,
        )
    )

    detect_internal_lines = bool(
        config.get(
            "detectInternalLines",
            True,
        )
    )

    preserve_small_details = bool(
        config.get(
            "preserveSmallDetails",
            True,
        )
    )

    ignore_background_texture = bool(
        config.get(
            "ignoreBackgroundTexture",
            True,
        )
    )

    fine_detail_cap = int(
        config.get(
            "fineDetailCap",
            DEFAULT_FINE_DETAIL_CAP,
        )
    )

    irregular_detail_cap = int(
        config.get(
            "irregularDetailCap",
            DEFAULT_IRREGULAR_DETAIL_CAP,
        )
    )

    return_diagnostics = bool(
        config.get(
            "returnDiagnostics",
            True,
        )
    )

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

        if not ignore_background_texture:
            subject_mask[:] = 255


        # ====================================================
        # PASS 1 — OUTER
        # ====================================================

        outer_paths = (
            extract_outer_contour(
                subject_mask,
                path_simplification,
            )
        )


        # ====================================================
        # PASS 2 — STRUCTURAL
        # ====================================================

        structural_raw = (
            detect_line_segments(
                enhanced,
                subject_mask,
                max(
                    20,
                    min_path_length,
                ),
                edge_threshold,
            )
        )

        horizontal_raw = []
        vertical_raw = []
        oblique_raw = []

        for line in structural_raw:

            orientation = (
                classify_orientation(
                    line
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
                ) >= max(
                    45,
                    min_path_length
                    * 1.5,
                ):

                    oblique_raw.append(
                        line
                    )

        horizontal = (
            merge_horizontal_lines(
                horizontal_raw,
                4.0,
                max_gap,
            )
        )

        vertical = (
            merge_vertical_lines(
                vertical_raw,
                4.0,
                max_gap,
            )
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

        structural_geometry = (
            horizontal
            + vertical
            + oblique
        )

        structural_paths = []

        for index, line in enumerate(
            structural_geometry,
            start=1,
        ):

            structural_paths.append(
                line_to_path_record(
                    line=line,

                    path_id=(
                        f"struct_{index}"
                    ),

                    group=(
                        GROUP_STRUCTURAL
                    ),

                    path_type=(
                        "structural"
                    ),

                    orientation=(
                        classify_orientation(
                            line
                        )
                    ),

                    stroke_width=1.0,
                )
            )


        # ====================================================
        # PASS 3 — FINE STRAIGHT DETAIL
        # ====================================================

        fine_paths = []

        fine_geometry = []

        fine_duplicates_removed = 0

        if detect_internal_lines:

            fine_raw = (
                detect_line_segments(
                    enhanced,
                    subject_mask,
                    max(
                        10,
                        int(
                            min_path_length
                            * 0.7
                        ),
                    ),
                    clamp(
                        int(
                            75
                            - line_sensitivity
                            * 0.35
                        ),
                        30,
                        70,
                    ),
                )
            )

            fine_raw = (
                deduplicate_lines(
                    fine_raw
                )
            )

            for line in fine_raw:

                if (
                    is_duplicate_of_structural(
                        line,
                        structural_geometry,
                    )
                ):

                    fine_duplicates_removed += 1

                    continue

                fine_geometry.append(
                    line
                )

            fine_geometry.sort(
                key=line_length,
                reverse=True,
            )

            fine_geometry = (
                fine_geometry[
                    :fine_detail_cap
                ]
            )

            for index, line in enumerate(
                fine_geometry,
                start=1,
            ):

                fine_paths.append(
                    line_to_path_record(
                        line=line,

                        path_id=(
                            f"fine_{index}"
                        ),

                        group=GROUP_FINE,

                        path_type=(
                            "fine-detail"
                        ),

                        orientation=(
                            classify_orientation(
                                line
                            )
                        ),

                        stroke_width=0.7,
                    )
                )


        # ====================================================
        # PASS 4 — IRREGULAR DETAIL
        # ====================================================

        all_straight_geometry = (
            structural_geometry
            + fine_geometry
        )

        irregular_mask = (
            create_irregular_detail_mask(
                enhanced=enhanced,

                subject_mask=(
                    subject_mask
                ),

                straight_lines=(
                    all_straight_geometry
                ),

                edge_threshold=(
                    edge_threshold
                ),
            )
        )

        irregular_paths, (
            irregular_skeleton
        ) = extract_irregular_paths(
            residual_mask=(
                irregular_mask
            ),

            min_component_area=12,

            min_line_pixels=max(
                6,
                min_path_length // 2,
            ),

            simplification=max(
                0.04,
                path_simplification,
            ),

            max_paths=(
                irregular_detail_cap
            ),
        )


        # ====================================================
        # COMBINE
        # ====================================================

        all_paths = (
            outer_paths
            + structural_paths
            + fine_paths
            + irregular_paths
        )

        if not all_paths:

            return make_error(
                422,
                "NO_PATHS_GENERATED",
                "No usable vector paths were generated",
                request_id,
            )

        svg = build_svg(
            width,
            height,
            outer_paths,
            structural_paths,
            fine_paths,
            irregular_paths,
        )


        # ====================================================
        # STATISTICS
        # ====================================================

        anchors = count_anchors(
            all_paths
        )

        (
            average_path_length,
            short_fragment_count,
            fragmentation_ratio,
        ) = path_statistics(
            structural_paths
            + fine_paths
            + irregular_paths
        )

        processing_ms = int(
            (
                time.time()
                - started
            )
            * 1000
        )

        warnings = []

        if len(
            all_paths
        ) > 2500:

            warnings.append(
                "Trace contains a high number of paths."
            )

        if fragmentation_ratio > 0.20:

            warnings.append(
                "Trace contains many short fragments."
            )


        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        diagnostics = {}

        if return_diagnostics:

            diagnostics = {
                "subjectMask":
                    image_to_base64_png(
                        subject_mask
                    ),

                "enhancedGray":
                    image_to_base64_png(
                        enhanced
                    ),

                "irregularResidualMask":
                    image_to_base64_png(
                        irregular_mask
                    ),

                "irregularSkeleton":
                    image_to_base64_png(
                        irregular_skeleton
                    ),
            }


        # ====================================================
        # GROUPS
        # ====================================================

        groups = [
            {
                "id":
                    GROUP_OUTER,

                "label":
                    "Outer Contour",

                "paths":
                    outer_paths,
            },

            {
                "id":
                    GROUP_STRUCTURAL,

                "label":
                    "Structural Lines",

                "paths":
                    structural_paths,
            },

            {
                "id":
                    GROUP_FINE,

                "label":
                    "Fine Line Detail",

                "paths":
                    fine_paths,
            },

            {
                "id":
                    GROUP_IRREGULAR,

                "label":
                    "Irregular Detail",

                "paths":
                    irregular_paths,
            },
        ]


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
                "0.6.0",

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

            "statistics": {

                "pathCount":
                    len(
                        all_paths
                    ),

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

                "fineDetailPathCount":
                    len(
                        fine_paths
                    ),

                "irregularDetailPathCount":
                    len(
                        irregular_paths
                    ),

                "fineDuplicatesRemoved":
                    fine_duplicates_removed,

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

                "detectInternalLines":
                    detect_internal_lines,

                "preserveSmallDetails":
                    preserve_small_details,

                "ignoreBackgroundTexture":
                    ignore_background_texture,

                "fineDetailCap":
                    fine_detail_cap,

                "irregularDetailCap":
                    irregular_detail_cap,

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
