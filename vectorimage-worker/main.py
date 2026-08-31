import os
import json
import time
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
    version="0.4.0",
)

API_TOKEN = os.environ.get(
    "VECTOR_WORKER_TOKEN",
    "",
).strip()


# ============================================================
# CONSTANTS
# ============================================================

GROUP_OUTER = "01_Outer_Contour"
GROUP_STRUCTURAL = "02_Structural_Lines"
GROUP_FINE = "03_Fine_Detail"

PIPELINE_NAME = "archaeological-multipass-v2"


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
        min(maximum, value),
    )


def np_image_to_base64_png(
    image: np.ndarray,
) -> str:
    if image is None:
        return ""

    success, buffer = cv2.imencode(
        ".png",
        image,
    )

    if not success:
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
        clipLimit=2.5,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


# ============================================================
# SUBJECT MASK
# ============================================================

def build_subject_mask(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Isolate the archaeological object from a
    predominantly dark photographic background.

    The largest foreground component is treated as
    the subject.
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

    _, threshold = cv2.threshold(
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

    cleaned = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        cleaned,
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

    # Slight expansion keeps features close to
    # damaged object edges.
    expand_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7),
    )

    mask = cv2.dilate(
        mask,
        expand_kernel,
        iterations=1,
    )

    return mask


# ============================================================
# POLYLINE / SVG HELPERS
# ============================================================

def simplify_points(
    points: List[Tuple[int, int]],
    epsilon: float,
    closed: bool = False,
) -> List[Tuple[int, int]]:
    if len(points) < 3:
        return points

    contour = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    approximation = cv2.approxPolyDP(
        contour,
        epsilon,
        closed,
    )

    return [
        (
            int(p[0][0]),
            int(p[0][1]),
        )
        for p in approximation
    ]


def polyline_length(
    points: List[Tuple[int, int]],
) -> float:
    if len(points) < 2:
        return 0.0

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


def points_to_svg_path(
    points: List[Tuple[int, int]],
    closed: bool = False,
) -> str:
    if len(points) < 2:
        return ""

    commands = [
        f"M{points[0][0]} {points[0][1]}"
    ]

    for x, y in points[1:]:
        commands.append(
            f"L{x} {y}"
        )

    if closed:
        commands.append("Z")

    return " ".join(
        commands
    )


def create_path_record(
    path_id: str,
    points: List[Tuple[int, int]],
    group: str,
    path_type: str,
    stroke_width: float,
    closed: bool = False,
    orientation: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if len(points) < 2:
        return None

    d = points_to_svg_path(
        points,
        closed=closed,
    )

    if not d:
        return None

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
        "group": group,
        "type": path_type,
        "orientation": orientation,
        "lengthPx": round(
            polyline_length(points),
            2,
        ),
    }


# ============================================================
# OUTER CONTOUR PASS
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

    raw_points = [
        (
            int(point[0][0]),
            int(point[0][1]),
        )
        for point in contour
    ]

    epsilon = max(
        1.0,
        simplification * 10.0,
    )

    points = simplify_points(
        raw_points,
        epsilon=epsilon,
        closed=True,
    )

    path = create_path_record(
        path_id="outer_1",
        points=points,
        group=GROUP_OUTER,
        path_type="outer-contour",
        stroke_width=1.4,
        closed=True,
        orientation=None,
    )

    return (
        [path]
        if path
        else []
    )


# ============================================================
# THINNING
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
        hasattr(cv2, "ximgproc")
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

    # Morphological fallback.
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

NEIGHBOURS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def get_neighbours(
    point,
    pixels,
):
    y, x = point

    neighbours = []

    for dy, dx in NEIGHBOURS:
        candidate = (
            y + dy,
            x + dx,
        )

        if candidate in pixels:
            neighbours.append(
                candidate
            )

    return neighbours


def edge_key(
    a,
    b,
):
    return tuple(
        sorted(
            [a, b]
        )
    )


def skeleton_to_polylines(
    skeleton: np.ndarray,
    min_pixels: int,
    offset_x: int = 0,
    offset_y: int = 0,
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
        point: len(
            get_neighbours(
                point,
                pixels,
            )
        )
        for point in pixels
    }

    important = {
        point
        for point, value
        in degree.items()
        if value != 2
    }

    visited = set()
    lines = []

    def save_line(
        line,
    ):
        if len(line) < min_pixels:
            return

        converted = [
            (
                x + offset_x,
                y + offset_y,
            )
            for y, x in line
        ]

        lines.append(
            converted
        )

    # Endpoints and junctions.
    for start in important:
        neighbours = get_neighbours(
            start,
            pixels,
        )

        for neighbour in neighbours:
            first_edge = edge_key(
                start,
                neighbour,
            )

            if first_edge in visited:
                continue

            visited.add(
                first_edge
            )

            line = [
                start,
                neighbour,
            ]

            previous = start
            current = neighbour

            while True:
                if (
                    current in important
                    and current != start
                ):
                    break

                candidates = [
                    candidate
                    for candidate
                    in get_neighbours(
                        current,
                        pixels,
                    )
                    if candidate != previous
                ]

                if not candidates:
                    break

                next_point = None

                for candidate in candidates:
                    candidate_edge = (
                        edge_key(
                            current,
                            candidate,
                        )
                    )

                    if (
                        candidate_edge
                        not in visited
                    ):
                        next_point = candidate
                        break

                if next_point is None:
                    break

                visited.add(
                    edge_key(
                        current,
                        next_point,
                    )
                )

                line.append(
                    next_point
                )

                previous = current
                current = next_point

            save_line(
                line
            )

    # Closed loops.
    for start in pixels:
        available = []

        for neighbour in get_neighbours(
            start,
            pixels,
        ):
            candidate_edge = edge_key(
                start,
                neighbour,
            )

            if candidate_edge not in visited:
                available.append(
                    neighbour
                )

        if not available:
            continue

        neighbour = available[0]

        line = [
            start,
            neighbour,
        ]

        visited.add(
            edge_key(
                start,
                neighbour,
            )
        )

        previous = start
        current = neighbour

        while True:
            candidates = [
                candidate
                for candidate
                in get_neighbours(
                    current,
                    pixels,
                )
                if candidate != previous
            ]

            next_point = None

            for candidate in candidates:
                candidate_edge = edge_key(
                    current,
                    candidate,
                )

                if candidate_edge not in visited:
                    next_point = candidate
                    break

            if next_point is None:
                break

            visited.add(
                edge_key(
                    current,
                    next_point,
                )
            )

            line.append(
                next_point
            )

            previous = current
            current = next_point

            if current == start:
                break

        save_line(
            line
        )

    return lines


# ============================================================
# COMPONENT FILTERING
# ============================================================

def connected_component_boxes(
    binary: np.ndarray,
):
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (
                binary > 0
            ).astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    components = []

    for label in range(
        1,
        count,
    ):
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

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        components.append(
            (
                label,
                x,
                y,
                width,
                height,
                area,
            )
        )

    return labels, components


def component_is_line_like(
    width: int,
    height: int,
    area: int,
    min_area: int,
    allow_compact: bool = False,
) -> bool:
    if area < min_area:
        return False

    if width <= 0 or height <= 0:
        return False

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
        longest / shortest
    )

    box_area = (
        width * height
    )

    density = (
        area / box_area
        if box_area
        else 1.0
    )

    if allow_compact:
        return (
            area >= min_area
            and density < 0.8
        )

    return (
        elongation >= 1.5
        or (
            area >= min_area * 3
            and density < 0.45
        )
    )


# ============================================================
# COMPONENT → CENTERLINES
# ============================================================

def component_centerlines(
    binary: np.ndarray,
    min_component_area: int,
    min_line_pixels: int,
    simplification: float,
    prefix: str,
    group: str,
    path_type: str,
    orientation: Optional[str],
    stroke_width: float,
    allow_compact: bool = False,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:

    labels, components = (
        connected_component_boxes(
            binary
        )
    )

    all_paths = []

    debug_skeleton = np.zeros_like(
        binary
    )

    counter = 1

    for (
        label,
        x,
        y,
        width,
        height,
        area,
    ) in components:

        if not component_is_line_like(
            width=width,
            height=height,
            area=area,
            min_area=min_component_area,
            allow_compact=allow_compact,
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
            binary.shape[1],
            x + width + padding,
        )

        y1 = min(
            binary.shape[0],
            y + height + padding,
        )

        label_crop = labels[
            y0:y1,
            x0:x1,
        ]

        component = (
            label_crop == label
        ).astype(
            np.uint8
        ) * 255

        skeleton = skeletonize(
            component
        )

        debug_skeleton[
            y0:y1,
            x0:x1,
        ] = cv2.bitwise_or(
            debug_skeleton[
                y0:y1,
                x0:x1,
            ],
            skeleton,
        )

        polylines = (
            skeleton_to_polylines(
                skeleton=skeleton,
                min_pixels=min_line_pixels,
                offset_x=x0,
                offset_y=y0,
            )
        )

        epsilon = max(
            0.5,
            simplification * 5.0,
        )

        for polyline in polylines:
            simplified = simplify_points(
                polyline,
                epsilon=epsilon,
                closed=False,
            )

            if (
                polyline_length(
                    simplified
                )
                < min_line_pixels
            ):
                continue

            path = create_path_record(
                path_id=(
                    f"{prefix}_{counter}"
                ),
                points=simplified,
                group=group,
                path_type=path_type,
                stroke_width=stroke_width,
                closed=False,
                orientation=orientation,
            )

            if path:
                all_paths.append(
                    path
                )

                counter += 1

    return (
        all_paths,
        debug_skeleton,
    )


# ============================================================
# EDGE DETECTION
# ============================================================

def make_canny_edges(
    enhanced: np.ndarray,
    subject_mask: np.ndarray,
    edge_threshold: int,
) -> np.ndarray:

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0,
    )

    low = clamp(
        int(edge_threshold * 0.45),
        5,
        200,
    )

    high = clamp(
        int(edge_threshold * 1.45),
        low + 1,
        255,
    )

    edges = cv2.Canny(
        blurred,
        low,
        high,
    )

    return cv2.bitwise_and(
        edges,
        edges,
        mask=subject_mask,
    )


# ============================================================
# STRUCTURAL SUBPASSES
# ============================================================

def horizontal_structure_mask(
    edges: np.ndarray,
    max_gap: int,
) -> np.ndarray:

    gap = clamp(
        max_gap,
        3,
        30,
    )

    reconnect_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                gap,
                1,
            ),
        )
    )

    reconnected = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        reconnect_kernel,
        iterations=1,
    )

    # Reject features with almost no horizontal
    # extent.
    selector_length = max(
        7,
        gap,
    )

    selector = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                selector_length,
                1,
            ),
        )
    )

    horizontal = cv2.morphologyEx(
        reconnected,
        cv2.MORPH_OPEN,
        selector,
        iterations=1,
    )

    return horizontal


def vertical_structure_mask(
    edges: np.ndarray,
    max_gap: int,
) -> np.ndarray:

    gap = clamp(
        max_gap,
        3,
        30,
    )

    reconnect_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                gap,
            ),
        )
    )

    reconnected = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        reconnect_kernel,
        iterations=1,
    )

    selector_length = max(
        7,
        gap,
    )

    selector = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                selector_length,
            ),
        )
    )

    vertical = cv2.morphologyEx(
        reconnected,
        cv2.MORPH_OPEN,
        selector,
        iterations=1,
    )

    return vertical


def irregular_structure_mask(
    edges: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
) -> np.ndarray:

    known_structures = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    # Slight dilation prevents the irregular pass
    # from tracing the same structural feature again.
    exclusion_kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    exclusion = cv2.dilate(
        known_structures,
        exclusion_kernel,
        iterations=1,
    )

    irregular = cv2.bitwise_and(
        edges,
        cv2.bitwise_not(
            exclusion
        ),
    )

    # Reconnect only very small local breaks.
    irregular = cv2.morphologyEx(
        irregular,
        cv2.MORPH_CLOSE,
        np.ones(
            (2, 2),
            np.uint8,
        ),
        iterations=1,
    )

    return irregular


# ============================================================
# FINE DETAIL PASS
# ============================================================

def fine_detail_mask(
    enhanced: np.ndarray,
    subject_mask: np.ndarray,
    line_sensitivity: int,
    structural_mask: np.ndarray,
    include_texture: bool,
) -> np.ndarray:

    sensitivity = clamp(
        line_sensitivity,
        0,
        100,
    )

    c_value = int(
        14
        - (
            sensitivity
            / 10.0
        )
    )

    c_value = clamp(
        c_value,
        2,
        12,
    )

    detail = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        c_value,
    )

    detail = cv2.bitwise_and(
        detail,
        detail,
        mask=subject_mask,
    )

    # Remove structures already represented in
    # the structural pass.
    structural_exclusion = cv2.dilate(
        structural_mask,
        np.ones(
            (3, 3),
            np.uint8,
        ),
        iterations=1,
    )

    detail = cv2.bitwise_and(
        detail,
        cv2.bitwise_not(
            structural_exclusion
        ),
    )

    if not include_texture:
        # Remove isolated one/two-pixel photographic
        # texture without destroying larger linework.
        detail = cv2.morphologyEx(
            detail,
            cv2.MORPH_OPEN,
            np.ones(
                (2, 2),
                np.uint8,
            ),
            iterations=1,
        )

    return detail


# ============================================================
# SVG
# ============================================================

def path_to_svg(
    path: Dict[str, Any],
) -> str:

    attributes = [
        f'id="{path["id"]}"',
        f'd="{path["d"]}"',
        'fill="none"',
        f'stroke="{path.get("stroke", "#000000")}"',
        (
            'stroke-width="'
            f'{path.get("strokeWidth", 1.0)}'
            '"'
        ),
        (
            'stroke-linecap="'
            f'{path.get("strokeLinecap", "round")}'
            '"'
        ),
        (
            'stroke-linejoin="'
            f'{path.get("strokeLinejoin", "round")}'
            '"'
        ),
        'vector-effect="non-scaling-stroke"',
    ]

    return (
        "<path "
        + " ".join(attributes)
        + " />"
    )


def svg_group(
    group_id: str,
    paths: List[Dict[str, Any]],
) -> str:

    body = "\n".join(
        path_to_svg(path)
        for path in paths
    )

    return (
        f'<g id="{group_id}">\n'
        f"{body}\n"
        "</g>"
    )


def build_svg(
    width: int,
    height: int,
    outer_paths: List[Dict[str, Any]],
    structural_paths: List[Dict[str, Any]],
    fine_paths: List[Dict[str, Any]],
) -> str:

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

def anchor_count(
    paths: List[Dict[str, Any]],
) -> int:
    total = 0

    for path in paths:
        d = path.get(
            "d",
            "",
        )

        # This pipeline produces M/L/Z geometry.
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
    ]

    usable = [
        value
        for value in lengths
        if value > 0
    ]

    average = (
        sum(usable)
        / len(usable)
        if usable
        else 0.0
    )

    short_count = sum(
        1
        for value in usable
        if value < 20
    )

    return (
        round(
            average,
            2,
        ),
        short_count,
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

    contrib_ok = bool(
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
        "status": "ok",
        "service": "vectorimage-worker",
        "provider": "opencv-potrace",
        "opencv": True,
        "opencvVersion": cv2.__version__,
        "opencvContrib": contrib_ok,
        "potrace": potrace_ok,
        "potraceVersion": (
            potrace_version
        ),
        "pipeline": PIPELINE_NAME,
        "version": "0.4.0",
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
                8,
            ),
        )
    )

    max_gap = int(
        config.get(
            "maxGapReconnect",
            config.get(
                "maximumGapToReconnect",
                6,
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

        subject_mask = (
            build_subject_mask(
                image_bgr
            )
        )

        if not ignore_background_texture:
            subject_mask[:] = 255

        # ----------------------------------------------------
        # PASS 1 — OUTER CONTOUR
        # ----------------------------------------------------

        outer_paths = (
            extract_outer_contour(
                subject_mask,
                simplification=(
                    max(
                        0.05,
                        path_simplification,
                    )
                ),
            )
        )

        # ----------------------------------------------------
        # BASE STRUCTURAL EDGES
        # ----------------------------------------------------

        edges = make_canny_edges(
            enhanced,
            subject_mask,
            edge_threshold,
        )

        # ----------------------------------------------------
        # PASS 2A — HORIZONTAL
        # ----------------------------------------------------

        horizontal_mask = (
            horizontal_structure_mask(
                edges,
                max_gap,
            )
        )

        horizontal_paths, (
            horizontal_skeleton
        ) = component_centerlines(
            binary=horizontal_mask,
            min_component_area=max(
                10,
                min_path_length,
            ),
            min_line_pixels=max(
                6,
                min_path_length,
            ),
            simplification=(
                path_simplification
            ),
            prefix="struct_h",
            group=GROUP_STRUCTURAL,
            path_type="structural",
            orientation="horizontal",
            stroke_width=1.0,
        )

        # ----------------------------------------------------
        # PASS 2B — VERTICAL
        # ----------------------------------------------------

        vertical_mask = (
            vertical_structure_mask(
                edges,
                max_gap,
            )
        )

        vertical_paths, (
            vertical_skeleton
        ) = component_centerlines(
            binary=vertical_mask,
            min_component_area=max(
                10,
                min_path_length,
            ),
            min_line_pixels=max(
                6,
                min_path_length,
            ),
            simplification=(
                path_simplification
            ),
            prefix="struct_v",
            group=GROUP_STRUCTURAL,
            path_type="structural",
            orientation="vertical",
            stroke_width=1.0,
        )

        # ----------------------------------------------------
        # PASS 2C — IRREGULAR STRUCTURAL
        # ----------------------------------------------------

        irregular_mask = (
            irregular_structure_mask(
                edges,
                horizontal_mask,
                vertical_mask,
            )
        )

        irregular_paths, (
            irregular_skeleton
        ) = component_centerlines(
            binary=irregular_mask,
            min_component_area=max(
                15,
                min_path_length * 2,
            ),
            min_line_pixels=max(
                10,
                min_path_length,
            ),
            simplification=max(
                0.08,
                path_simplification,
            ),
            prefix="struct_i",
            group=GROUP_STRUCTURAL,
            path_type="structural",
            orientation="irregular",
            stroke_width=0.9,
        )

        structural_paths = (
            horizontal_paths
            + vertical_paths
            + irregular_paths
        )

        structural_mask = (
            cv2.bitwise_or(
                horizontal_mask,
                vertical_mask,
            )
        )

        structural_mask = (
            cv2.bitwise_or(
                structural_mask,
                irregular_mask,
            )
        )

        # ----------------------------------------------------
        # PASS 3 — FINE DETAIL
        # ----------------------------------------------------

        fine_paths = []

        detail_mask = np.zeros_like(
            gray
        )

        detail_skeleton = np.zeros_like(
            gray
        )

        if detect_internal_lines:

            detail_mask = (
                fine_detail_mask(
                    enhanced=enhanced,
                    subject_mask=(
                        subject_mask
                    ),
                    line_sensitivity=(
                        line_sensitivity
                    ),
                    structural_mask=(
                        structural_mask
                    ),
                    include_texture=(
                        include_texture
                    ),
                )
            )

            # Fine detail must be filtered heavily.
            fine_min_area = (
                8
                if preserve_small_details
                else 16
            )

            fine_min_pixels = (
                max(
                    4,
                    min_path_length // 2,
                )
                if preserve_small_details
                else max(
                    8,
                    min_path_length,
                )
            )

            fine_paths, (
                detail_skeleton
            ) = component_centerlines(
                binary=detail_mask,
                min_component_area=(
                    fine_min_area
                ),
                min_line_pixels=(
                    fine_min_pixels
                ),
                simplification=max(
                    0.05,
                    path_simplification
                    * 0.7,
                ),
                prefix="fine",
                group=GROUP_FINE,
                path_type="fine-detail",
                orientation=None,
                stroke_width=0.75,
                allow_compact=False,
            )

        # ----------------------------------------------------
        # MERGE
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

        svg = build_svg(
            width=width,
            height=height,
            outer_paths=outer_paths,
            structural_paths=(
                structural_paths
            ),
            fine_paths=fine_paths,
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        anchors = anchor_count(
            all_paths
        )

        average_length, (
            short_fragments
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

        if len(
            all_paths
        ) > 4000:
            warnings.append(
                "Trace contains a very high number of paths."
            )

        denominator = max(
            1,
            len(
                structural_paths
                + fine_paths
            ),
        )

        fragment_ratio = (
            short_fragments
            / denominator
        )

        if fragment_ratio > 0.30:
            warnings.append(
                "Trace contains many short fragments."
            )

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        diagnostics = {}

        if return_diagnostics:

            combined_structural_skeleton = (
                cv2.bitwise_or(
                    horizontal_skeleton,
                    vertical_skeleton,
                )
            )

            combined_structural_skeleton = (
                cv2.bitwise_or(
                    combined_structural_skeleton,
                    irregular_skeleton,
                )
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

                "baseEdges":
                    np_image_to_base64_png(
                        edges
                    ),

                "horizontalMask":
                    np_image_to_base64_png(
                        horizontal_mask
                    ),

                "horizontalSkeleton":
                    np_image_to_base64_png(
                        horizontal_skeleton
                    ),

                "verticalMask":
                    np_image_to_base64_png(
                        vertical_mask
                    ),

                "verticalSkeleton":
                    np_image_to_base64_png(
                        vertical_skeleton
                    ),

                "irregularMask":
                    np_image_to_base64_png(
                        irregular_mask
                    ),

                "irregularSkeleton":
                    np_image_to_base64_png(
                        irregular_skeleton
                    ),

                "structuralSkeleton":
                    np_image_to_base64_png(
                        combined_structural_skeleton
                    ),

                "fineDetailMask":
                    np_image_to_base64_png(
                        detail_mask
                    ),

                "fineDetailSkeleton":
                    np_image_to_base64_png(
                        detail_skeleton
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
                    "horizontal": (
                        horizontal_paths
                    ),
                    "vertical": (
                        vertical_paths
                    ),
                    "irregular": (
                        irregular_paths
                    ),
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

            # Keep this for Base44 compatibility.
            "provider": "opencv-potrace",

            "pipeline": PIPELINE_NAME,
            "workerVersion": "0.4.0",

            "width": width,
            "height": height,
            "viewBox": (
                f"0 0 {width} {height}"
            ),

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
                    len(outer_paths),

                "structuralPathCount":
                    len(
                        structural_paths
                    ),

                "horizontalStructuralPathCount":
                    len(
                        horizontal_paths
                    ),

                "verticalStructuralPathCount":
                    len(
                        vertical_paths
                    ),

                "irregularStructuralPathCount":
                    len(
                        irregular_paths
                    ),

                "fineDetailPathCount":
                    len(
                        fine_paths
                    ),

                "averagePathLength":
                    average_length,

                "shortFragmentCount":
                    short_fragments,

                "fragmentationRatio":
                    round(
                        fragment_ratio,
                        4,
                    ),

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
