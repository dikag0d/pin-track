"""Definisi parameter GUI dan nilai default kalibrasi 640 x 480."""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIDE_REFERENCE = ROOT / "calib" / "reference_side.png"
SIDE_SAMPLE = ROOT / "samples" / "side.webm"

# key, label, minimum, maksimum, default, langkah
IMAGE_FIELDS = [
    ("brightness", "Kecerahan", -100, 100, 0, 1),
    ("contrast", "Kontras", 0.10, 3.0, 1.0, 0.05),
    ("gamma", "Gamma (>1 lebih terang)", 0.20, 3.0, 1.0, 0.05),
    ("clahe", "CLAHE (0 = mati)", 0.0, 8.0, 0.0, 0.20),
    ("blur", "Radius Gaussian blur", 0, 5, 0, 1),
]

HSV_FIELDS = [
    ("hmin", "H minimum", 0, 179, 0, 1),
    ("hmax", "H maksimum", 0, 179, 179, 1),
    ("smin", "S minimum", 0, 255, 0, 1),
    ("smax", "S maksimum", 0, 255, 255, 1),
    ("vmin", "V minimum", 0, 255, 0, 1),
    ("vmax", "V maksimum", 0, 255, 255, 1),
    ("opening", "Radius opening (0 = mati)", 0, 5, 0, 1),
    ("closing", "Radius closing (0 = mati)", 0, 5, 0, 1),
]

TRACK_FIELDS = [
    ("min_match", "Minimum match", 0.0, 1.0, 0.68, 0.01),
    ("min_ecc", "Minimum ECC", 0.0, 1.0, 0.72, 0.01),
    ("scale_min", "Skala minimum", 0.50, 2.0, 0.90, 0.05),
    ("scale_max", "Skala maksimum", 0.50, 2.0, 1.10, 0.05),
    ("scale_count", "Jumlah skala", 1, 21, 5, 1),
    ("ecc_iterations", "Iterasi ECC", 10, 200, 50, 10),
    ("ecc_shift", "Batas translasi ECC (px)", 1, 100, 10, 1),
]

# Benang coklat/tembaga. Hue rendah, saturasi tinggi, bukan latar hampir putih.
THREAD_HSV_FIELDS = [
    ("th_hmin", "H minimum benang", 0, 179, 0, 1),
    ("th_hmax", "H maksimum benang", 0, 179, 22, 1),
    ("th_smin", "S minimum benang", 0, 255, 60, 1),
    ("th_smax", "S maksimum benang", 0, 255, 255, 1),
    ("th_vmin", "V minimum benang", 0, 255, 18, 1),
    ("th_vmax", "V maksimum benang", 0, 255, 230, 1),
]

THREAD_MEASURE_FIELDS = [
    ("thread_min_area", "Luas minimum (px)", 20, 500000, 250, 10),
    ("thread_min_width", "Lebar minimum (px)", 4, 4000, 30, 1),
    ("thread_min_right_x", "Batas sisi masuk (0–1)", 0.0, 1.0, 0.50, 0.01),
    ("thread_max_top", "Y awal komponen maks (0–1)", 0.0, 1.0, 0.92, 0.01),
    ("thread_tip_ratio", "Rasio tebal ujung", 0.05, 1.0, 0.30, 0.01),
]

THREAD_SMOOTH_FIELDS = [
    ("thread_still_radius", "Radius diam (px/frame)", 0.0, 80.0, 3.2, 0.1),
    ("thread_still_alpha", "Perataan saat diam", 0.0, 1.0, 0.55, 0.01),
    ("thread_quiet_radius", "Radius sangat diam (px)", 0.0, 80.0, 1.6, 0.1),
    ("thread_quiet_alpha", "Perataan sangat diam", 0.0, 1.0, 0.32, 0.01),
]

# Piksel per 1 mm pada referensi 640 x 480. 0 = tampilkan piksel.
# TOP: checkerboard 1 mm, 61.5 px mendatar dan 63.9 px menurun.
# SIDE: hanya sumbu tegak sebagai Z. Foto 489 px tinggi, kotak tegak 61.8 px,
# disetarakan ke tinggi 480 menjadi 60.7 px per mm.
SCALE_FIELDS = [
    ("px_per_mm_x", "Piksel per mm X", 0.0, 500.0, 0.0, 0.1),
    ("px_per_mm_y", "Piksel per mm Y", 0.0, 500.0, 0.0, 0.1),
    ("px_per_mm_z", "Piksel per mm Z", 0.0, 500.0, 0.0, 0.1),
]

TOP_PX_PER_MM_X = 61.5
TOP_PX_PER_MM_Y = 63.9
SIDE_PX_PER_MM_Z = 60.7

# Oval ujung pada referensi 640 x 480. Lebar dikunci karena benang tidak berubah.
THREAD_OVAL_FIELDS = [
    ("thread_oval_along", "Oval sepanjang benang", 1.0, 640.0, 12.0, 0.5),
    ("thread_oval_across", "Lebar benang (oval)", 1.0, 480.0, 18.0, 0.5),
    ("thread_oval_angle", "Sudut oval benang", -180.0, 180.0, 0.0, 0.5),
    ("thread_oval_cx", "Pratinjau oval X", 0.0, 640.0, 400.0, 0.5),
    ("thread_oval_cy", "Pratinjau oval Y", 0.0, 480.0, 230.0, 0.5),
]

# Kamera/Sumber: resolusi direkomendasikan (Otomatis/native lancar; dan util per frame untuk patch)
VIDEO_RES_FIELDS = [
    ("video_res_x", "Ukuran Sumber X (fx)", 0.0, 9999.0, 0.0, 10.0),
    ("video_res_y", "Ukuran Sumber Y (fy)", 0.0, 9999.0, 0.0, 10.0),
]

# Zoom overlay patch (floating, kecil), patch per tampilan: mask/n/blur latar
ZOOM_FIELDS = [
    ("video_patch_width", "Patch lebar (px)", 0, 1920, 80, 10),
    ("video_patch_height", "Patch tinggi (px)", 0, 1080, 60, 10),
    ("video_zoom_factor", "Faktor zoom [1..50]", 0.1, 50.0, 1.0, 0.1),
]

GEOMETRY_FIELDS = [
    ("cx", "Pusat oval X", 0.0, 640.0, 175.5, 0.5),
    ("cy", "Pusat oval Y", 0.0, 480.0, 222.5, 0.5),
    ("axis_a", "Diameter sumbu A", 1.0, 640.0, 25.0, 0.5),
    ("axis_b", "Diameter sumbu B", 1.0, 640.0, 67.0, 0.5),
    ("angle", "Sudut oval (derajat)", -180.0, 180.0, -6.0, 0.5),
    ("tx", "Template X", 0.0, 639.0, 150.0, 1.0),
    ("ty", "Template Y", 0.0, 479.0, 178.0, 1.0),
    ("tw", "Lebar template", 3.0, 640.0, 49.0, 1.0),
    ("th", "Tinggi template", 3.0, 480.0, 91.0, 1.0),
    ("rx0", "Search X awal (0–1)", 0.0, 1.0, 0.10, 0.01),
    ("ry0", "Search Y awal (0–1)", 0.0, 1.0, 0.18, 0.01),
    ("rx1", "Search X akhir (0–1)", 0.0, 1.0, 0.46, 0.01),
    ("ry1", "Search Y akhir (0–1)", 0.0, 1.0, 0.64, 0.01),
]

ALL_FIELDS = (
    IMAGE_FIELDS + HSV_FIELDS + TRACK_FIELDS + SCALE_FIELDS + GEOMETRY_FIELDS
    + VIDEO_RES_FIELDS + ZOOM_FIELDS
    + THREAD_HSV_FIELDS + THREAD_MEASURE_FIELDS + THREAD_SMOOTH_FIELDS
    + THREAD_OVAL_FIELDS
)

DEFAULTS = {key: value for key, _, _, _, value, _ in ALL_FIELDS}
DEFAULTS.update(
    hsv_on=False,
    invert=False,
    ecc_on=True,
    loop=True,
    guides=False,
    thread_on=True,
    thread_from_right=True,
    thread_oval_lock=True,
    path_overlay=True,
    video_res_x=0.0,
    video_res_y=0.0,
    video_patch_width=0,
    video_patch_height=0,
    video_zoom_factor=1.0,
)

# Kalibrasi SIDE pada samples/side.webm, bukaan tabung ~t=15s.
SIDE_DEFAULTS = {
    **DEFAULTS,
    "min_match": 0.58,
    "scale_min": 0.80,
    "scale_max": 1.15,
    "scale_count": 6,
    "ecc_shift": 12,
    "cx": 190.0,
    "cy": 171.0,
    "axis_a": 24.0,
    "axis_b": 44.0,
    "angle": -2.0,
    "tx": 168.0,
    "ty": 138.0,
    "tw": 48.0,
    "th": 64.0,
    "rx0": 0.00,
    "ry0": 0.16,
    "rx1": 0.52,
    "ry1": 0.55,
}


def defaults_for(role: str) -> dict:
    values = dict(SIDE_DEFAULTS if role == "side" else DEFAULTS)
    if role == "top":
        values["px_per_mm_x"] = TOP_PX_PER_MM_X
        values["px_per_mm_y"] = TOP_PX_PER_MM_Y
        values["px_per_mm_z"] = 0.0
    elif role == "side":
        values["px_per_mm_x"] = 0.0
        values["px_per_mm_y"] = 0.0
        values["px_per_mm_z"] = SIDE_PX_PER_MM_Z
    return values


def format_xy(x, y, px_per_mm_x=0.0, px_per_mm_y=0.0, frame_w=640, frame_h=480) -> str:
    """Koordinat tanpa koma. Skala > 0 mengubah piksel frame ke milimeter.

    Skala diukur pada referensi 640 x 480. Frame lain diskalakan ke referensi
    itu sebelum dibagi piksel-per-mm.
    """
    if (
        px_per_mm_x > 0
        and px_per_mm_y > 0
        and frame_w > 0
        and frame_h > 0
    ):
        mm_x = float(x) * 640.0 / float(frame_w) / float(px_per_mm_x)
        mm_y = float(y) * 480.0 / float(frame_h) / float(px_per_mm_y)
        return f"X {mm_x:.2f} Y {mm_y:.2f} mm"
    return f"X {int(round(float(x)))} Y {int(round(float(y)))}"


def format_z(y, px_per_mm_z, frame_h=480) -> str | None:
    """Koordinat Z dari piksel tegak. None bila skala Z belum diisi."""
    if px_per_mm_z <= 0 or frame_h <= 0:
        return None
    mm_z = float(y) * 480.0 / float(frame_h) / float(px_per_mm_z)
    return f"Z {mm_z:.2f} mm"


def format_measure(
    x, y, px_per_mm_x=0.0, px_per_mm_y=0.0, frame_w=640, frame_h=480,
    px_per_mm_z=0.0,
) -> str:
    """SIDE memakai Z saja. TOP memakai X dan Y. Tanpa skala, tetap piksel."""
    z_text = format_z(y, px_per_mm_z, frame_h)
    if z_text is not None:
        return z_text
    return format_xy(x, y, px_per_mm_x, px_per_mm_y, frame_w, frame_h)


def length_mm(px, px_per_mm, frame_span, ref_span) -> float | None:
    """Panjang piksel pada satu sumbu, atau None bila skala belum diisi."""
    if px_per_mm <= 0 or frame_span <= 0:
        return None
    return float(px) * float(ref_span) / float(frame_span) / float(px_per_mm)


def ruler_measurement(
    x0, y0, x1, y1,
    px_per_mm_x=0.0, px_per_mm_y=0.0, frame_w=640, frame_h=480,
    px_per_mm_z=0.0,
) -> dict:
    """Jarak penggaris dari titik awal ke titik akhir.

    Koordinat dalam piksel frame. Skala diukur pada referensi 640×480.
    TOP memakai X dan Y, jadi jarak adalah garis lurus dalam mm.
    SIDE memakai Z pada sumbu tegak. Jika skala X juga diisi, jarak miring
    ikut dihitung. Tanpa skala, hasil tetap piksel.
    """
    x0, y0, x1, y1 = (float(v) for v in (x0, y0, x1, y1))
    vertical_scale = float(px_per_mm_z) if float(px_per_mm_z) > 0 else float(px_per_mm_y)
    vertical_axis = "Z" if float(px_per_mm_z) > 0 else "Y"
    mm_x0 = length_mm(x0, px_per_mm_x, frame_w, 640)
    mm_x1 = length_mm(x1, px_per_mm_x, frame_w, 640)
    mm_y0 = length_mm(y0, vertical_scale, frame_h, 480)
    mm_y1 = length_mm(y1, vertical_scale, frame_h, 480)
    delta_x = None if mm_x0 is None or mm_x1 is None else mm_x1 - mm_x0
    delta_y = None if mm_y0 is None or mm_y1 is None else mm_y1 - mm_y0

    if delta_x is not None and delta_y is not None:
        distance = math.hypot(delta_x, delta_y)
        unit = "mm"
        kind = "euclidean"
    elif delta_y is not None:
        distance = abs(delta_y)
        unit = "mm"
        kind = "vertical"
    elif delta_x is not None:
        distance = abs(delta_x)
        unit = "mm"
        kind = "horizontal"
    else:
        distance = math.hypot(x1 - x0, y1 - y0)
        unit = "px"
        kind = "pixels"

    if kind == "euclidean" and vertical_axis == "Z":
        text = (
            f"Penggaris: awal X {mm_x0:.2f} Z {mm_y0:.2f} mm"
            f" | akhir X {mm_x1:.2f} Z {mm_y1:.2f} mm"
            f" | jarak {distance:.2f} mm"
        )
        overlay = f"{distance:.2f} mm"
    elif kind == "euclidean":
        text = (
            f"Penggaris: awal X {mm_x0:.2f} Y {mm_y0:.2f} mm"
            f" | akhir X {mm_x1:.2f} Y {mm_y1:.2f} mm"
            f" | jarak {distance:.2f} mm"
        )
        overlay = f"{distance:.2f} mm"
    elif kind == "vertical":
        horizontal = ""
        if abs(x1 - x0) >= 1.0:
            horizontal = " | mendatar belum berskala"
        text = (
            f"Penggaris: awal Z {mm_y0:.2f} mm"
            f" | akhir Z {mm_y1:.2f} mm"
            f" | ΔZ {distance:.2f} mm"
            f"{horizontal}"
        )
        overlay = f"ΔZ {distance:.2f} mm"
    elif kind == "horizontal":
        text = (
            f"Penggaris: awal X {mm_x0:.2f} mm"
            f" | akhir X {mm_x1:.2f} mm"
            f" | ΔX {distance:.2f} mm"
            f" | tegak belum berskala"
        )
        overlay = f"ΔX {distance:.2f} mm"
    else:
        text = (
            f"Penggaris: awal X {int(round(x0))} Y {int(round(y0))}"
            f" | akhir X {int(round(x1))} Y {int(round(y1))}"
            f" | jarak {distance:.1f} px"
        )
        overlay = f"{distance:.1f} px"

    return {
        "distance": distance,
        "unit": unit,
        "kind": kind,
        "vertical_axis": vertical_axis,
        "text": text,
        "overlay": overlay,
    }


def validate_parameters(p, top=True):
    for key, _, low, high, _, _ in ALL_FIELDS:
        if key not in p:
            continue
        value = p[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError(f"Parameter tidak valid: {key}")

    for key in (
        "hsv_on", "invert", "ecc_on", "loop", "guides",
        "thread_on", "thread_from_right", "thread_oval_lock",
        "path_overlay",
    ):
        if key in p and not isinstance(p[key], bool):
            raise ValueError(f"Parameter harus boolean: {key}")

    if p["smin"] > p["smax"] or p["vmin"] > p["vmax"]:
        raise ValueError("S/V minimum harus <= maksimum.")

    if (
        "th_smin" in p
        and (
            p["th_smin"] > p["th_smax"]
            or p["th_vmin"] > p["th_vmax"]
        )
    ):
        raise ValueError("S/V benang: minimum harus <= maksimum.")

    if (
        "thread_quiet_radius" in p
        and p["thread_quiet_radius"] > p["thread_still_radius"]
    ):
        raise ValueError("Radius sangat diam harus <= radius diam.")

    if p["scale_min"] > p["scale_max"]:
        raise ValueError("Skala minimum harus <= maksimum.")

    if not (p["rx0"] < p["rx1"] and p["ry0"] < p["ry1"]):
        raise ValueError("Koordinat awal search harus lebih kecil dari akhir.")

    if p["tx"] + p["tw"] > 640 or p["ty"] + p["th"] > 480:
        raise ValueError("Template melewati batas referensi 640 x 480.")
