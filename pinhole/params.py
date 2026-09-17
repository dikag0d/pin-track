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

ALL_FIELDS = IMAGE_FIELDS + HSV_FIELDS + TRACK_FIELDS + GEOMETRY_FIELDS

DEFAULTS = {key: value for key, _, _, _, value, _ in ALL_FIELDS}
DEFAULTS.update(
    hsv_on=False,
    invert=False,
    ecc_on=True,
    loop=True,
    guides=False,
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
    return dict(SIDE_DEFAULTS if role == "side" else DEFAULTS)


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

    for key in ("hsv_on", "invert", "ecc_on", "loop", "guides"):
        if key in p and not isinstance(p[key], bool):
            raise ValueError(f"Parameter harus boolean: {key}")

    if p["smin"] > p["smax"] or p["vmin"] > p["vmax"]:
        raise ValueError("S/V minimum harus <= maksimum.")

    if p["scale_min"] > p["scale_max"]:
        raise ValueError("Skala minimum harus <= maksimum.")

    if not (p["rx0"] < p["rx1"] and p["ry0"] < p["ry1"]):
        raise ValueError("Koordinat awal search harus lebih kecil dari akhir.")

    if p["tx"] + p["tw"] > 640 or p["ty"] + p["th"] > 480:
        raise ValueError("Template melewati batas referensi 640 x 480.")
