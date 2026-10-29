"""Pemrosesan citra dan pelacak oval.

Algoritme TOP (PinholeTracker.detect) tidak diubah dari sumber asli:
template matching multi-skala + ECC affine pada koordinat 640 x 480.
"""

from __future__ import annotations

import json

import cv2
import numpy as np


def prepare_image(frame, p):
    """Proses yang sama digunakan untuk frame dan gambar referensi."""
    adjusted = np.clip(
        frame.astype(np.float32) * p["contrast"] + p["brightness"],
        0, 255,
    ).astype(np.uint8)

    if abs(p["gamma"] - 1.0) > 1e-6:
        lut = np.clip(
            (np.arange(256, dtype=np.float32) / 255.0)
            ** (1.0 / p["gamma"]) * 255.0,
            0, 255,
        ).astype(np.uint8)
        adjusted = cv2.LUT(adjusted, lut)

    if p["clahe"] > 0:
        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=float(p["clahe"]), tileGridSize=(8, 8)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    radius = int(p["blur"])
    if radius:
        kernel = 2 * radius + 1
        adjusted = cv2.GaussianBlur(adjusted, (kernel, kernel), 0)

    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
    h0, h1 = int(p["hmin"]), int(p["hmax"])
    s0, s1 = int(p["smin"]), int(p["smax"])
    v0, v1 = int(p["vmin"]), int(p["vmax"])

    # H minimum > maksimum berarti rentang hue melewati 179 -> 0.
    if h0 <= h1:
        mask = cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    else:
        a = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1))
        b = cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
        mask = cv2.bitwise_or(a, b)

    if p["invert"]:
        mask = cv2.bitwise_not(mask)

    for key, operation in (
        ("opening", cv2.MORPH_OPEN),
        ("closing", cv2.MORPH_CLOSE),
    ):
        radius = int(p[key])
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            mask = cv2.morphologyEx(mask, operation, kernel)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    if p["hsv_on"]:
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    return adjusted, mask, gray


def ellipse_points(ellipse):
    (cx, cy), (a, b), angle = ellipse
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    r = np.deg2rad(angle)
    rotation = np.array([
        [np.cos(r), -np.sin(r)],
        [np.sin(r), np.cos(r)],
    ])
    points = np.column_stack(
        (a / 2 * np.cos(t), b / 2 * np.sin(t))
    )
    return points @ rotation.T + (cx, cy)


def transformed_ellipse(ellipse, matrix):
    points = ellipse_points(ellipse)
    points = points @ matrix[:, :2].T + matrix[:, 2]
    return cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))


def geometry(p, width, height):
    sx, sy = width / 640.0, height / 480.0

    reference = (
        (p["cx"], p["cy"]),
        (p["axis_a"], p["axis_b"]),
        p["angle"],
    )
    ellipse = transformed_ellipse(
        reference,
        np.array([[sx, 0, 0], [0, sy, 0]], np.float32),
    )

    x = round(p["tx"] * sx)
    y = round(p["ty"] * sy)
    x_end = round((p["tx"] + p["tw"]) * sx)
    y_end = round((p["ty"] + p["th"]) * sy)

    search = (
        int(p["rx0"] * width), int(p["ry0"] * height),
        int(p["rx1"] * width), int(p["ry1"] * height),
    )
    return ellipse, (x, y, x_end - x, y_end - y), search


class PinholeTracker:
    def __init__(self, reference_frame):
        self.reference = reference_frame.copy()
        self.cache_key = None
        self.templates = []

    def prepare_reference(self, shape, p):
        height, width = shape
        key = (height, width, json.dumps(p, sort_keys=True))
        if key == self.cache_key:
            return

        reference = self.reference
        if reference.shape[:2] != shape:
            reference = cv2.resize(reference, (width, height))

        _, _, gray = prepare_image(reference, p)
        self.ellipse, self.box, self.search = geometry(p, width, height)
        tx, ty, tw, th = self.box

        if (
            tw < 3 or th < 3 or tx < 0 or ty < 0
            or tx + tw > width or ty + th > height
        ):
            raise ValueError("Template terlalu kecil atau di luar gambar.")

        points = ellipse_points(self.ellipse)
        if (
            points[:, 0].min() < tx
            or points[:, 0].max() >= tx + tw
            or points[:, 1].min() < ty
            or points[:, 1].max() >= ty + th
        ):
            raise ValueError("Perbesar template agar seluruh oval berada di dalamnya.")

        template = gray[ty:ty + th, tx:tx + tw].copy()
        if template.std() < 1.0:
            raise ValueError(
                "Template terlalu seragam. Periksa HSV, cahaya, atau kalibrasi."
            )

        self.templates = []
        for scale in np.linspace(
            p["scale_min"], p["scale_max"], int(p["scale_count"])
        ):
            size = (max(3, round(tw * scale)), max(3, round(th * scale)))
            candidate = cv2.resize(template, size)
            if candidate.std() >= 1.0:
                self.templates.append(candidate)

        self.cache_key = key

    def detect(self, gray, p):
        self.prepare_reference(gray.shape, p)

        x0, y0, x1, y1 = self.search
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0 or roi.std() < 1.0:
            return None, "Area pencarian kosong atau terlalu seragam."

        best = None
        for template in self.templates:
            hh, ww = template.shape
            if hh > roi.shape[0] or ww > roi.shape[1]:
                continue

            scores = cv2.matchTemplate(
                roi, template, cv2.TM_CCOEFF_NORMED
            )
            scores = np.nan_to_num(
                scores, nan=-1.0, posinf=-1.0, neginf=-1.0
            )
            _, score, _, location = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (float(score), location, template)

        if best is None:
            return None, "Search terlalu kecil untuk ukuran template."

        if best[0] < p["min_match"]:
            return None, f"Tidak terdeteksi | match terbaik {best[0]:.3f}"

        score, (dx, dy), template = best
        hh, ww = template.shape
        px, py = x0 + dx, y0 + dy
        patch = gray[py:py + hh, px:px + ww]

        if patch.std() < 1.0:
            return None, "Patch kandidat terlalu seragam."

        warp = np.eye(2, 3, dtype=np.float32)
        ecc_score = None
        refined = False

        if p["ecc_on"]:
            try:
                cc, candidate = cv2.findTransformECC(
                    template,
                    patch,
                    warp.copy(),
                    cv2.MOTION_AFFINE,
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        int(p["ecc_iterations"]),
                        1e-4,
                    ),
                    None,
                    3,
                )

                if np.isfinite(cc) and np.isfinite(candidate).all():
                    ecc_score = float(cc)
                    singular = np.linalg.svd(
                        candidate[:, :2], compute_uv=False
                    )
                    valid = (
                        cc >= p["min_ecc"]
                        and np.all((singular > 0.85) & (singular < 1.18))
                        and np.linalg.det(candidate[:, :2]) > 0
                        and np.linalg.norm(candidate[:, 2]) < p["ecc_shift"]
                    )
                    if valid:
                        warp = candidate
                        refined = True
            except cv2.error:
                pass

        tx, ty, tw, th = self.box
        sx, sy = ww / tw, hh / th

        # Koordinat full frame referensi -> template hasil resize.
        # Koreksi setengah piksel mengikuti pemetaan pusat piksel resize.
        origin = np.array([
            [sx, 0, -tx * sx + (sx - 1) / 2],
            [0, sy, -ty * sy + (sy - 1) / 2],
            [0, 0, 1],
        ], dtype=np.float32)

        # ECC memetakan koordinat template -> patch saat ini.
        affine = np.vstack((warp, [0, 0, 1])) @ origin
        affine[0, 2] += px
        affine[1, 2] += py

        ellipse = transformed_ellipse(self.ellipse, affine[:2])
        cx, cy = ellipse[0]
        height, width = gray.shape

        if not (
            np.isfinite(ellipse_points(ellipse)).all()
            and 0 <= cx < width
            and 0 <= cy < height
        ):
            return None, "Transformasi di luar frame."

        result = {
            "ellipse": ellipse,
            "center": (float(cx), float(cy)),
            "score": score,
            "ecc": ecc_score,
            "refined": refined,
        }
        return result, "TM + ECC" if refined else "TM; tanpa penyempurnaan ECC"


def detect_dark_oval(gray, p, previous=None):
    """Oval lubang: piksel gelap di dalam logam terang, di kotak pencarian.

    Template matching mengunci tekstur terang. Lubang jarum samping adalah
    oval yang lebih gelap daripada lingkarannya.
    """
    height, width = gray.shape[:2]
    reference, _box, (x0, y0, x1, y1) = geometry(p, width, height)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0 or min(roi.shape) < 32:
        return None, "Area pencarian terlalu kecil."

    (_center, (axis_a, axis_b), _angle) = reference
    expected = max(float(axis_a) * float(axis_b), 1.0)
    kernel = int(max(axis_a, axis_b) * 1.6)
    kernel = max(31, min(81, kernel + (kernel % 2 == 0)))
    blur = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    contrast = blur.astype(np.int16) - roi.astype(np.int16)
    if int(contrast.max()) < 40:
        return None, "Tidak ada lubang gelap."
    # Seluruh isi lubang, bukan hanya tepi kontras tertinggi.
    mask = ((blur > 70) & (contrast >= 40)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)

    found = []
    for index in range(1, count):
        _x, _y, comp_w, comp_h, area = (int(v) for v in stats[index])
        if area < 25 or area > expected * 3 or comp_w < 4 or comp_h < 8:
            continue
        ys, xs = np.where(labels == index)
        if len(xs) < 8:
            continue
        try:
            ellipse = cv2.fitEllipse(np.stack([xs, ys], 1).astype(np.float32))
        except cv2.error:
            continue
        (cx, cy), (axis_u, axis_v), angle = ellipse
        major, minor = max(axis_u, axis_v), min(axis_u, axis_v)
        expect_major = max(float(axis_a), float(axis_b))
        expect_minor = min(float(axis_a), float(axis_b))
        if (
            minor < 0.22 * expect_minor
            or major < 0.55 * expect_major
            or major > 1.7 * expect_major
            or minor > 1.6 * expect_minor
            or major / max(minor, 1.0) > 6
        ):
            continue
        # Blob yang menempel tepi kotak pencarian biasanya bayangan, bukan lubang.
        margin = 12
        if not (
            margin <= cx < roi.shape[1] - margin
            and margin <= cy < roi.shape[0] - margin
        ):
            continue
        gx = int(np.clip(round(cx), 0, roi.shape[1] - 1))
        gy = int(np.clip(round(cy), 0, roi.shape[0] - 1))
        filled = np.zeros(roi.shape, np.uint8)
        cv2.ellipse(
            filled, (gx, gy),
            (max(1, int(round(axis_u / 2))), max(1, int(round(axis_v / 2)))),
            float(angle), 0, 360, 255, -1,
        )
        inside = roi[filled > 0]
        if inside.size < 12:
            continue
        ring = np.zeros(roi.shape, np.uint8)
        cv2.ellipse(
            ring, (gx, gy),
            (max(1, int(round(axis_u / 2)) + 6), max(1, int(round(axis_v / 2)) + 6)),
            float(angle), 0, 360, 255, -1,
        )
        ring[filled > 0] = 0
        outside = roi[ring > 0]
        mean_in = float(inside.mean())
        mean_out = float(outside.mean()) if outside.size else mean_in
        gap = mean_out - mean_in
        if gap < 40 or mean_in > 90:
            continue
        if previous is None and gap < 70:
            continue
        full_x = float(cx) + x0
        full_y = float(cy) + y0
        size_error = abs(np.log((axis_u * axis_v) / expected))
        jump = 0.0
        if previous is not None:
            jump = (
                (full_x - previous[0]) ** 2 + (full_y - previous[1]) ** 2
            ) ** 0.5
        rank = gap - 12.0 * size_error - (0.15 * max(0.0, jump - 20.0))
        found.append((rank, jump, full_x, full_y, axis_u, axis_v, float(angle), gap))

    if previous is not None and found:
        found = [item for item in found if item[1] <= 60.0]
    if not found:
        return None, "Oval gelap tidak ditemukan."
    best = max(found)

    _rank, _jump, full_x, full_y, axis_u, axis_v, angle, gap = best
    return {
        "ellipse": ((full_x, full_y), (float(axis_u), float(axis_v)), angle),
        "center": (full_x, full_y),
        "score": float(np.clip(gap / 160.0, 0.0, 1.0)),
        "ecc": None,
        "refined": False,
    }, "Oval gelap"


class SidePinholeTracker(PinholeTracker):
    """Pelacak samping: template matching multi-skala + ECC affine (sama seperti TOP).

    Algoritme identik dengan PinholeTracker.detect; hanya menambahkan
    pengecekan loncatan pusat antar-frame untuk menolak deteksi yang melompat
    terlalu jauh ketika skor rendah.
    """

    MAX_JUMP = 42.0

    def __init__(self, reference_frame):
        super().__init__(reference_frame)
        self.last_center = None
        self.misses = 0

    def detect(self, gray, p):
        result, note = super().detect(gray, p)
        if result is None:
            self.misses += 1
            if self.misses > 12:
                self.last_center = None
            return None, note

        cx, cy = result["center"]
        if self.last_center is not None:
            dx = cx - self.last_center[0]
            dy = cy - self.last_center[1]
            jump = (dx * dx + dy * dy) ** 0.5
            if jump > self.MAX_JUMP and result["score"] < 0.55:
                self.misses += 1
                if self.misses > 12:
                    self.last_center = None
                return None, "Loncatan pusat terlalu besar."

        self.misses = 0
        self.last_center = (cx, cy)
        return result, note


def tracker_for(role: str, reference_frame):
    if role == "side":
        return SidePinholeTracker(reference_frame)
    return PinholeTracker(reference_frame)


def draw_guides(frame, p):
    out = frame.copy()
    height, width = out.shape[:2]
    ellipse, (x, y, tw, th), (x0, y0, x1, y1) = geometry(
        p, width, height
    )
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 180, 0), 1)
    cv2.rectangle(out, (x, y), (x + tw, y + th), (0, 220, 255), 1)
    cv2.ellipse(out, ellipse, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def draw_detection(
    frame, result, px_per_mm_x=0.0, px_per_mm_y=0.0, px_per_mm_z=0.0,
):
    from pinhole.params import format_measure

    out = frame.copy()
    if result:
        cv2.ellipse(out, result["ellipse"], (0, 255, 0), 1, cv2.LINE_AA)
        center = tuple(round(v) for v in result["center"])
        cv2.circle(out, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
        label = "PINHOLE " + format_measure(
            result["center"][0], result["center"][1],
            px_per_mm_x, px_per_mm_y, out.shape[1], out.shape[0], px_per_mm_z,
        )
    else:
        label = "PINHOLE: tidak terdeteksi"

    cv2.putText(
        out, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return out
