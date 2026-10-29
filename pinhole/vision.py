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

    def detect(self, gray, p, search=None):
        self.prepare_reference(gray.shape, p)

        if search is None:
            x0, y0, x1, y1 = self.search
        else:
            height, width = gray.shape[:2]
            x0 = max(0, min(width - 1, int(search[0])))
            y0 = max(0, min(height - 1, int(search[1])))
            x1 = max(0, min(width, int(search[2])))
            y1 = max(0, min(height, int(search[3])))
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
    """Lubang samping: oval gelap digabung dengan template matching + ECC.

    Template kalibrasi dipakai bila pusatnya memang gelap. Kalau template
    mengunci logam terang, oval gelap yang menang, lalu dicocokkan lagi
    secara lokal. Bila oval kalibrasi tidak muat, bukaan gelap dilacak
    dengan template yang dipelajari dari frame itu sendiri.
    """

    MAX_JUMP = 42.0
    DARK_MEAN = 80.0

    def __init__(self, reference_frame):
        super().__init__(reference_frame)
        self.last_center = None
        self.misses = 0
        self.opening_template = None
        self.opening_center = None
        self.opening_ellipse = None

    def _ellipse_is_dark(self, gray, result):
        if result is None:
            return False
        mask = np.zeros(gray.shape[:2], np.uint8)
        try:
            cv2.ellipse(mask, result["ellipse"], 255, -1)
        except cv2.error:
            return False
        inside = gray[mask > 0]
        if inside.size < 8:
            return False
        return float(inside.mean()) < self.DARK_MEAN

    def _near(self, a, b, limit):
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return (dx * dx + dy * dy) ** 0.5 <= limit

    def _window(self, center):
        cx, cy = center
        _tx, _ty, tw, th = self.box
        return (cx - tw, cy - th, cx + tw, cy + th)

    def _prefer_dark(self, gray, p, tm, tm_note, dark):
        if self._ellipse_is_dark(gray, tm) and self._near(
            tm["center"], dark["center"], 24
        ):
            return tm, tm_note
        refined, _note = super().detect(
            gray, p, search=self._window(dark["center"])
        )
        if (
            self._ellipse_is_dark(gray, refined)
            and self._near(refined["center"], dark["center"], 18)
        ):
            return refined, "Oval gelap + TM"
        return dark, "Oval gelap"

    def _opening_seed(self, gray):
        """Bukaan kecil: inti gelap, lingkaran lebih terang, bukan bayangan besar."""
        blurred_core = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.6)
        blurred_ring = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 5.0)
        score = blurred_ring - blurred_core
        score[blurred_core > 115] = 0
        height, width = gray.shape[:2]
        score[:12, :] = 0
        score[-12:, :] = 0
        score[:, :12] = 0
        score[:, -12:] = 0
        dark = (gray < 100).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark)
        for index in range(1, count):
            x, y, comp_w, comp_h, area = (int(v) for v in stats[index])
            touches = (
                x <= 1 or y <= 1
                or x + comp_w >= width - 1
                or y + comp_h >= height - 1
            )
            if (
                touches
                or area > 500
                or area < 8
                or max(comp_w, comp_h) > 36
                or min(comp_w, comp_h) < 4
            ):
                score[labels == index] = 0
        if float(score.max()) < 42:
            return None
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        if int(gray[y, x]) >= 98:
            return None
        return float(x), float(y), float(score[y, x])

    def _crop_opening(self, gray, center):
        cx, cy = int(round(center[0])), int(round(center[1]))
        half_w, half_h = 16, 14
        x0, y0 = cx - half_w, cy - half_h
        x1, y1 = cx + half_w, cy + half_h
        height, width = gray.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            return None
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0 or float(patch.std()) < 1.0:
            return None
        return patch.copy()

    def _result_at(self, center, p, score, axes):
        axis_u, axis_v = axes
        return {
            "ellipse": (
                (float(center[0]), float(center[1])),
                (float(axis_u), float(axis_v)),
                float(p["angle"]),
            ),
            "center": (float(center[0]), float(center[1])),
            "score": float(score),
            "ecc": None,
            "refined": False,
        }

    def _match_opening(self, gray, p):
        template = self.opening_template
        if template is None or self.opening_center is None:
            return None
        th, tw = template.shape
        cx, cy = self.opening_center
        x0 = int(round(cx)) - tw
        y0 = int(round(cy)) - th
        x1 = int(round(cx)) + tw * 2
        y1 = int(round(cy)) + th * 2
        height, width = gray.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        roi = gray[y0:y1, x0:x1]
        if roi.shape[0] < th or roi.shape[1] < tw or float(roi.std()) < 1.0:
            return None
        scores = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _min_score, score, _min_loc, location = cv2.minMaxLoc(scores)
        if score < 0.45:
            return None
        px, py = x0 + location[0], y0 + location[1]
        center = (px + tw / 2.0, py + th / 2.0)
        core = gray[
            int(center[1]) - 3:int(center[1]) + 4,
            int(center[0]) - 3:int(center[0]) + 4,
        ]
        if core.size == 0 or float(core.mean()) >= 100:
            return None
        warp = np.eye(2, 3, dtype=np.float32)
        patch = gray[py:py + th, px:px + tw]
        if p.get("ecc_on", True) and patch.shape == template.shape:
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
                    singular = np.linalg.svd(candidate[:, :2], compute_uv=False)
                    valid = (
                        cc >= p["min_ecc"]
                        and np.all((singular > 0.85) & (singular < 1.18))
                        and np.linalg.det(candidate[:, :2]) > 0
                        and np.linalg.norm(candidate[:, 2]) < p["ecc_shift"]
                    )
                    if valid:
                        center = (
                            px + tw / 2.0 + float(candidate[0, 2]),
                            py + th / 2.0 + float(candidate[1, 2]),
                        )
            except cv2.error:
                pass
        return self._result_at(center, p, score, self._opening_axes(p))

    def _adopt_opening(self, gray, p, center, score):
        template = self._crop_opening(gray, center)
        if template is None:
            return None, ""
        self.opening_template = template
        self.opening_center = center
        return self._result_at(
            center, p, score, self._opening_axes(p)
        ), "TM bukaan"

    def _opening_axes(self, p):
        return (min(float(p["axis_a"]), 16.0), min(float(p["axis_b"]), 28.0))

    def _lip_radius(self, gray, cx, cy, dx, dy):
        """Jarak dari inti sampai bibir terang mulut jarum, pada satu arah."""
        height, width = gray.shape[:2]
        ix, iy = int(round(cx)), int(round(cy))
        if not (0 <= ix < width and 0 <= iy < height):
            return None
        base = int(gray[iy, ix])
        peak_radius = None
        peak_value = base
        for radius in range(2, 28):
            x = int(round(cx + dx * radius))
            y = int(round(cy + dy * radius))
            if not (1 <= x < width - 1 and 1 <= y < height - 1):
                break
            value = int(gray[y, x])
            if value > peak_value:
                peak_value = value
                peak_radius = radius
            if (
                peak_radius is not None
                and peak_value >= base + 40
                and radius >= peak_radius + 2
                and value <= peak_value - 15
            ):
                return float(peak_radius)
            if value >= 170 and radius >= 3:
                return float(radius)
        if peak_radius is not None and peak_value >= base + 35:
            return float(peak_radius)
        return None

    def _fit_rim(self, gray, center):
        """Oval mulut jarum: lebar kiri-kanan, tinggi atas-bawah, sudut tegak.

        Batas atas dan bawah inilah yang menjadi garis putus-putus koridor.
        """
        cx, cy = float(center[0]), float(center[1])
        left = self._lip_radius(gray, cx, cy, -1.0, 0.0)
        right = self._lip_radius(gray, cx, cy, 1.0, 0.0)
        up = self._lip_radius(gray, cx, cy, 0.0, -1.0)
        down = self._lip_radius(gray, cx, cy, 0.0, 1.0)
        if None in (left, right, up, down):
            return None
        oval_w = left + right
        oval_h = up + down
        if not (8.0 <= oval_w <= 40.0 and 10.0 <= oval_h <= 48.0):
            return None
        if oval_h / oval_w > 3.2 or oval_w / oval_h > 2.2:
            return None
        ecx = cx + (right - left) / 2.0
        ecy = cy + (down - up) / 2.0
        if abs(ecx - cx) > 12.0 or abs(ecy - cy) > 14.0:
            return None
        # Sudut 0: sumbu pertama mendatar, sumbu kedua tegak. Bentuk jarum.
        return (float(ecx), float(ecy)), (float(oval_w), float(oval_h)), 0.0

    def _precise_opening(self, gray, result):
        fitted = self._fit_rim(gray, result["center"])
        if fitted is None:
            if self.opening_ellipse is None:
                center = result["center"]
                axes, angle = (12.0, 18.0), 0.0
            else:
                axes, angle = self.opening_ellipse
                center = result["center"]
        else:
            center, axes, angle = fitted
            if self.opening_ellipse is not None:
                old_axes, _old_angle = self.opening_ellipse
                axes = (
                    0.65 * axes[0] + 0.35 * old_axes[0],
                    0.65 * axes[1] + 0.35 * old_axes[1],
                )
            self.opening_ellipse = (axes, angle)
        precise = dict(result)
        precise["ellipse"] = (
            (float(center[0]), float(center[1])),
            (float(axes[0]), float(axes[1])),
            float(angle),
        )
        precise["center"] = (float(center[0]), float(center[1]))
        return precise

    def _track_opening(self, gray, p):
        seed = self._opening_seed(gray)
        matched = self._match_opening(gray, p)
        if matched is not None and (
            seed is None or self._near(seed[:2], matched["center"], 12)
        ):
            self.opening_center = matched["center"]
            return self._precise_opening(gray, matched), "TM bukaan"
        if seed is None:
            self._forget_opening()
            return None, ""
        adopted, note = self._adopt_opening(
            gray, p, seed[:2], min(seed[2] / 80.0, 1.0)
        )
        if adopted is None:
            return None, ""
        return self._precise_opening(gray, adopted), note

    def _forget_opening(self):
        self.opening_template = None
        self.opening_center = None
        self.opening_ellipse = None

    def detect(self, gray, p):
        self.prepare_reference(gray.shape, p)
        tm, tm_note = super().detect(gray, p)
        dark, _dark_note = detect_dark_oval(
            gray, p, previous=self.last_center
        )
        if dark is not None:
            chosen, note = self._prefer_dark(gray, p, tm, tm_note, dark)
            self._forget_opening()
        elif self._ellipse_is_dark(gray, tm):
            chosen, note = tm, tm_note
            self._forget_opening()
        else:
            chosen, note = self._track_opening(gray, p)
            if chosen is None and tm is not None:
                note = "Template mengenai logam terang."

        if chosen is None:
            self.misses += 1
            if self.misses > 12:
                self.last_center = None
                self._forget_opening()
            return None, note

        cx, cy = chosen["center"]
        if self.last_center is not None:
            dx = cx - self.last_center[0]
            dy = cy - self.last_center[1]
            jump = (dx * dx + dy * dy) ** 0.5
            if jump > self.MAX_JUMP and chosen["score"] < 0.55:
                self.misses += 1
                if self.misses > 12:
                    self.last_center = None
                return None, "Loncatan pusat terlalu besar."

        self.misses = 0
        self.last_center = (cx, cy)
        return chosen, note


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
