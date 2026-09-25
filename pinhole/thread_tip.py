"""Ujung bebas benang coklat, subpiksel, untuk kamera TOP dan SIDE.

Benang masuk dari satu tepi gambar. Yang dilacak adalah ujung bebasnya
(ujung yang tidak terpotong batas gambar), diukur pada mask sebelum
penutupan morfologi supaya kernel tidak menggeser ujung.

Saat ujung diam, koordinat diratakan. Saat perpindahan melewati radius
diam, marker langsung mengikuti pengukuran baru.

Ukuran oval ujung dikunci ke kalibrasi: lebar benang tidak berubah,
yang mengikuti pengukuran hanya posisi (dan sudut saat kunci mati).
"""

from __future__ import annotations

import cv2
import numpy as np

from pinhole.vision import ellipse_points

# Morfologi tetap: opening membuang bintik, closing hanya untuk komponen.
# Titik ukur memakai mask opened, bukan closed.
OPEN_KERNEL = np.ones((2, 2), np.uint8)
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

MIN_PROFILE_LENGTH = 24
MIN_GOOD_COLUMNS = 12
BODY_SPAN = (0.35, 0.75)
MIN_BODY_THICKNESS = 4.0
TIP_RUN = 4
THRESHOLD_FLOOR = 3.0
SHAFT_OFFSET = 5.0
SHAFT_START = 12.0
SHAFT_END = 42.0
MAX_SLOPE = 1.0
LOCAL_CLIP = (8.0, 70.0)
ALONG_SCALE = 0.34
ALONG_CLIP = (8.0, 22.0)
EDGE_MARGIN = 2.0
NEAR_SCORE_AREA = 0.35
NEAR_PAD_X = 40
NEAR_PAD_Y = 50
DISTANCE_PENALTY = 6.0
MISS_RESET_SMOOTH = 4
MISS_DROP_TRACK = 6


def _hsv_range(hsv, p):
    h0, h1 = int(p["th_hmin"]), int(p["th_hmax"])
    s0, s1 = int(p["th_smin"]), int(p["th_smax"])
    v0, v1 = int(p["th_vmin"]), int(p["th_vmax"])
    if h0 <= h1:
        return cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
    high = cv2.inRange(hsv, (h0, s0, v0), (179, s1, v1))
    low = cv2.inRange(hsv, (0, s0, v0), (h1, s1, v1))
    return cv2.bitwise_or(high, low)


class BrownThreadTipTracker:
    """Pelacak ujung bebas benang coklat pada satu aliran kamera."""

    def __init__(self):
        self.prev_raw = None
        self.smooth = None
        self.smooth_thickness = None
        self.smooth_angle = 0.0
        self.missed = 0
        self.from_right = True

    def reset(self):
        self.prev_raw = None
        self.smooth = None
        self.smooth_thickness = None
        self.smooth_angle = 0.0
        self.missed = 0

    def preview_mask(self, frame, p):
        """Mask opened pada koordinat gambar, untuk tampilan kalibrasi.

        Kernel 2x2 tidak simetris, jadi sisi kiri dibalik dulu — sama dengan
        jalur ukur — lalu dikembalikan. Mask tampilan tetap menempel pada benang.
        """
        from_right = bool(p.get("thread_from_right", True))
        working = frame if from_right else cv2.flip(frame, 1)
        opened, _closed = self._masks(working, p)
        return opened if from_right else cv2.flip(opened, 1)

    def _masks(self, frame, p):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        opened = _hsv_range(hsv, p)
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, OPEN_KERNEL)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, CLOSE_KERNEL)
        return opened, closed

    def _column_profile(self, binary):
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None
        x0 = int(xs.min())
        x1 = int(xs.max())
        length = x1 - x0 + 1
        if length < MIN_PROFILE_LENGTH:
            return None

        relative = xs - x0
        order = np.argsort(relative, kind="mergesort")
        relative = relative[order]
        ys = ys[order]
        counts = np.bincount(relative, minlength=length)

        med = np.full(length, np.nan, np.float64)
        thick = counts.astype(np.float64)
        start = 0
        for index, count in enumerate(counts):
            count = int(count)
            if count:
                med[index] = float(np.median(ys[start:start + count]))
            start += count

        good = np.isfinite(med)
        if int(good.sum()) < MIN_GOOD_COLUMNS:
            return None
        columns = np.arange(length)
        filled = np.interp(columns, columns[good], med[good])
        smooth = np.empty_like(filled)
        for index in range(length):
            window = filled[max(0, index - 4):min(length, index + 5)]
            smooth[index] = float(np.median(window))

        thickness = np.convolve(thick, np.ones(5) / 5.0, mode="same")
        thickness[0] = thick[0]
        thickness[1] = thick[1]
        thickness[-1] = thick[-1]
        thickness[-2] = thick[-2]
        return x0, smooth, thickness, thick

    def _measure_tip(self, opened, component, p):
        profile = self._column_profile(cv2.bitwise_and(opened, component))
        if profile is None:
            return None
        x0, centerline, thickness, raw_thick = profile
        length = len(centerline)
        lo = int(BODY_SPAN[0] * length)
        hi = int(BODY_SPAN[1] * length)
        body = thickness[lo:hi]
        body = body[body > 1.0]
        if len(body) < 4:
            return None
        body_thickness = float(np.median(body))
        if body_thickness < MIN_BODY_THICKNESS:
            return None

        threshold = max(
            THRESHOLD_FLOOR,
            float(p["thread_tip_ratio"]) * body_thickness,
        )
        hit = None
        for index in range(0, length - 5):
            if np.all(thickness[index:index + TIP_RUN] >= threshold):
                hit = index
                break
        if hit is None:
            return None

        if hit > 0 and thickness[hit] > thickness[hit - 1]:
            rise = max(thickness[hit] - thickness[hit - 1], 1e-3)
            frac = (threshold - thickness[hit - 1]) / rise
            tip_index = (hit - 1) + float(np.clip(frac, 0.0, 1.0))
        else:
            tip_index = float(hit)

        columns = np.arange(length)
        near = float(np.interp(tip_index + SHAFT_OFFSET, columns, centerline))
        shaft_x = tip_index + np.arange(SHAFT_START, SHAFT_END)
        shaft_x = shaft_x[(shaft_x >= 1.0) & (shaft_x <= length - 2.0)]
        if len(shaft_x) < 8:
            return None
        shaft_y = np.interp(shaft_x, columns, centerline)
        slope = float(np.polyfit(shaft_x, shaft_y, 1)[0])
        if abs(slope) > MAX_SLOPE:
            slope = 0.0
            tip_y = near
        else:
            tip_y = near - SHAFT_OFFSET * slope
        tip_x = float(x0) + tip_index

        height, width = opened.shape
        if not (
            EDGE_MARGIN <= tip_x < width - EDGE_MARGIN
            and EDGE_MARGIN <= tip_y < height - EDGE_MARGIN
        ):
            return None

        loc = int(np.clip(round(tip_index + SHAFT_START), 0, length - 1))
        local = float(np.median(raw_thick[max(0, loc - 2):loc + 3]))
        if local <= 1.0:
            local = body_thickness
        local = float(np.clip(local, LOCAL_CLIP[0], LOCAL_CLIP[1]))
        angle = float(np.degrees(np.arctan(slope)))
        along = float(np.clip(local * ALONG_SCALE, ALONG_CLIP[0], ALONG_CLIP[1]))
        return {
            "tip": (tip_x, tip_y),
            "tip_ellipse": ((tip_x, tip_y), (along, local), angle),
            "local_thickness": local,
            "body_thickness": body_thickness,
            "angle": angle,
        }

    def _candidates(self, opened, closed, p):
        height, width = closed.shape
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed)
        min_area = float(p["thread_min_area"])
        min_width = int(p["thread_min_width"])
        min_right = int(float(p["thread_min_right_x"]) * width)
        max_top = int(float(p["thread_max_top"]) * height)
        found = []
        for label in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[label])
            if area < min_area or bw < min_width:
                continue
            if x + bw < min_right:
                continue
            if y > max_top:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            measured = self._measure_tip(opened, component, p)
            if measured is None:
                continue
            measured["area"] = float(area)
            measured["bbox"] = (x, y, bw, bh)
            measured["score"] = float(area) + 2.0 * bw
            found.append(measured)
        return found

    def _choose(self, found):
        if not found:
            return None
        best = max(found, key=lambda item: item["score"])
        if self.prev_raw is None or self.missed > MISS_DROP_TRACK:
            return best
        px, py = self.prev_raw
        nearby = []
        for item in found:
            if item["area"] < NEAR_SCORE_AREA * best["area"]:
                continue
            x, y, bw, bh = item["bbox"]
            if not (
                (x - NEAR_PAD_X) <= px <= (x + bw + NEAR_PAD_X)
                and (y - NEAR_PAD_Y) <= py <= (y + bh + NEAR_PAD_Y)
            ):
                continue
            tip_x, tip_y = item["tip"]
            ranked = dict(item)
            ranked["score"] -= DISTANCE_PENALTY * float(
                np.hypot(tip_x - px, tip_y - py)
            )
            nearby.append(ranked)
        if nearby:
            return max(nearby, key=lambda item: item["score"])
        return best

    def _stabilize(self, raw_tip, thickness, angle, p):
        raw = np.array(raw_tip, np.float64)
        if self.smooth is None or self.missed > MISS_RESET_SMOOTH:
            self.smooth = raw.copy()
            self.smooth_thickness = float(thickness)
            self.smooth_angle = float(angle)
        else:
            if self.prev_raw is not None:
                step = float(np.hypot(*(raw - self.prev_raw)))
            else:
                step = 99.0
            if step <= float(p["thread_quiet_radius"]):
                alpha = float(p["thread_quiet_alpha"])
            elif step <= float(p["thread_still_radius"]):
                alpha = float(p["thread_still_alpha"])
            else:
                alpha = 1.0
            self.smooth = self.smooth + alpha * (raw - self.smooth)
            self.smooth_thickness += alpha * (
                float(thickness) - self.smooth_thickness
            )
            delta_angle = (
                (float(angle) - self.smooth_angle + 180.0) % 360.0 - 180.0
            )
            self.smooth_angle += alpha * delta_angle
        self.prev_raw = raw
        self.missed = 0
        return (
            float(self.smooth[0]),
            float(self.smooth[1]),
            float(self.smooth_thickness),
            float(self.smooth_angle),
        )

    def _unflip(self, result, width):
        def flip_x(value):
            return (width - 1) - float(value)

        tip_x, tip_y = result["tip"]
        (cx, cy), axes, angle = result["tip_ellipse"]
        x, y, bw, bh = result["bbox"]
        flipped = dict(result)
        flipped["tip"] = (flip_x(tip_x), float(tip_y))
        flipped["angle"] = -float(result["angle"])
        flipped["tip_ellipse"] = (
            (flip_x(cx), float(cy)),
            axes,
            -float(angle),
        )
        flipped["bbox"] = (int(width - (x + bw)), int(y), int(bw), int(bh))
        return flipped

    def detect(self, frame, p):
        """Kembalikan (hasil atau None, mask opened pada koordinat gambar)."""
        from_right = bool(p.get("thread_from_right", True))
        if from_right != self.from_right:
            self.reset()
            self.from_right = from_right

        working = frame if from_right else cv2.flip(frame, 1)
        opened, closed = self._masks(working, p)
        display = opened if from_right else cv2.flip(opened, 1)

        chosen = self._choose(self._candidates(opened, closed, p))
        if chosen is None:
            self.missed += 1
            if self.missed > MISS_DROP_TRACK:
                self.prev_raw = None
                self.smooth = None
                self.smooth_thickness = None
            return None, display

        tip_x, tip_y, local, angle = self._stabilize(
            chosen["tip"], chosen["local_thickness"], chosen["angle"], p
        )
        along = float(np.clip(local * ALONG_SCALE, ALONG_CLIP[0], ALONG_CLIP[1]))
        result = {
            "score": chosen["score"],
            "area": chosen["area"],
            "bbox": chosen["bbox"],
            "tip": (tip_x, tip_y),
            "tip_ellipse": ((tip_x, tip_y), (along, local), angle),
            "local_thickness": local,
            "body_thickness": float(chosen["body_thickness"]),
            "angle": angle,
            "oval_locked": False,
        }
        if not from_right:
            result = self._unflip(result, frame.shape[1])
        return lock_tip_oval(result, p, frame.shape), display


def locked_axes(p, width, height):
    """Diameter oval dalam piksel frame, diskalakan dari referensi 640 x 480."""
    sx, sy = width / 640.0, height / 480.0
    along = max(1.0, float(p["thread_oval_along"]) * sx)
    across = max(1.0, float(p["thread_oval_across"]) * sy)
    angle = float(p["thread_oval_angle"])
    return along, across, angle


def thread_guide_ellipse(p, width, height):
    """Oval kalibrasi pada pusat pratinjau, untuk frame beku."""
    sx, sy = width / 640.0, height / 480.0
    along, across, angle = locked_axes(p, width, height)
    center = (
        float(p["thread_oval_cx"]) * sx,
        float(p["thread_oval_cy"]) * sy,
    )
    return (center, (along, across), angle)


def lock_tip_oval(result, p, shape):
    """Ganti ukuran oval dengan kalibrasi. Pusat tetap di ujung terukur."""
    if result is None or not p.get("thread_oval_lock", True):
        return result
    height, width = shape[:2]
    along, across, angle = locked_axes(p, width, height)
    tip_x, tip_y = result["tip"]
    locked = dict(result)
    locked["tip_ellipse"] = ((float(tip_x), float(tip_y)), (along, across), angle)
    locked["oval_locked"] = True
    return locked


def draw_thread_guide(frame, p):
    """Oval cyan kalibrasi. Ukuran ini yang dikunci saat pelacakan."""
    out = frame.copy()
    height, width = out.shape[:2]
    try:
        cv2.ellipse(
            out, thread_guide_ellipse(p, width, height),
            (0, 255, 255), 2, cv2.LINE_AA,
        )
    except cv2.error:
        pass
    return out


# Garis putus-putus jalur insertion. Hijau = bukaan jarum, kuning = benang.
PATH_DASH = 14
PATH_GAP = 8
PATH_ALIGN_TOLERANCE = 0.75
NEEDLE_COLOR = (0, 255, 0)
THREAD_PATH_COLOR = (0, 255, 255)


def _span(points):
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def insertion_paths(pinhole, thread, from_right=True):
    """Koridor vertikal jalur jarum dan jalur benang.

    Jalur benang selaras bila seluruh tebalnya berada di dalam bukaan
    jarum. Di luar itu ujung akan menabrak dinding lubang.
    """
    paths = {"needle": None, "thread": None, "aligned": None}
    if pinhole is not None and pinhole.get("ellipse") is not None:
        x_min, y_min, x_max, y_max = _span(ellipse_points(pinhole["ellipse"]))
        paths["needle"] = {
            "y0": y_min,
            "y1": y_max,
            "gate_x": x_max if from_right else x_min,
        }
    if thread is not None and thread.get("tip_ellipse") is not None:
        tip_x, tip_y = thread["tip"]
        _x0, y_min, _x1, y_max = _span(ellipse_points(thread["tip_ellipse"]))
        half = max((y_max - y_min) / 2.0, 1.0)
        paths["thread"] = {
            "y0": float(tip_y) - half,
            "y1": float(tip_y) + half,
            "tip": (float(tip_x), float(tip_y)),
            "radius": half,
        }
    needle = paths["needle"]
    band = paths["thread"]
    if needle is not None and band is not None:
        tol = PATH_ALIGN_TOLERANCE
        paths["aligned"] = (
            band["y0"] >= needle["y0"] - tol
            and band["y1"] <= needle["y1"] + tol
        )
    return paths


def _dashed_segment(image, start, end, color):
    start = np.array(start, np.float64)
    end = np.array(end, np.float64)
    delta = end - start
    length = float(np.hypot(delta[0], delta[1]))
    if length < 1.0:
        return
    direction = delta / length
    pos = 0.0
    while pos < length:
        stop = min(pos + PATH_DASH, length)
        a = start + direction * pos
        b = start + direction * stop
        cv2.line(
            image,
            (int(round(a[0])), int(round(a[1]))),
            (int(round(b[0])), int(round(b[1]))),
            color,
            2,
            cv2.LINE_8,
        )
        pos += PATH_DASH + PATH_GAP


def _hline(image, y, color):
    height, width = image.shape[:2]
    yy = int(round(y))
    if 0 <= yy < height:
        _dashed_segment(image, (0, yy), (width - 1, yy), color)


def draw_insertion_overlay(frame, pinhole, thread, from_right=True):
    """Garis putus-putus jalur jarum (hijau) dan jalur benang (kuning).

    Garis tegak hijau adalah sisi lubang yang menghadap benang.
    Lingkaran kuning menandai ujung, dengan diameter sama dengan tebal jalur.
    """
    out = frame.copy()
    height, width = out.shape[:2]
    paths = insertion_paths(pinhole, thread, from_right=from_right)
    needle = paths["needle"]
    band = paths["thread"]
    if needle is not None:
        _hline(out, needle["y0"], NEEDLE_COLOR)
        _hline(out, needle["y1"], NEEDLE_COLOR)
        gate = int(round(needle["gate_x"]))
        if 0 <= gate < width:
            _dashed_segment(
                out,
                (gate, needle["y0"]),
                (gate, needle["y1"]),
                NEEDLE_COLOR,
            )
    if band is not None:
        _hline(out, band["y0"], THREAD_PATH_COLOR)
        _hline(out, band["y1"], THREAD_PATH_COLOR)
        tip_x, tip_y = band["tip"]
        tip = (int(round(tip_x)), int(round(tip_y)))
        radius = max(4, int(round(band["radius"])))
        if 0 <= tip[0] < width and 0 <= tip[1] < height:
            cv2.circle(out, tip, radius, THREAD_PATH_COLOR, 2, cv2.LINE_8)

    aligned = paths["aligned"]
    if aligned is True:
        label, color = "JALUR SELARAS", NEEDLE_COLOR
    elif aligned is False:
        label, color = "BENANG MELEWATI JARUM", (0, 80, 255)
    else:
        label = None
    if label is not None:
        cv2.putText(
            out, label, (16, 76), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 1, cv2.LINE_AA,
        )
    return out


def draw_thread(frame, thread, enabled=True):
    """Gambar elips ujung, silang, dan koordinat subpiksel."""
    if not enabled:
        return frame
    out = frame.copy()
    if thread is not None:
        tip_x, tip_y = thread["tip"]
        tip = (int(round(tip_x)), int(round(tip_y)))
        try:
            cv2.ellipse(
                out, thread["tip_ellipse"], (0, 255, 255), 2, cv2.LINE_AA
            )
        except cv2.error:
            pass
        cv2.drawMarker(
            out, tip, (0, 0, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA
        )
        cv2.circle(out, tip, 2, (0, 0, 255), -1, cv2.LINE_AA)
        text = f"UJUNG BENANG {tip_x:.1f},{tip_y:.1f}"
    else:
        text = "UJUNG BENANG: tidak terdeteksi"
    cv2.putText(
        out, text, (16, 52), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return out
