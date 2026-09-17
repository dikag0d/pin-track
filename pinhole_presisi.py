"""Pelacak oval lubang jarum untuk rekaman samping (side.webm).

Membutuhkan numpy, opencv-python, ffmpeg/ffprobe.

Jalankan:
    python pinhole_presisi.py samples/side.webm hasil.mp4

Oval referensi dikalibrasi manual pada bukaan frame yang lubangnya terlihat
jelas (bukan ukuran metrologi). Template tidak diambil dari frame pertama
karena pada video samping lubang belum tampak di awal rekaman.

Jika video/kamera berubah, kalibrasi ulang REFERENCE_ELLIPSE, TEMPLATE_BOX,
dan berkas calib/template_side.png.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Koordinat referensi pada resolusi 640 x 480; panjang sumbu adalah diameter.
# Dikunci pada bukaan tabung sekitar t=15s pada samples/side.webm.
REFERENCE_ELLIPSE = ((190.0, 171.0), (24.0, 44.0), -2.0)
TEMPLATE_BOX = (168, 138, 48, 64)
SEARCH_BOX = (0.00, 0.16, 0.52, 0.55)
MIN_MATCH = 0.58
MIN_MATCH_TRACK = 0.50
MIN_ECC = 0.72
CALIB_SIZE = (640, 480)
SCALES = (0.80, 0.88, 0.95, 1.0, 1.07, 1.15)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = SCRIPT_DIR / "calib" / "template_side.png"
DEFAULT_INPUT = SCRIPT_DIR / "samples" / "side.webm"


def transformed_ellipse(ellipse, matrix):
    (cx, cy), (a, b), angle = ellipse
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    r = np.deg2rad(angle)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    points = np.column_stack((a / 2 * np.cos(t), b / 2 * np.sin(t))) @ rot.T
    points += (cx, cy)
    points = points @ matrix[:, :2].T + matrix[:, 2]
    return cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))


def _gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def load_template(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Template tidak dapat dibaca: {path}")
    return image


class PinholeTracker:
    def __init__(self, first_frame, template=None):
        h, w = first_frame.shape[:2]
        sx, sy = w / CALIB_SIZE[0], h / CALIB_SIZE[1]
        x, y, tw, th = TEMPLATE_BOX
        self.box = tuple(
            int(round(v * s)) for v, s in zip((x, y, tw, th), (sx, sy, sx, sy))
        )
        x, y, tw, th = self.box
        gray = _gray(first_frame)
        if template is not None:
            if template.shape != (th, tw):
                template = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
            self.template = template.copy()
        else:
            self.template = gray[y : y + th, x : x + tw].copy()
            if self.template.size == 0:
                raise RuntimeError("TEMPLATE_BOX di luar frame.")
        self.ellipse = transformed_ellipse(
            REFERENCE_ELLIPSE,
            np.array([[sx, 0, 0], [0, sy, 0]], np.float32),
        )
        self.last_center = None

    def _match(self, gray, search_box, min_match):
        h, w = gray.shape
        x0, y0, x1, y1 = [int(v * s) for v, s in zip(search_box, (w, h, w, h))]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        roi = gray[y0:y1, x0:x1]
        tx, ty, tw, th = self.box
        best = None
        for scale in SCALES:
            template = cv2.resize(
                self.template,
                (max(8, round(tw * scale)), max(8, round(th * scale))),
            )
            hh, ww = template.shape
            if hh >= roi.shape[0] or ww >= roi.shape[1]:
                continue
            scores = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            if best is None or score > best[0]:
                best = (score, location, template)
        if best is None or best[0] < min_match:
            return None
        score, (dx, dy), template = best
        hh, ww = template.shape
        return score, x0 + dx, y0 + dy, template

    def detect(self, frame):
        gray = _gray(frame)
        matched = self._match(gray, SEARCH_BOX, MIN_MATCH)
        if matched is None and self.last_center is not None:
            h, w = gray.shape
            cx, cy = self.last_center
            pad_x, pad_y = 0.14, 0.16
            local = (
                max(0.0, (cx / w) - pad_x),
                max(0.0, (cy / h) - pad_y),
                min(1.0, (cx / w) + pad_x),
                min(1.0, (cy / h) + pad_y),
            )
            matched = self._match(gray, local, MIN_MATCH_TRACK)
        if matched is None:
            return None
        score, px, py, template = matched
        hh, ww = template.shape
        tx, ty, tw, th = self.box
        patch = gray[py : py + hh, px : px + ww]
        if patch.shape[:2] != template.shape:
            return None
        warp = np.eye(2, 3, dtype=np.float32)
        # ECC memetakan koordinat template ke patch frame saat ini.
        try:
            cc, candidate = cv2.findTransformECC(
                template,
                patch,
                warp.copy(),
                cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4),
                None,
                3,
            )
            singular = np.linalg.svd(candidate[:, :2], compute_uv=False)
            if (
                cc >= MIN_ECC
                and np.all((singular > 0.80) & (singular < 1.25))
                and np.linalg.det(candidate[:, :2]) > 0
                and np.linalg.norm(candidate[:, 2]) < 12
            ):
                warp = candidate
        except cv2.error:
            pass
        origin = np.array(
            [[ww / tw, 0, -tx * ww / tw], [0, hh / th, -ty * hh / th], [0, 0, 1]],
            np.float32,
        )
        affine = np.vstack((warp, [0, 0, 1])) @ origin
        affine[0, 2] += px
        affine[1, 2] += py
        ellipse = transformed_ellipse(self.ellipse, affine[:2])
        self.last_center = ellipse[0]
        return {"ellipse": ellipse, "center": ellipse[0], "score": float(score)}


def draw(frame, result):
    out = frame.copy()
    if result is not None:
        cv2.ellipse(out, result["ellipse"], (0, 255, 0), 1, cv2.LINE_AA)
        center = tuple(int(round(v)) for v in result["center"])
        cv2.circle(out, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
        text = f"PINHOLE {center[0]},{center[1]}"
    else:
        text = "PINHOLE: tidak terdeteksi"
    cv2.putText(
        out,
        text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def serialize(index, result):
    if result is None:
        return {"frame": index, "detected": False, "center": None, "score": None}
    cx, cy = result["center"]
    (ex, ey), (a, b), angle = result["ellipse"]
    return {
        "frame": index,
        "detected": True,
        "center": [round(float(cx), 2), round(float(cy), 2)],
        "axes": [round(float(a), 2), round(float(b), 2)],
        "angle": round(float(angle), 2),
        "score": round(float(result["score"]), 4),
    }


def probe_timestamps(input_video):
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time,pkt_duration_time",
                "-of",
                "json",
                input_video,
            ]
        )
    )
    frames = info.get("frames") or []
    timestamps = [
        float(f["best_effort_timestamp_time"])
        for f in frames
        if "best_effort_timestamp_time" in f
    ]
    return info, timestamps


def write_concat_video(folder: Path, records, timestamps, info, input_video, output_video):
    if timestamps and len(timestamps) == len(records):
        durations = np.diff(timestamps).tolist()
        last = float(
            info["frames"][-1].get(
                "pkt_duration_time",
                np.median(durations) if durations else 1 / 30,
            )
        )
        frame_durations = durations + [last]
    else:
        fps = 30.0
        if timestamps and len(timestamps) > 1:
            fps = max(1.0, (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))
        frame_durations = [1.0 / fps] * len(records)
    lines = []
    for i, duration in enumerate(frame_durations):
        duration = max(0.001, float(duration))
        lines.extend(
            [f"file '{i:05d}.png'", "option framerate 1000", f"duration {duration:.6f}"]
        )
    # Frame terakhir diulang sebagai penanda akhir durasi untuk concat.
    lines.append(f"file '{len(records) - 1:05d}.png'")
    lines.append("option framerate 1000")
    manifest = folder / "frames.txt"
    manifest.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest),
            "-i",
            input_video,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "vfr",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_video,
        ],
        check=True,
    )


def main(input_video="samples/side.webm", output_video="hasil.mp4", template_path=None):
    input_video = str(input_video)
    output_video = str(output_video)
    if Path(input_video).resolve() == Path(output_video).resolve():
        raise ValueError("Gunakan nama output berbeda dari input.")
    info, timestamps = probe_timestamps(input_video)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {input_video}")
    template = load_template(Path(template_path) if template_path else DEFAULT_TEMPLATE)
    records = []
    try:
        with tempfile.TemporaryDirectory(prefix="pinhole_") as directory:
            folder = Path(directory)
            tracker = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if tracker is None:
                    tracker = PinholeTracker(frame, template=template)
                result = tracker.detect(frame)
                index = len(records)
                if not cv2.imwrite(str(folder / f"{index:05d}.png"), draw(frame, result)):
                    raise RuntimeError("Gagal menulis frame.")
                records.append(serialize(index, result))
            if not records:
                raise RuntimeError("Tidak ada frame yang terbaca.")
            write_concat_video(folder, records, timestamps, info, input_video, output_video)
    finally:
        cap.release()
    Path(output_video).with_suffix(".json").write_text(
        json.dumps(records, indent=2)
    )
    detected = sum(r["detected"] for r in records)
    print(f"Selesai: {output_video}; deteksi {detected}/{len(records)} frame")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT if DEFAULT_INPUT.is_file() else "samples/side.webm"),
    )
    parser.add_argument("output", nargs="?", default="hasil.mp4")
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Gambar grayscale template lubang (crop kalibrasi).",
    )
    args = parser.parse_args()
    main(args.input, args.output, args.template)
