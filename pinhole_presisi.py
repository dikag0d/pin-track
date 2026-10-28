"""Pelacak oval lubang jarum untuk rekaman samping (side.webm).

Membutuhkan numpy, opencv-python, ffmpeg/ffprobe.

Jalankan:
    python pinhole_presisi.py samples/side_train.mp4 hasil.mp4

Oval lubang mengikuti template matching multi-skala + ECC di kotak pencarian
SIDE. Bukan ukuran metrologi. Kalibrasi ulang di GUI jika kamera berubah.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pinhole.params import SIDE_SAMPLE, defaults_for
from pinhole.vision import SidePinholeTracker

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SIDE_SAMPLE


def _gray(frame):
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


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


def main(input_video="samples/side_train.mp4", output_video="hasil.mp4"):
    input_video = str(input_video)
    output_video = str(output_video)
    if Path(input_video).resolve() == Path(output_video).resolve():
        raise ValueError("Gunakan nama output berbeda dari input.")
    info, timestamps = probe_timestamps(input_video)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Video tidak dapat dibuka: {input_video}")
    records = []
    params = defaults_for("side")
    try:
        with tempfile.TemporaryDirectory(prefix="pinhole_") as directory:
            folder = Path(directory)
            tracker = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if tracker is None:
                    tracker = SidePinholeTracker(frame)
                result, _note = tracker.detect(_gray(frame), params)
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
        default=str(DEFAULT_INPUT if DEFAULT_INPUT.is_file() else "samples/side_train.mp4"),
    )
    parser.add_argument("output", nargs="?", default="hasil.mp4")
    args = parser.parse_args()
    main(args.input, args.output)
