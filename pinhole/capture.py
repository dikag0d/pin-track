"""Capture kamera/video dan pelacakan di thread terpisah dari GUI."""

from __future__ import annotations

import glob
import math
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

from pinhole.params import validate_parameters
from pinhole.vision import prepare_image, tracker_for

BACKENDS = [
    ("Auto", cv2.CAP_ANY),
    ("V4L2 / Linux", cv2.CAP_V4L2),
    ("DirectShow / Windows", cv2.CAP_DSHOW),
    ("MSMF / Windows", cv2.CAP_MSMF),
]


def detect_cameras():
    """Deteksi kamera yang terhubung dari /sys/class/video4linux.

    Mengembalikan list of (label, index) untuk setiap perangkat capture
    (hanya index=0 di sysfs, yaitu node capture utama).
    Kamera internal (integrated) dan eksternal (USB) diberi label berbeda.
    Jika tidak ada kamera terdeteksi, kembalikan fallback indeks 0–9.
    """
    cameras = []
    if sys.platform.startswith("linux"):
        sysfs_dirs = sorted(
            glob.glob("/sys/class/video4linux/video*"),
            key=lambda p: int(os.path.basename(p).replace("video", "")),
        )
        for dev_path in sysfs_dirs:
            dev_name = os.path.basename(dev_path)
            try:
                num = int(dev_name.replace("video", ""))
            except ValueError:
                continue

            # Hanya ambil node capture utama (index 0 di sysfs).
            index_file = os.path.join(dev_path, "index")
            if os.path.exists(index_file):
                try:
                    with open(index_file) as fh:
                        if fh.read().strip() != "0":
                            continue
                except OSError:
                    pass

            name_file = os.path.join(dev_path, "name")
            hw_name = ""
            if os.path.exists(name_file):
                try:
                    with open(name_file) as fh:
                        hw_name = fh.read().strip()
                except OSError:
                    pass

            lower = hw_name.lower()
            if any(kw in lower for kw in ("integrated", "internal", "built-in")):
                tag = "Internal"
            else:
                tag = "Eksternal USB"

            if hw_name:
                label = f"Kamera {num}: {hw_name} ({tag})"
            else:
                label = f"Kamera {num}"

            cameras.append((label, num))

    if not cameras:
        cameras = [(f"Kamera indeks {i}", i) for i in range(10)]

    return cameras


class CaptureWorker(threading.Thread):
    """Capture dan tracking berjalan di luar thread GUI."""

    def __init__(self, source, backend, size, fps, p, reference, role):
        super().__init__(daemon=True)
        self.source = source
        self.backend = backend
        self.size = size
        self.requested_fps = fps
        self.role = role
        self.initial_reference = (
            reference.copy() if reference is not None else None
        )
        self.stop_event = threading.Event()
        self.paused = threading.Event()
        self.lock = threading.Lock()
        self.parameters = p.copy()
        self.pending_reference = None

        self.latest = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.commands = queue.Queue()

    def configure(self, p, reference=None):
        with self.lock:
            self.parameters = p.copy()
            if reference is not None:
                self.pending_reference = reference.copy()

    def publish(self, packet):
        try:
            self.latest.get_nowait()
        except queue.Empty:
            pass
        self.latest.put_nowait(packet)

    def apply_hardware(self, cap, settings):
        if not isinstance(self.source, int):
            self.events.put(("message", "Kontrol hardware hanya untuk kamera."))
            return

        backend = cap.getBackendName().upper()
        auto_mode, exposure, gain = settings
        messages = []

        if auto_mode is not None:
            if "V4L" in backend:
                value = 0.75 if auto_mode else 0.25
            elif "DSHOW" in backend:
                value = 1.0 if auto_mode else 0.0
            else:
                value = None
                messages.append(
                    f"Auto exposure {backend}: atur melalui aplikasi driver."
                )

            if value is not None:
                ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
                messages.append(
                    "Perintah auto exposure diterima."
                    if ok else "Auto exposure ditolak driver."
                )

        if auto_mode is False:
            ok = cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            readback = cap.get(cv2.CAP_PROP_EXPOSURE)
            messages.append(
                f"Exposure: {'diterima' if ok else 'ditolak'}, "
                f"readback={readback:g}."
            )

        ok = cap.set(cv2.CAP_PROP_GAIN, float(gain))
        readback = cap.get(cv2.CAP_PROP_GAIN)
        messages.append(
            f"Gain: {'diterima' if ok else 'ditolak'}, readback={readback:g}."
        )
        self.events.put(("message", " ".join(messages)))

    def run(self):
        cap = None
        try:
            camera = isinstance(self.source, int)
            cap = cv2.VideoCapture(
                self.source, self.backend if camera else cv2.CAP_ANY
            )
            if not cap.isOpened():
                raise RuntimeError(
                    "Sumber tidak dapat dibuka. Periksa indeks, path, "
                    "backend, izin kamera, atau aplikasi lain."
                )

            if camera:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or not 0.1 <= fps <= 240:
                fps = self.requested_fps

            self.events.put((
                "message",
                f"Terbuka: {cap.getBackendName()} | FPS dilaporkan {fps:.2f}",
            ))

            tracker = (
                tracker_for(self.role, self.initial_reference)
                if self.initial_reference is not None
                else None
            )
            raw = None
            index = -1
            failures = 0

            while not self.stop_event.is_set():
                tick = time.monotonic()

                with self.lock:
                    p = self.parameters.copy()
                    reference = self.pending_reference
                    self.pending_reference = None

                if reference is not None:
                    tracker = tracker_for(self.role, reference)

                while True:
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self.apply_hardware(cap, command)
                    except cv2.error as exc:
                        self.events.put(("message", f"Kontrol kamera: {exc}"))

                if camera or not self.paused.is_set() or raw is None:
                    ok, next_frame = cap.read()

                    if not ok and not camera and p["loop"]:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, next_frame = cap.read()
                        index = -1

                    if not ok:
                        if not camera:
                            self.events.put(("message", "Video selesai."))
                            break
                        failures += 1
                        if failures >= 10:
                            raise RuntimeError("Kamera berhenti mengirim frame.")
                        self.stop_event.wait(0.1)
                        continue

                    failures = 0
                    raw = next_frame
                    index += 1

                if tracker is None:
                    tracker = tracker_for(self.role, raw)
                    self.events.put(("reference", raw.copy()))

                result = None
                adjusted = raw
                mask = np.zeros(raw.shape[:2], np.uint8)
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

                try:
                    validate_parameters(p, top=(self.role == "top"))
                    adjusted, mask, gray = prepare_image(raw, p)
                    result, note = tracker.detect(gray, p)
                except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                    note = str(exc)

                elapsed_ms = (time.monotonic() - tick) * 1000
                self.publish({
                    "raw": raw,
                    "adjusted": adjusted,
                    "mask": mask,
                    "gray": gray,
                    "result": result,
                    "note": note,
                    "index": index,
                    "ms": elapsed_ms,
                })

                if not camera:
                    period = 1.0 / (30 if self.paused.is_set() else fps)
                    self.stop_event.wait(
                        max(0, period - (time.monotonic() - tick))
                    )

        except Exception as exc:
            self.events.put(("message", f"ERROR: {exc}"))
        finally:
            if cap is not None:
                cap.release()
