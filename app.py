"""Aplikasi pelacakan pinhole 2 tampilan (Top & Side) dalam 1 file .py lengkap dengan tombol kalibrasi, slider parameter, dan visualisasi panduan (kotak pencarian kuning, kotak template biru, oval hijau)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pinhole.params import SIDE_SAMPLE, defaults_for
from pinhole.vision import geometry, tracker_for


class DualWindow(QMainWindow):
    def __init__(self, top_path: str, side_path: str):
        super().__init__()
        self.setWindowTitle("Pinhole Tracker - Kalibrasi & Panduan")
        self.resize(1400, 800)

        self.top_path = top_path
        self.side_path = side_path

        self.cap_top = None
        self.cap_side = None
        self.tracker_top = None
        self.tracker_side = None

        self.params_top = defaults_for("top")
        self.params_side = defaults_for("side")

        self.show_guides = True  # Tampilkan garis kuning, biru, oval hijau
        self.is_playing = False

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: Videos & Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        ctrl_layout = QHBoxLayout()
        self.btn_top_src = QPushButton("Pilih Video Top")
        self.btn_top_src.clicked.connect(self.select_top_video)
        ctrl_layout.addWidget(self.btn_top_src)

        self.btn_side_src = QPushButton("Pilih Video Side")
        self.btn_side_src.clicked.connect(self.select_side_video)
        ctrl_layout.addWidget(self.btn_side_src)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.start_video)
        ctrl_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_video)
        ctrl_layout.addWidget(self.btn_pause)

        # Tombol Kalibrasi / Toggle Guide
        self.chk_kalibrasi = QCheckBox("Kalibrasi (Panduan Kuning/Biru/Hijau)")
        self.chk_kalibrasi.setChecked(True)
        self.chk_kalibrasi.toggled.connect(self.toggle_kalibrasi)
        ctrl_layout.addWidget(self.chk_kalibrasi)

        ctrl_layout.addStretch()
        left_layout.addLayout(ctrl_layout)

        vid_layout = QHBoxLayout()

        top_box = QVBoxLayout()
        self.lbl_top_info = QLabel(f"TOP: {Path(top_path).name if top_path else 'Tidak ada'}")
        top_box.addWidget(self.lbl_top_info)
        self.lbl_top = QLabel("Video Top belum dimuat")
        self.lbl_top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_top.setStyleSheet("background-color: black; color: white;")
        self.lbl_top.setMinimumSize(480, 360)
        top_box.addWidget(self.lbl_top)
        vid_layout.addLayout(top_box)

        side_box = QVBoxLayout()
        self.lbl_side_info = QLabel(f"SIDE: {Path(side_path).name if side_path else 'Tidak ada'}")
        side_box.addWidget(self.lbl_side_info)
        self.lbl_side = QLabel("Video Side belum dimuat")
        self.lbl_side.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_side.setStyleSheet("background-color: black; color: white;")
        self.lbl_side.setMinimumSize(480, 360)
        side_box.addWidget(self.lbl_side)
        vid_layout.addLayout(side_box)

        left_layout.addLayout(vid_layout)
        main_layout.addWidget(left_widget, stretch=3)

        # Right panel: Sliders for Side tracking & calibration
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        slider_widget = QWidget()
        self.slider_layout = QFormLayout(slider_widget)
        
        self.slider_labels = {}
        self.sliders = {}

        slider_specs = [
            ("min_match", "Min Match", 0.0, 1.0, 0.01),
            ("scale_min", "Scale Min", 0.5, 2.0, 0.05),
            ("scale_max", "Scale Max", 0.5, 2.0, 0.05),
            ("cx", "Pusat X (cx)", 0.0, 640.0, 0.5),
            ("cy", "Pusat Y (cy)", 0.0, 480.0, 0.5),
            ("axis_a", "Axis A", 1.0, 200.0, 0.5),
            ("axis_b", "Axis B", 1.0, 200.0, 0.5),
            ("angle", "Angle", -180.0, 180.0, 0.5),
            ("tx", "Template X (tx)", 0.0, 640.0, 1.0),
            ("ty", "Template Y (ty)", 0.0, 480.0, 1.0),
            ("tw", "Template Width", 3.0, 300.0, 1.0),
            ("th", "Template Height", 3.0, 300.0, 1.0),
            ("ecc_shift", "ECC Shift", 1, 100, 1.0),
        ]

        self.slider_layout.addRow(QLabel("<b>Kalibrasi & Parameter Samping (Side)</b>"))

        for key, label_text, min_v, max_v, step in slider_specs:
            val = self.params_side.get(key, min_v)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            
            int_val = int((val - min_v) / (max_v - min_v) * 1000)
            slider.setValue(max(0, min(1000, int_val)))

            val_lbl = QLabel(f"{val:.2f}" if isinstance(val, float) else f"{val}")
            self.slider_labels[key] = val_lbl

            row_layout = QHBoxLayout()
            row_layout.addWidget(slider)
            row_layout.addWidget(val_lbl)

            self.slider_layout.addRow(QLabel(label_text), row_layout)
            self.sliders[key] = (slider, min_v, max_v)
            slider.valueChanged.connect(lambda v, k=key: self.on_slider_changed(k))

        scroll.setWidget(slider_widget)
        main_layout.addWidget(scroll, stretch=1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frames)

        self.init_videos(top_path, side_path)

    def toggle_kalibrasi(self, checked):
        self.show_guides = checked

    def on_slider_changed(self, key):
        slider, min_v, max_v = self.sliders[key]
        ratio = slider.value() / 1000.0
        val = min_v + ratio * (max_v - min_v)
        if isinstance(min_v, int) and isinstance(max_v, int):
            val = int(round(val))
        self.params_side[key] = val
        self.slider_labels[key].setText(f"{val:.2f}" if isinstance(val, float) else f"{val}")

    def init_videos(self, top_path: str, side_path: str):
        if self.cap_top:
            self.cap_top.release()
        if self.cap_side:
            self.cap_side.release()

        if top_path and Path(top_path).is_file():
            self.top_path = top_path
            self.cap_top = cv2.VideoCapture(top_path)
            ok, frame = self.cap_top.read()
            if ok:
                self.tracker_top = tracker_for("top", frame)
                self.show_frame(frame, self.lbl_top, self.tracker_top, self.params_top, "TOP")
            self.lbl_top_info.setText(f"TOP: {Path(top_path).name}")

        if side_path and Path(side_path).is_file():
            self.side_path = side_path
            self.cap_side = cv2.VideoCapture(side_path)
            ok, frame = self.cap_side.read()
            if ok:
                self.tracker_side = tracker_for("side", frame)
                self.show_frame(frame, self.lbl_side, self.tracker_side, self.params_side, "SIDE")
            self.lbl_side_info.setText(f"SIDE: {Path(side_path).name}")

    def select_top_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Video Top", str(SIDE_SAMPLE.parent), "Video Files (*.mp4 *.webm *.avi *.mov)"
        )
        if path:
            self.pause_video()
            self.init_videos(path, self.side_path)

    def select_side_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih Video Side", str(SIDE_SAMPLE.parent), "Video Files (*.mp4 *.webm *.avi *.mov)"
        )
        if path:
            self.pause_video()
            self.init_videos(self.top_path, path)

    def start_video(self):
        self.is_playing = True
        self.timer.start(30)

    def pause_video(self):
        self.is_playing = False
        self.timer.stop()

    def process_frames(self):
        if self.cap_top and self.cap_top.isOpened():
            ok, frame = self.cap_top.read()
            if not ok:
                self.cap_top.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap_top.read()
            if ok and self.tracker_top:
                self.show_frame(frame, self.lbl_top, self.tracker_top, self.params_top, "TOP")

        if self.cap_side and self.cap_side.isOpened():
            ok, frame = self.cap_side.read()
            if not ok:
                self.cap_side.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap_side.read()
            if ok and self.tracker_side:
                self.show_frame(frame, self.lbl_side, self.tracker_side, self.params_side, "SIDE")

    def show_frame(self, frame, label_widget, tracker, params, role):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            result, _note = tracker.detect(gray, params)
        except Exception:
            result = None

        disp = frame.copy()
        height, width = disp.shape[:2]

        # Jika mode kalibrasi/panduan aktif, gambar kotak pencarian kuning, kotak template biru, dan oval hijau kalibrasi
        if self.show_guides:
            try:
                calib_ellipse, (tx, ty, tw, th), (x0, y0, x1, y1) = geometry(params, width, height)
                # 1. Kotak pencarian (Kuning)
                cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 1, cv2.LINE_AA)
                # 2. Kotak template (Biru)
                cv2.rectangle(disp, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 1, cv2.LINE_AA)
                # 3. Kalibrasi oval referensi (Hijau)
                cv2.ellipse(disp, calib_ellipse, (0, 255, 0), 1, cv2.LINE_AA)
            except Exception:
                pass

        if result is not None:
            # Hasil deteksi aktual (Oval Hijau tebal & Titik merah pusat)
            cv2.ellipse(disp, result["ellipse"], (0, 255, 0), 2, cv2.LINE_AA)
            cx, cy = result["center"]
            cv2.circle(disp, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1, cv2.LINE_AA)
            text = f"{role} ({int(round(cx))},{int(round(cy))})"
        else:
            text = f"{role}: Tidak terdeteksi"

        cv2.putText(
            disp, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )

        h, w, ch = disp.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label_widget.setPixmap(QPixmap.fromImage(qimg).scaled(
            label_widget.width(), label_widget.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))


def main():
    app = QApplication(sys.argv)
    video_file = str(SIDE_SAMPLE) if SIDE_SAMPLE.is_file() else "side.webm"
    window = DualWindow(video_file, video_file)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
