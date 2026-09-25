"""Antarmuka PySide6: panel SIDE kiri, panel TOP kanan.

Kedua panel melacak oval lubang dan ujung bebas benang coklat.
"""

from __future__ import annotations

import base64
import json
import math
import os
import queue
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from pinhole.capture import BACKENDS, CaptureWorker, detect_cameras
from pinhole.params import (
    DEFAULTS,
    GEOMETRY_FIELDS,
    HSV_FIELDS,
    IMAGE_FIELDS,
    SIDE_REFERENCE,
    THREAD_HSV_FIELDS,
    THREAD_MEASURE_FIELDS,
    THREAD_OVAL_FIELDS,
    THREAD_SMOOTH_FIELDS,
    TRACK_FIELDS,
    defaults_for,
    validate_parameters,
)
from pinhole.thread_tip import (
    draw_insertion_overlay, draw_thread, draw_thread_guide, insertion_paths,
)
from pinhole.vision import draw_detection, draw_guides, prepare_image, tracker_for

RESOLUTIONS = ("640x480", "800x600", "1280x720", "1920x1080")


class VideoView(QWidget):
    selected = Signal(float, float, float, float)

    def __init__(self):
        super().__init__()
        self.image = None
        self.image_rect = QRectF()
        self.selecting = False
        self.drag_start = None
        self.drag_end = None
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame):
        frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        self.image = QImage(
            frame.data, width, height, frame.strides[0],
            QImage.Format.Format_BGR888,
        ).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#08111c"))

        if self.image is None:
            painter.setPen(QColor("#93a6bc"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Pilih sumber lalu klik Mulai",
            )
            return

        width, height = self.image.width(), self.image.height()
        scale = min(self.width() / width, self.height() / height)
        self.image_rect = QRectF(
            (self.width() - width * scale) / 2,
            (self.height() - height * scale) / 2,
            width * scale,
            height * scale,
        )
        painter.drawImage(self.image_rect, self.image)

        if self.drag_start is not None and self.drag_end is not None:
            painter.setPen(QPen(QColor("#ffdc63"), 2))
            painter.drawRect(
                QRectF(self.drag_start, self.drag_end).normalized()
            )

    def mousePressEvent(self, event):
        if (
            self.selecting
            and self.image is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self.image_rect.contains(event.position())
        ):
            self.drag_start = event.position()
            self.drag_end = event.position()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_start is None or self.image is None:
            return

        rect = self.image_rect
        if rect.width() <= 0 or rect.height() <= 0:
            return

        def convert(point):
            x = (point.x() - rect.left()) / rect.width()
            y = (point.y() - rect.top()) / rect.height()
            return (
                float(np.clip(x, 0, 1) * self.image.width()),
                float(np.clip(y, 0, 1) * self.image.height()),
            )

        x0, y0 = convert(self.drag_start)
        x1, y1 = convert(event.position())
        self.drag_start = None
        self.drag_end = None
        self.update()

        if abs(x1 - x0) >= 3 and abs(y1 - y0) >= 3:
            self.selected.emit(
                min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
            )


class CameraPane(QGroupBox):
    def __init__(self, title, role, owner, initial_file=""):
        super().__init__(title)
        self.role = role
        self.top = role == "top"
        self.owner = owner
        self.worker = None
        self.packet = None
        self.reference = None
        self.previous_source = None
        self.active_source = None
        self.frozen = None
        self.shown = None
        self.old_parameters = None
        self.was_paused = False
        self.loading = False
        self.recorder = None
        self.recording = False
        self.record_path = None
        self.stopping_since = None
        self.inputs = {}
        self.checks = {}

        if role == "side" and SIDE_REFERENCE.is_file():
            self.reference = cv2.imread(str(SIDE_REFERENCE))

        layout = QVBoxLayout(self)

        self.source = QComboBox()
        self._detected_cameras = detect_cameras()
        for label, cam_index in self._detected_cameras:
            self.source.addItem(label, cam_index)
        self.source.addItem("File video", "file")
        file_idx = len(self._detected_cameras)
        if initial_file:
            self.source.setCurrentIndex(file_idx)
        elif len(self._detected_cameras) >= 2 and role == "side":
            self.source.setCurrentIndex(1)
        else:
            self.source.setCurrentIndex(0)

        self.backend = QComboBox()
        for label, value in BACKENDS:
            self.backend.addItem(label, value)

        self.resolution = QComboBox()
        for size in RESOLUTIONS:
            self.resolution.addItem(size)

        self.fps = QDoubleSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setDecimals(0)
        self.fps.setValue(30)
        self.fps.setSuffix(" FPS")

        source_row = QHBoxLayout()
        for widget in (self.source, self.backend, self.resolution, self.fps):
            source_row.addWidget(widget)
        layout.addLayout(source_row)

        placeholder = (
            "Path video, misalnya top.webm"
            if self.top else "Path video, misalnya samples/side.webm"
        )
        self.file_path = QLineEdit(initial_file)
        self.file_path.setPlaceholderText(placeholder)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse)

        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path, 1)
        file_row.addWidget(self.browse_button)
        layout.addLayout(file_row)

        self.start_button = QPushButton("Mulai")
        self.stop_button = QPushButton("Stop")
        self.pause_button = QPushButton("Pause video")
        self.pause_button.setCheckable(True)
        self.snapshot_button = QPushButton("Snapshot")
        self.record_button = QPushButton("Rekam")
        self.record_button.setCheckable(True)

        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.pause_button.toggled.connect(self.pause)
        self.snapshot_button.clicked.connect(self.snapshot)
        self.record_button.toggled.connect(self.toggle_recording)

        button_row = QHBoxLayout()
        for button in (
            self.start_button, self.stop_button,
            self.pause_button, self.snapshot_button,
            self.record_button,
        ):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.view_mode = QComboBox()
        self.view_mode.addItems([
            "Overlay deteksi",
            "Gambar asli",
            "Gambar terkoreksi",
            "Mask HSV",
            "Grayscale detektor",
            "Mask benang",
        ])
        self.view_mode.currentIndexChanged.connect(self.render)
        layout.addWidget(self.view_mode)

        self.view = VideoView()
        self.view.selected.connect(self.select_region)
        layout.addWidget(self.view, 1)

        self.stats = QLabel("Belum ada frame.")
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)

        self.thread_readout = QLabel()
        self.thread_readout.setWordWrap(True)
        self.thread_readout.setTextFormat(Qt.TextFormat.RichText)
        self._set_thread_readout("Ujung benang: —")
        layout.addWidget(self.thread_readout)

        self.message = QLabel(
            "Preset awal untuk top.webm. Kalibrasi ulang untuk kamera/video lain."
            if self.top else
            "Preset SIDE untuk samples/side.webm. Kalibrasi ulang jika kamera berubah."
        )
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        tabs = QTabWidget()
        tabs.setMinimumHeight(230)
        layout.addWidget(tabs)

        image_form = self.add_tab(tabs, "Citra")
        self.add_fields(image_form, IMAGE_FIELDS, sliders=True)
        self.add_check(image_form, "loop", "Ulangi file video")

        hsv_form = self.add_tab(tabs, "HSV")
        self.add_check(hsv_form, "hsv_on", "Gunakan mask HSV pada detektor")
        self.add_check(hsv_form, "invert", "Balik mask")
        self.add_fields(hsv_form, HSV_FIELDS, sliders=True)
        note = QLabel("H: 0–179. H minimum > maksimum mendukung rentang melingkar.")
        note.setWordWrap(True)
        hsv_form.addRow(note)

        tracking_form = self.add_tab(tabs, "Tracking")
        self.add_check(tracking_form, "ecc_on", "Aktifkan ECC")
        self.add_check(tracking_form, "guides", "Tampilkan panduan referensi")
        self.add_fields(tracking_form, TRACK_FIELDS)

        thread_form = self.add_tab(tabs, "Benang")
        self.add_check(thread_form, "thread_on", "Aktifkan deteksi ujung benang")
        self.add_check(
            thread_form,
            "path_overlay",
            "Jalur insertion: benang di dalam bukaan jarum",
        )
        self.add_check(
            thread_form, "thread_from_right", "Benang masuk dari tepi kanan"
        )
        self.add_check(
            thread_form,
            "thread_oval_lock",
            "Kunci ukuran oval (lebar benang tetap)",
        )
        self.lock_oval_button = QPushButton("Kunci oval ke lebar benang")
        self.lock_oval_button.clicked.connect(self.lock_thread_oval)
        thread_form.addRow(self.lock_oval_button)
        oval_note = QLabel(
            "Lebar benang tidak berubah, jadi ukuran oval dikunci. "
            "Posisi tetap mengikuti ujung. Sudut 0: lebar diukur vertikal. "
            "Drag \"Oval ujung benang\" di tab Kalibrasi untuk menyesuaikan, "
            "atau pakai tombol di atas agar lebar diambil dari benang yang terukur. "
            "Pratinjau X/Y hanya menempatkan oval pada frame beku."
        )
        oval_note.setWordWrap(True)
        thread_form.addRow(oval_note)
        self.add_fields(thread_form, THREAD_OVAL_FIELDS)
        thread_note = QLabel(
            "Ujung bebas (tidak terpotong tepi gambar) diukur subpiksel pada "
            "mask sebelum penutupan. Saat perpindahan di bawah radius diam, "
            "koordinat diratakan supaya tidak bergetar. Di atas radius itu, "
            "marker langsung mengikuti pengukuran baru. Alpha lebih kecil "
            "berarti lebih halus. Matikan 'dari tepi kanan' jika benang masuk "
            "dari kiri; yang dilacak tetap ujung bebasnya."
        )
        thread_note.setWordWrap(True)
        thread_form.addRow(thread_note)
        self.add_fields(thread_form, THREAD_HSV_FIELDS, sliders=True)
        hsv_note = QLabel(
            "HSV benang coklat/tembaga, terpisah dari mask lubang. "
            "Mask dihitung dari gambar asli. H minimum > maksimum berarti "
            "rentang melingkar. Gunakan tampilan Mask benang untuk menyetel. "
            "Batas sisi masuk: komponen harus mencapai sisi tempat benang masuk."
        )
        hsv_note.setWordWrap(True)
        thread_form.addRow(hsv_note)
        self.add_fields(thread_form, THREAD_MEASURE_FIELDS)
        smooth_note = QLabel(
            "Radius sangat diam memakai perataan lebih kuat. "
            "Radius itu harus lebih kecil atau sama dengan radius diam."
        )
        smooth_note.setWordWrap(True)
        thread_form.addRow(smooth_note)
        self.add_fields(thread_form, THREAD_SMOOTH_FIELDS)

        geometry_form = self.add_tab(tabs, "Kalibrasi")
        self.calibration_button = QPushButton("Bekukan untuk kalibrasi")
        self.cancel_button = QPushButton("Batal kalibrasi")
        self.cancel_button.setEnabled(False)
        self.calibration_button.clicked.connect(self.calibrate)
        self.cancel_button.clicked.connect(self.cancel_calibration)

        self.selection_mode = QComboBox()
        self.selection_mode.addItems([
            "Oval", "Template", "Area pencarian", "Oval ujung benang",
        ])
        geometry_form.addRow(self.calibration_button)
        geometry_form.addRow(self.cancel_button)
        geometry_form.addRow("Objek yang di-drag", self.selection_mode)

        note = QLabel(
            "Bekukan, drag pada gambar, lalu terapkan. "
            "Kuning: template; biru: search; hijau: oval lubang; "
            "cyan: oval ujung benang. "
            "Angka geometri memakai referensi 640 x 480. "
            "Oval benang yang dikunci mengikuti ujung saat pelacakan."
        )
        note.setWordWrap(True)
        geometry_form.addRow(note)
        self.add_fields(geometry_form, GEOMETRY_FIELDS)

        hardware_form = self.add_tab(tabs, "Kamera")
        self.auto_exposure = QComboBox()
        self.auto_exposure.addItem("Biarkan pengaturan driver", None)
        self.auto_exposure.addItem("Manual", False)
        self.auto_exposure.addItem("Otomatis", True)

        self.exposure = QDoubleSpinBox()
        self.exposure.setRange(-20, 10000)
        self.exposure.setValue(-6)

        self.gain = QDoubleSpinBox()
        self.gain.setRange(0, 1000)
        self.gain.setValue(0)

        self.hardware_button = QPushButton("Terapkan ke kamera")
        self.hardware_button.clicked.connect(self.apply_hardware)
        hardware_form.addRow("Auto exposure", self.auto_exposure)
        hardware_form.addRow("Exposure", self.exposure)
        hardware_form.addRow("Gain", self.gain)
        hardware_form.addRow(self.hardware_button)

        note = QLabel(
            "Satuan exposure/gain bergantung driver. "
            "Exposure manual dikirim saat mode Manual dipilih. "
            "Hasil perintah dan nilai readback muncul di atas."
        )
        note.setWordWrap(True)
        hardware_form.addRow(note)

        if role == "side":
            self.fill_parameters(defaults_for("side"))

        self.source.currentIndexChanged.connect(self.update_controls)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)
        self.update_controls()

    @staticmethod
    def add_tab(tabs, name):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        scroll.setWidget(content)
        tabs.addTab(scroll, name)
        return form

    def add_fields(self, form, definitions, sliders=False):
        for key, label, low, high, default, step in definitions:
            spin = QDoubleSpinBox()
            integer = all(
                isinstance(value, int) for value in (low, high, default, step)
            )
            spin.setDecimals(0 if integer else 3)
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setKeyboardTracking(False)
            self.inputs[key] = spin

            if sliders:
                container = QWidget()
                row = QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)
                slider = QSlider(Qt.Orientation.Horizontal)
                factor = 1 if integer else 1000
                slider.setRange(round(low * factor), round(high * factor))
                slider.setSingleStep(max(1, round(step * factor)))
                slider.setValue(round(default * factor))
                slider.valueChanged.connect(
                    lambda value, target=spin, f=factor:
                    target.setValue(value / f)
                )
                spin.valueChanged.connect(
                    lambda value, target=slider, f=factor:
                    target.setValue(round(value * f))
                )
                row.addWidget(slider, 1)
                row.addWidget(spin)
                form.addRow(label, container)
            else:
                form.addRow(label, spin)

            spin.valueChanged.connect(self.changed)

    def add_check(self, form, key, label):
        checkbox = QCheckBox(label)
        checkbox.setChecked(DEFAULTS[key])
        checkbox.toggled.connect(self.changed)
        self.checks[key] = checkbox
        form.addRow(checkbox)

    def parameters(self):
        p = defaults_for(self.role)
        p.update({key: widget.value() for key, widget in self.inputs.items()})
        p.update({key: widget.isChecked() for key, widget in self.checks.items()})
        return p

    def fill_parameters(self, p):
        self.loading = True
        try:
            for key, widget in self.inputs.items():
                widget.setValue(p[key])
            for key, widget in self.checks.items():
                widget.setChecked(p[key])
        finally:
            self.loading = False
        self.changed()

    def changed(self, *_):
        if self.loading:
            return
        if self.worker is not None:
            self.worker.configure(self.parameters())
        if self.frozen is not None:
            self.render()

    def source_value(self):
        selected = self.source.currentData()
        if selected == "file":
            return str(Path(self.file_path.text()).expanduser().resolve())
        return int(selected)

    def source_signature(self):
        return (
            self.source_value(),
            self.backend.currentData(),
            self.resolution.currentText(),
            self.fps.value(),
        )

    def update_controls(self, *_):
        busy = self.worker is not None
        file_source = self.source.currentData() == "file"
        for widget in (self.source, self.resolution, self.fps):
            widget.setEnabled(not busy)
        self.backend.setEnabled(not busy and not file_source)
        self.file_path.setEnabled(not busy and file_source)
        self.browse_button.setEnabled(not busy and file_source)
        self.start_button.setEnabled(not busy and self.frozen is None)
        self.stop_button.setEnabled(busy)
        self.pause_button.setEnabled(
            busy and file_source and self.frozen is None
        )
        self.record_button.setEnabled(busy and self.frozen is None)
        self.hardware_button.setEnabled(busy and not file_source)

        if self.recording:
            self.record_button.setStyleSheet(
                "QPushButton { background-color: #c0392b; color: white; "
                "font-weight: bold; }"
            )
            self.record_button.setText("Berhenti rekam")
        else:
            self.record_button.setStyleSheet("")
            self.record_button.setText("Rekam")

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih video", self.file_path.text(),
            "Video (*.webm *.mp4 *.avi *.mkv *.mov);;Semua file (*)",
        )
        if path:
            self.file_path.setText(path)

    def start(self):
        if self.worker is not None:
            return
        try:
            p = self.parameters()
            validate_parameters(p, self.top)
            source = self.source_value()
            if isinstance(source, str) and not Path(source).is_file():
                raise ValueError(f"File tidak ditemukan: {source}")

            for other in self.owner.panes:
                if (
                    other is not self
                    and other.worker is not None
                    and isinstance(source, int)
                    and other.active_source == source
                ):
                    raise ValueError(
                        "Indeks kamera sedang digunakan panel lain. "
                        "Pilih kamera berbeda."
                    )

            signature = self.source_signature()
            if signature != self.previous_source:
                if self.role != "side" or not SIDE_REFERENCE.is_file():
                    self.reference = None
                self.previous_source = signature

            size = tuple(map(int, self.resolution.currentText().split("x")))
            self.packet = None
            self.shown = None
            self.view.image = None
            self.view.update()
            self._set_thread_readout("Ujung benang: —")
            self.pause_button.setChecked(False)
            self.active_source = source
            self.worker = CaptureWorker(
                source, self.backend.currentData(), size,
                self.fps.value(), p, self.reference, self.role,
            )
            self.worker.start()
            self.message.setText("Membuka sumber...")
            self.update_controls()

        except Exception as exc:
            QMessageBox.warning(self, "Tidak dapat memulai", str(exc))

    def stop(self):
        if self.recording:
            self.stop_recording()
            self.record_button.setChecked(False)
        if self.frozen is not None:
            self.cancel_calibration()
        if self.worker is not None:
            self.worker.stop_event.set()
            self.stopping_since = time.monotonic()
            self.message.setText("Menghentikan sumber...")

    def pause(self, checked):
        if self.worker is not None:
            if checked:
                self.worker.paused.set()
            else:
                self.worker.paused.clear()

    def _release_worker(self):
        if self.recording:
            self.stop_recording()
            self.record_button.setChecked(False)
        self.worker = None
        self.active_source = None
        self.stopping_since = None
        self.pause_button.setChecked(False)
        self._set_thread_readout("Ujung benang: —")
        self.message.setText("Sumber dihentikan. Klik Mulai untuk membuka lagi.")
        self.update_controls()

    def poll(self):
        worker = self.worker
        if worker is None:
            return

        while True:
            try:
                kind, payload = worker.events.get_nowait()
            except queue.Empty:
                break
            if kind == "reference":
                self.reference = payload
            elif not worker.stop_event.is_set():
                self.message.setText(payload)

        try:
            self.packet = worker.latest.get_nowait()
        except queue.Empty:
            pass
        else:
            if not worker.stop_event.is_set():
                self.render()

        hung = (
            worker.stop_event.is_set()
            and self.stopping_since is not None
            and (time.monotonic() - self.stopping_since) > 2.0
        )
        if not worker.is_alive() or hung:
            self._release_worker()

    def render(self, *_):
        if self.frozen is not None:
            try:
                p = self.parameters()
                adjusted, _, _ = prepare_image(self.frozen, p)
                self.shown = draw_thread_guide(draw_guides(adjusted, p), p)
                self.view.set_frame(self.shown)
            except (ValueError, cv2.error) as exc:
                self.message.setText(str(exc))
            return

        if self.packet is None:
            return

        packet = self.packet
        mode = self.view_mode.currentIndex()

        if mode == 1:
            out = packet["raw"]
        elif mode == 3:
            out = cv2.cvtColor(packet["mask"], cv2.COLOR_GRAY2BGR)
        elif mode == 4:
            out = cv2.cvtColor(packet["gray"], cv2.COLOR_GRAY2BGR)
        elif mode == 5:
            thread_mask = packet.get("thread_mask")
            if thread_mask is None:
                thread_mask = np.zeros(packet["raw"].shape[:2], np.uint8)
            out = cv2.cvtColor(thread_mask, cv2.COLOR_GRAY2BGR)
        else:
            out = packet["adjusted"]

        enabled = bool(self.parameters().get("thread_on", True))
        thread = packet.get("thread") if enabled else None
        if mode == 0:
            if self.parameters()["guides"]:
                out = draw_guides(out, self.parameters())
            out = draw_detection(out, packet["result"])
            out = draw_thread(out, thread, enabled=enabled)
            if self.parameters().get("path_overlay", True):
                out = draw_insertion_overlay(
                    out,
                    packet["result"],
                    thread,
                    from_right=bool(self.parameters().get("thread_from_right", True)),
                )
        aligned = None
        if enabled and self.parameters().get("path_overlay", True):
            aligned = insertion_paths(
                packet["result"],
                thread,
                from_right=bool(self.parameters().get("thread_from_right", True)),
            )["aligned"]
        self._show_thread_readout(thread, enabled, aligned)

        self.shown = out
        self.view.set_frame(out)

        if self.recording and self.recorder is not None:
            try:
                self.recorder.write(out)
            except Exception:
                pass

        height, width = packet["raw"].shape[:2]
        text = (
            f"{width}x{height} | frame {packet['index']} | "
            f"{packet['ms']:.1f} ms | {packet['note']}"
        )
        result = packet["result"]
        if result:
            cx, cy = result["center"]
            text += f" | X={cx:.2f}, Y={cy:.2f} px | match={result['score']:.3f}"
            if result["ecc"] is not None:
                text += f" | ECC={result['ecc']:.3f}"
        if enabled:
            if thread:
                tip_x, tip_y = thread["tip"]
                text += f" | ujung {tip_x:.1f},{tip_y:.1f}"
            else:
                text += " | ujung tidak terdeteksi"
        self.stats.setText(text)

    def _set_thread_readout(self, body):
        self.thread_readout.setText(f'<span style="color:#7eecf0">{body}</span>')

    def _show_thread_readout(self, thread, enabled, aligned=None):
        if not enabled:
            self._set_thread_readout("Ujung benang: deteksi mati")
            return
        if thread is None:
            self._set_thread_readout("Ujung benang: tidak terdeteksi")
            return
        tip_x, tip_y = thread["tip"]
        body = thread.get("body_thickness")
        width_txt = f" | lebar {body:.1f} px" if body else ""
        if thread.get("oval_locked"):
            along, across = thread["tip_ellipse"][1]
            oval_txt = f" | oval kunci {along:.1f}×{across:.1f}"
        else:
            oval_txt = f" | tebal {thread['local_thickness']:.1f} px"
        if aligned is True:
            align_txt = " | jalur selaras"
        elif aligned is False:
            align_txt = " | benang melewati jarum"
        else:
            align_txt = ""
        self._set_thread_readout(
            f"Ujung benang: {tip_x:.1f}, {tip_y:.1f} px{width_txt}{oval_txt}{align_txt}"
        )

    def calibrate(self):
        if self.frozen is None:
            if self.packet is None:
                QMessageBox.information(
                    self, "Kalibrasi", "Mulai sumber sampai gambar muncul."
                )
                return
            self.old_parameters = self.parameters()
            self.frozen = self.packet["raw"].copy()
            self.was_paused = self.pause_button.isChecked()

            if isinstance(self.active_source, str):
                self.pause_button.setChecked(True)

            self.view.selecting = True
            self.calibration_button.setText("Terapkan referensi")
            self.cancel_button.setEnabled(True)
            self._place_thread_oval_preview()
            self.message.setText(
                "Drag oval lubang, template, search, atau oval ujung benang. "
                "Oval cyan dikunci ukurannya; saat jalan ia menempel di ujung."
            )
            self.update_controls()
            self.render()
            return

        try:
            p = self.parameters()
            validate_parameters(p, self.top)

            trial = tracker_for(self.role, self.frozen)
            trial.prepare_reference(self.frozen.shape[:2], p)

            self.reference = self.frozen.copy()
            if self.worker is not None:
                self.worker.configure(p, self.reference)

            self.finish_calibration()
            self.message.setText("Referensi baru diterapkan.")

        except Exception as exc:
            QMessageBox.warning(self, "Kalibrasi belum valid", str(exc))

    def finish_calibration(self):
        self.frozen = None
        self.view.selecting = False
        self.view.drag_start = None
        self.view.drag_end = None
        self.calibration_button.setText("Bekukan untuk kalibrasi")
        self.cancel_button.setEnabled(False)
        self.pause_button.setChecked(self.was_paused)
        self.update_controls()
        self.render()

    def cancel_calibration(self):
        if self.frozen is not None:
            self.fill_parameters(self.old_parameters)
            self.finish_calibration()
            self.message.setText("Kalibrasi dibatalkan.")

    def select_region(self, x0, y0, x1, y1):
        if self.frozen is None:
            return

        height, width = self.frozen.shape[:2]
        p = self.parameters()
        mode = self.selection_mode.currentText()

        if mode == "Area pencarian":
            p.update(
                rx0=x0 / width, ry0=y0 / height,
                rx1=x1 / width, ry1=y1 / height,
            )
        elif mode == "Oval ujung benang":
            x0, x1 = x0 * 640 / width, x1 * 640 / width
            y0, y1 = y0 * 480 / height, y1 * 480 / height
            p.update(
                thread_oval_cx=(x0 + x1) / 2,
                thread_oval_cy=(y0 + y1) / 2,
                thread_oval_along=max(1.0, x1 - x0),
                thread_oval_across=max(1.0, y1 - y0),
                thread_oval_angle=0.0,
                thread_oval_lock=True,
            )
        else:
            x0, x1 = x0 * 640 / width, x1 * 640 / width
            y0, y1 = y0 * 480 / height, y1 * 480 / height
            if mode == "Template":
                p.update(tx=x0, ty=y0, tw=x1 - x0, th=y1 - y0)
            else:
                p.update(
                    cx=(x0 + x1) / 2,
                    cy=(y0 + y1) / 2,
                    axis_a=x1 - x0,
                    axis_b=y1 - y0,
                    angle=0.0,
                )

        self.fill_parameters(p)

    def _place_thread_oval_preview(self):
        """Taruh pratinjau oval cyan di ujung yang terakhir terukur."""
        packet = self.packet
        thread = None if packet is None else packet.get("thread")
        if thread is None or self.frozen is None:
            return
        height, width = self.frozen.shape[:2]
        p = self.parameters()
        tip_x, tip_y = thread["tip"]
        p["thread_oval_cx"] = float(tip_x) * 640.0 / width
        p["thread_oval_cy"] = float(tip_y) * 480.0 / height
        self.fill_parameters(p)

    def lock_thread_oval(self):
        """Kunci diameter oval ke lebar badan benang yang sedang terukur."""
        packet = self.packet
        thread = None if packet is None else packet.get("thread")
        if thread is None:
            QMessageBox.information(
                self, "Oval ujung",
                "Ujung benang belum terdeteksi. Mulai sumber sampai ujung tampak.",
            )
            return
        height, width = packet["raw"].shape[:2]
        body = float(thread.get("body_thickness") or thread["local_thickness"])
        across = body * 480.0 / height
        along = float(np.clip(across * 0.6, 6.0, 120.0))
        tip_x, tip_y = thread["tip"]
        p = self.parameters()
        p.update(
            thread_oval_lock=True,
            thread_oval_across=across,
            thread_oval_along=along,
            thread_oval_angle=float(thread["angle"]),
            thread_oval_cx=float(tip_x) * 640.0 / width,
            thread_oval_cy=float(tip_y) * 480.0 / height,
        )
        self.fill_parameters(p)
        self.message.setText(
            f"Oval dikunci ke lebar benang {across:.1f} px (referensi 480)."
        )

    def apply_hardware(self):
        if self.worker is not None:
            self.worker.commands.put((
                self.auto_exposure.currentData(),
                self.exposure.value(),
                self.gain.value(),
            ))

    def snapshot(self):
        if self.shown is None:
            return
        prefix = self.role
        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan snapshot",
            f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png)",
        )
        if path:
            try:
                if not path.lower().endswith(".png"):
                    path += ".png"
                if not cv2.imwrite(path, self.shown):
                    raise RuntimeError("Gagal menulis gambar.")
                self.message.setText(f"Snapshot tersimpan: {path}")
            except Exception as exc:
                QMessageBox.warning(self, "Snapshot gagal", str(exc))

    def toggle_recording(self, checked):
        if checked:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.recording:
            return
        if self.shown is None:
            self.record_button.setChecked(False)
            QMessageBox.information(
                self, "Rekam",
                "Mulai sumber sampai gambar muncul sebelum merekam.",
            )
            return

        default_name = f"rekam_{self.role}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan rekaman video", default_name,
            "Video MP4 (*.mp4);;Video AVI (*.avi)",
        )
        if not path:
            self.record_button.setChecked(False)
            return

        if not (path.lower().endswith(".mp4") or path.lower().endswith(".avi")):
            path += ".mp4"

        try:
            height, width = self.shown.shape[:2]
            fps = self.fps.value()
            if path.lower().endswith(".avi"):
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.recorder = cv2.VideoWriter(path, fourcc, fps, (width, height))
            if not self.recorder.isOpened():
                raise RuntimeError("Gagal membuka VideoWriter.")

            self.recording = True
            self.record_path = path
            self.message.setText(f"Merekam ke: {path}")
            self.update_controls()

        except Exception as exc:
            self.recorder = None
            self.recording = False
            self.record_button.setChecked(False)
            QMessageBox.warning(self, "Rekam gagal", str(exc))

    def stop_recording(self):
        if not self.recording:
            return
        try:
            if self.recorder is not None:
                self.recorder.release()
        except Exception:
            pass
        self.recorder = None
        self.recording = False
        self.message.setText(f"Rekaman tersimpan: {self.record_path}")
        self.record_path = None
        self.update_controls()

    def profile(self):
        p = self.parameters()
        validate_parameters(p, self.top)
        reference = None
        if self.reference is not None:
            ok, encoded = cv2.imencode(".png", self.reference)
            if not ok:
                raise RuntimeError("Gagal mengodekan gambar referensi.")
            reference = base64.b64encode(encoded.tobytes()).decode("ascii")

        return {
            "role": self.role,
            "source": self.source.currentData(),
            "file": self.file_path.text(),
            "backend": self.backend.currentData(),
            "resolution": self.resolution.currentText(),
            "fps": self.fps.value(),
            "parameters": p,
            "reference_png": reference,
        }

    def restore_profile(self, data, decoded_reference):
        self.source.setCurrentIndex(self.source.findData(data["source"]))
        self.file_path.setText(data["file"])
        self.backend.setCurrentIndex(self.backend.findData(data["backend"]))
        self.resolution.setCurrentText(data["resolution"])
        self.fps.setValue(data["fps"])
        self.fill_parameters(data["parameters"])
        self.reference = decoded_reference
        self.previous_source = self.source_signature()
        self.packet = None
        self.shown = None
        self.view.image = None
        self.view.update()
        self.message.setText("Profil dimuat. Klik Mulai.")
        self.update_controls()


def atomic_json_write(path, data):
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=str(destination.parent),
            prefix="pinhole_", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2, allow_nan=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class MainWindow(QMainWindow):
    def __init__(self, top_file, side_file):
        super().__init__()
        self.setWindowTitle("Pinhole + Ujung Benang | SIDE + TOP")
        self.setMinimumSize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        title = QLabel("PINHOLE + UJUNG BENANG — SIDE / TOP")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        toolbar.addWidget(title)
        toolbar.addStretch()

        save_button = QPushButton("Simpan profil")
        load_button = QPushButton("Muat profil")
        stop_button = QPushButton("Stop semua")
        save_button.clicked.connect(self.save_profile)
        load_button.clicked.connect(self.load_profile)
        stop_button.clicked.connect(self.stop_all)

        for button in (save_button, load_button, stop_button):
            toolbar.addWidget(button)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.panes = [
            CameraPane("SIDE — samping", "side", self, side_file),
            CameraPane("TOP — pelacak oval", "top", self, top_file),
        ]
        for pane in self.panes:
            splitter.addWidget(pane)
        splitter.setSizes([700, 700])
        layout.addWidget(splitter)

        self.statusBar().showMessage(
            "Kiri: samping. Kanan: atas. Oval lubang dan ujung benang coklat, "
            "dalam piksel, bukan metrologi."
        )

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, screen.width() - 40), min(980, screen.height() - 40))
        frame = self.frameGeometry()
        frame.moveCenter(screen.center())
        self.move(frame.topLeft())

    def pane(self, role):
        for item in self.panes:
            if item.role == role:
                return item
        raise KeyError(role)

    def stop_all(self):
        for pane in self.panes:
            pane.stop()

    def save_profile(self):
        if any(pane.frozen is not None for pane in self.panes):
            QMessageBox.information(
                self, "Profil",
                "Terapkan atau batalkan kalibrasi sebelum menyimpan profil.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Simpan profil", "pinhole_profile.json", "JSON (*.json)"
        )
        if not path:
            return

        try:
            if not path.lower().endswith(".json"):
                path += ".json"
            data = {
                "version": 3,
                "panes": [pane.profile() for pane in self.panes],
            }
            atomic_json_write(path, data)
            self.statusBar().showMessage(f"Profil tersimpan: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Simpan profil gagal", str(exc))

    def load_profile(self):
        if any(
            pane.worker is not None or pane.frozen is not None
            for pane in self.panes
        ):
            QMessageBox.information(
                self, "Profil",
                "Stop kedua sumber dan selesaikan kalibrasi sebelum memuat profil.",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Muat profil", "", "JSON (*.json)"
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            panes_data = data.get("panes", [])
            if data.get("version") not in (1, 2, 3) or len(panes_data) != 2:
                raise ValueError("Format profil tidak didukung.")

            prepared = []
            for index, pane_data in enumerate(panes_data):
                role = pane_data.get("role")
                if role not in ("top", "side"):
                    # Profil v1: indeks 0 = TOP kiri. Profil baru: 0 = SIDE kiri.
                    if data.get("version") == 1:
                        role = "top" if index == 0 else "side"
                    else:
                        role = "side" if index == 0 else "top"
                pane_data["role"] = role

                p = defaults_for(role)
                p.update({
                    key: value
                    for key, value in pane_data["parameters"].items()
                    if key in p
                })
                validate_parameters(p, top=(role == "top"))
                pane_data["parameters"] = p

                source = pane_data["source"]
                if source != "file" and (
                    type(source) is not int or source not in range(10)
                ):
                    raise ValueError("Indeks sumber tidak valid.")

                if not isinstance(pane_data["file"], str):
                    raise ValueError("Path video tidak valid.")

                if pane_data["backend"] not in [value for _, value in BACKENDS]:
                    raise ValueError("Backend tidak didukung.")

                if pane_data["resolution"] not in RESOLUTIONS:
                    raise ValueError("Resolusi tidak didukung.")

                fps = float(pane_data["fps"])
                if not math.isfinite(fps) or not 1 <= fps <= 120:
                    raise ValueError("FPS tidak valid.")

                reference = None
                encoded = pane_data.get("reference_png")
                if encoded:
                    raw = base64.b64decode(encoded, validate=True)
                    reference = cv2.imdecode(
                        np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
                    )
                    if reference is None:
                        raise ValueError("Gambar referensi rusak.")

                prepared.append((pane_data, reference))

            by_role = {item[0]["role"]: item for item in prepared}
            if set(by_role) != {"top", "side"}:
                raise ValueError("Profil harus berisi panel top dan side.")

            for pane in self.panes:
                pane_data, reference = by_role[pane.role]
                pane.restore_profile(pane_data, reference)

            self.statusBar().showMessage(f"Profil dimuat: {path}")

        except Exception as exc:
            QMessageBox.warning(self, "Muat profil gagal", str(exc))

    def closeEvent(self, event):
        workers = [
            pane.worker for pane in self.panes if pane.worker is not None
        ]
        for worker in workers:
            worker.stop_event.set()
        for worker in workers:
            worker.join(timeout=0.5)
        event.accept()


STYLE = """
QWidget {
    background: #142132;
    color: #e6edf5;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #364960;
    border-radius: 7px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
}
QPushButton {
    background: #263e56;
    border: 1px solid #405b75;
    border-radius: 5px;
    padding: 7px 10px;
}
QPushButton:hover { background: #31536f; }
QPushButton:checked { background: #146b70; }
QPushButton:disabled { color: #8ea0b5; background: #1a293b; }
QLineEdit, QComboBox, QDoubleSpinBox {
    background: #0c1725;
    border: 1px solid #40536a;
    border-radius: 4px;
    padding: 4px;
}
QTabBar::tab {
    background: #203449;
    padding: 7px 10px;
}
QTabBar::tab:selected {
    background: #17636c;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #354b61;
}
QSlider::handle:horizontal {
    background: #50d0cb;
    width: 12px;
    margin: -4px 0;
    border-radius: 5px;
}
"""
