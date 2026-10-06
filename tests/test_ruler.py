"""Penggaris dua titik: jarak dalam milimeter."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from pinhole.params import (
    SIDE_PX_PER_MM_Z,
    TOP_PX_PER_MM_X,
    TOP_PX_PER_MM_Y,
    ruler_measurement,
)
from pinhole.ui import VideoView


def _packet(frame):
    return {
        "raw": frame,
        "adjusted": frame,
        "mask": np.zeros(frame.shape[:2], np.uint8),
        "gray": cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        "result": None,
        "thread": None,
        "thread_mask": np.zeros(frame.shape[:2], np.uint8),
        "note": "uji",
        "index": 1,
        "ms": 1.0,
    }


class RulerMeasurementTest(unittest.TestCase):
    def test_top_straight_line_is_euclidean_millimetres(self):
        end_x = 3 * TOP_PX_PER_MM_X
        end_y = 4 * TOP_PX_PER_MM_Y
        measured = ruler_measurement(
            0, 0, end_x, end_y,
            TOP_PX_PER_MM_X, TOP_PX_PER_MM_Y, 640, 480, 0.0,
        )
        self.assertEqual(measured["kind"], "euclidean")
        self.assertEqual(measured["unit"], "mm")
        self.assertAlmostEqual(measured["distance"], 5.0, places=6)
        self.assertIn("awal X 0.00 Y 0.00 mm", measured["text"])
        self.assertIn("akhir X 3.00 Y 4.00 mm", measured["text"])
        self.assertIn("jarak 5.00 mm", measured["text"])
        self.assertEqual(measured["overlay"], "5.00 mm")
        self.assertNotIn(",", measured["text"])

    def test_larger_frame_uses_the_640x480_scale(self):
        measured = ruler_measurement(
            0, 0, 2 * TOP_PX_PER_MM_X, 0,
            TOP_PX_PER_MM_X, TOP_PX_PER_MM_Y, 1280, 960, 0.0,
        )
        self.assertAlmostEqual(measured["distance"], 1.0, places=6)
        self.assertIn("jarak 1.00 mm", measured["text"])

    def test_side_reports_vertical_z_only(self):
        measured = ruler_measurement(
            40, 10, 180, 10 + SIDE_PX_PER_MM_Z,
            0.0, 0.0, 640, 480, SIDE_PX_PER_MM_Z,
        )
        self.assertEqual(measured["kind"], "vertical")
        self.assertAlmostEqual(measured["distance"], 1.0, places=6)
        self.assertIn("awal Z 0.16 mm", measured["text"])
        self.assertIn("ΔZ 1.00 mm", measured["text"])
        self.assertIn("mendatar belum berskala", measured["text"])
        self.assertNotIn(",", measured["text"])

    def test_side_with_x_scale_uses_straight_line(self):
        measured = ruler_measurement(
            0, 0, SIDE_PX_PER_MM_Z, SIDE_PX_PER_MM_Z,
            SIDE_PX_PER_MM_Z, 0.0, 640, 480, SIDE_PX_PER_MM_Z,
        )
        self.assertEqual(measured["kind"], "euclidean")
        self.assertAlmostEqual(measured["distance"], 2 ** 0.5, places=4)
        self.assertIn("X 0.00 Z 0.00 mm", measured["text"])
        self.assertIn("jarak 1.41 mm", measured["text"])

    def test_without_scale_stays_in_pixels(self):
        measured = ruler_measurement(10, 20, 40, 60, 0, 0, 640, 480, 0)
        self.assertEqual(measured["unit"], "px")
        self.assertAlmostEqual(measured["distance"], 50.0, places=6)
        self.assertIn("jarak 50.0 px", measured["text"])
        self.assertNotIn("mm", measured["text"])
        self.assertNotIn(",", measured["text"])


class RulerGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pinhole.ui import MainWindow

        cls.app = QApplication.instance() or QApplication([])
        cls.window = MainWindow("", "")

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def test_both_panes_have_a_ruler(self):
        for pane in self.window.panes:
            self.assertEqual(pane.ruler_button.text(), "Penggaris")
            self.assertIn("titik awal", pane.ruler_readout.text())

    def test_top_points_draw_millimetre_distance(self):
        pane = self.window.panes[1]
        frame = np.full((480, 640, 3), 24, np.uint8)
        pane.packet = _packet(frame)
        pane.view.ruler_mode = True
        pane.view.ruler_start = (0.0, TOP_PX_PER_MM_Y)
        pane.view.ruler_end = (TOP_PX_PER_MM_X, TOP_PX_PER_MM_Y)
        pane.on_ruler_placed()

        text = pane.ruler_readout.text()
        self.assertIn("jarak 1.00 mm", text)
        self.assertNotIn(",", text)
        magenta = (
            (pane.shown[:, :, 0] > 180)
            & (pane.shown[:, :, 1] < 40)
            & (pane.shown[:, :, 2] > 180)
        )
        self.assertGreater(int(magenta.sum()), 20)

        pane.clear_ruler()
        self.assertIsNone(pane.view.ruler_start)
        self.assertIn("klik titik awal", pane.ruler_readout.text())
        cleared = (
            (pane.shown[:, :, 0] > 180)
            & (pane.shown[:, :, 1] < 40)
            & (pane.shown[:, :, 2] > 180)
        )
        self.assertEqual(int(cleared.sum()), 0)

    def test_side_vertical_points_report_delta_z(self):
        pane = self.window.panes[0]
        frame = np.full((480, 640, 3), 24, np.uint8)
        pane.packet = _packet(frame)
        pane.view.ruler_start = (200.0, 0.0)
        pane.view.ruler_end = (200.0, SIDE_PX_PER_MM_Z)
        pane.on_ruler_placed()
        text = pane.ruler_readout.text()
        self.assertIn("ΔZ 1.00 mm", text)
        self.assertNotIn("mendatar belum berskala", text)
        self.assertNotIn(",", text)
        pane.clear_ruler()

    def test_click_sets_start_then_end(self):
        frame = np.full((480, 640, 3), 20, np.uint8)
        view = VideoView()
        view.ruler_mode = True
        view.resize(640, 480)
        view.set_frame(frame)
        view.show()
        self.app.processEvents()
        view.repaint()
        self.assertGreater(view.image_rect.width(), 0)

        placed = []
        view.rulerPlaced.connect(lambda: placed.append(1))
        QTest.mouseClick(
            view, Qt.MouseButton.LeftButton, pos=QPoint(30, 40),
        )
        self.assertEqual(len(placed), 1)
        self.assertIsNotNone(view.ruler_start)
        self.assertIsNone(view.ruler_end)
        self.assertAlmostEqual(view.ruler_start[0], 30, delta=1.5)
        self.assertAlmostEqual(view.ruler_start[1], 40, delta=1.5)

        QTest.mouseClick(
            view, Qt.MouseButton.LeftButton, pos=QPoint(90, 40),
        )
        self.assertEqual(len(placed), 2)
        self.assertAlmostEqual(view.ruler_end[0], 90, delta=1.5)
        self.assertAlmostEqual(view.ruler_end[1], 40, delta=1.5)
        view.close()
