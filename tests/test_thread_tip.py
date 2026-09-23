"""Ujung benang pada kamera TOP dan SIDE, plus overlay GUI."""

from __future__ import annotations

import os
import queue
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

from pinhole.capture import CaptureWorker
from pinhole.params import defaults_for, validate_parameters
from pinhole.thread_tip import BrownThreadTipTracker


def copper_bgr():
    pixel = np.uint8([[[10, 200, 170]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in bgr)


def make_thread_frame(
    tip_y=232,
    x_start=280,
    body_x=430,
    width=640,
    height=480,
    thickness=25,
):
    image = np.full((height, width, 3), 246, np.uint8)
    color = copper_bgr()
    span = max(1, body_x - x_start)
    for x in range(x_start, width):
        if x >= body_x:
            thick = thickness
        else:
            thick = int(round(thickness * (x - x_start) / span))
        if thick <= 0:
            continue
        y0 = int(tip_y - thick // 2)
        y1 = min(height, y0 + thick)
        y0 = max(0, y0)
        image[y0:y1, x] = color
    return image


def raw_tip(frame, p):
    result, _mask = BrownThreadTipTracker().detect(frame, p)
    if result is None:
        raise AssertionError("ujung tidak terukur")
    return np.array(result["tip"], np.float64)


class ThreadTipTest(unittest.TestCase):
    def setUp(self):
        self.params = defaults_for("top")

    def test_defaults_validate_for_both_cameras(self):
        for role in ("top", "side"):
            p = defaults_for(role)
            validate_parameters(p, top=(role == "top"))
            self.assertTrue(p["thread_on"])
            self.assertTrue(p["thread_from_right"])
            self.assertEqual(p["th_hmin"], 0)
            self.assertEqual(p["th_hmax"], 22)
            self.assertEqual(p["th_smin"], 60)
            self.assertAlmostEqual(p["thread_tip_ratio"], 0.30)
            self.assertAlmostEqual(p["thread_still_radius"], 3.2)
            self.assertAlmostEqual(p["thread_quiet_radius"], 1.6)

        side = defaults_for("side")
        top = defaults_for("top")
        self.assertNotEqual(side["cx"], top["cx"])
        self.assertEqual(side["th_vmax"], top["th_vmax"])

    def test_legacy_profile_keeps_thread_defaults(self):
        legacy = {
            key: value
            for key, value in defaults_for("side").items()
            if not key.startswith("th_") and not key.startswith("thread_")
        }
        merged = defaults_for("side")
        merged.update({key: value for key, value in legacy.items() if key in merged})
        validate_parameters(merged, top=False)
        self.assertTrue(merged["thread_on"])
        self.assertTrue(merged["thread_from_right"])
        self.assertTrue(merged["thread_oval_lock"])
        self.assertEqual(merged["th_hmax"], 22)
        self.assertAlmostEqual(merged["thread_still_radius"], 3.2)
        self.assertAlmostEqual(merged["thread_oval_across"], 18.0)

    def test_rejects_inverted_smooth_radius(self):
        p = defaults_for("side")
        p["thread_quiet_radius"] = 8
        p["thread_still_radius"] = 2
        with self.assertRaises(ValueError):
            validate_parameters(p)

    def test_tip_is_free_end_not_image_border(self):
        frame = make_thread_frame()
        result, mask = BrownThreadTipTracker().detect(frame, self.params)
        self.assertIsNotNone(result)
        tip_x, tip_y = result["tip"]
        self.assertGreater(tip_x, 280)
        self.assertLess(tip_x, 430)
        self.assertAlmostEqual(tip_y, 232, delta=4)
        self.assertGreater(int(mask.sum()), 0)
        preview = BrownThreadTipTracker().preview_mask(frame, self.params)
        self.assertTrue(np.array_equal(mask, preview))

    def test_still_tip_is_smoothed_and_motion_follows(self):
        p = self.params
        settled = make_thread_frame(tip_y=232)
        nudged = make_thread_frame(tip_y=233)
        jumped = make_thread_frame(tip_y=270)

        raw_settled = raw_tip(settled, p)
        raw_nudged = raw_tip(nudged, p)
        raw_jumped = raw_tip(jumped, p)
        nudge = float(np.hypot(*(raw_nudged - raw_settled)))
        self.assertLessEqual(nudge, p["thread_quiet_radius"])
        self.assertGreater(nudge, 0.2)

        tracker = BrownThreadTipTracker()
        first, _ = tracker.detect(settled, p)
        second, _ = tracker.detect(nudged, p)
        followed = np.array(first["tip"]) + p["thread_quiet_alpha"] * (
            raw_nudged - raw_settled
        )
        np.testing.assert_allclose(second["tip"], followed, atol=0.05)

        third, _ = tracker.detect(jumped, p)
        np.testing.assert_allclose(third["tip"], raw_jumped, atol=0.05)

        again, _ = tracker.detect(jumped, p)
        np.testing.assert_allclose(again["tip"], third["tip"], atol=1e-6)

    def test_long_miss_drops_old_smooth(self):
        p = self.params
        first_frame = make_thread_frame(tip_y=200)
        second_frame = make_thread_frame(tip_y=260)
        blank = np.full_like(first_frame, 246)
        tracker = BrownThreadTipTracker()
        tracker.detect(first_frame, p)
        for _ in range(7):
            missed, _mask = tracker.detect(blank, p)
            self.assertIsNone(missed)
        found, _ = tracker.detect(second_frame, p)
        np.testing.assert_allclose(found["tip"], raw_tip(second_frame, p), atol=0.05)

    def test_left_entry_tracks_free_end(self):
        right = make_thread_frame()
        left = cv2.flip(right, 1)
        right_tip = raw_tip(right, self.params)
        p = dict(self.params)
        p["thread_from_right"] = False
        result, mask = BrownThreadTipTracker().detect(left, p)
        self.assertIsNotNone(result)
        width = left.shape[1]
        self.assertAlmostEqual(result["tip"][0], (width - 1) - right_tip[0], places=2)
        self.assertAlmostEqual(result["tip"][1], right_tip[1], places=2)
        preview = BrownThreadTipTracker().preview_mask(left, p)
        self.assertTrue(np.array_equal(mask, preview))

    def test_locked_oval_keeps_calibrated_width(self):
        from pinhole.thread_tip import locked_axes

        p = dict(self.params)
        p["thread_oval_lock"] = True
        p["thread_oval_along"] = 12.0
        p["thread_oval_across"] = 21.0
        p["thread_oval_angle"] = -3.0
        result, _mask = BrownThreadTipTracker().detect(make_thread_frame(), p)
        (cx, cy), axes, angle = result["tip_ellipse"]
        self.assertTrue(result["oval_locked"])
        self.assertAlmostEqual(axes[0], 12.0)
        self.assertAlmostEqual(axes[1], 21.0)
        self.assertAlmostEqual(angle, -3.0)
        self.assertAlmostEqual(cx, result["tip"][0])
        self.assertAlmostEqual(cy, result["tip"][1])
        self.assertGreater(result["body_thickness"], 10.0)
        self.assertGreater(abs(axes[1] - result["body_thickness"]), 0.5)

        scaled = locked_axes(p, 1280, 960)
        self.assertAlmostEqual(scaled[0], 24.0)
        self.assertAlmostEqual(scaled[1], 42.0)
        self.assertAlmostEqual(scaled[2], -3.0)

    def test_unlocked_oval_follows_measured_width(self):
        p = dict(self.params)
        p["thread_oval_lock"] = False
        p["thread_oval_across"] = 80.0
        result, _mask = BrownThreadTipTracker().detect(make_thread_frame(), p)
        self.assertFalse(result["oval_locked"])
        self.assertLess(result["tip_ellipse"][1][1], 75.0)

    def test_blue_thread_is_ignored(self):
        frame = make_thread_frame()
        blue = np.uint8([[[120, 200, 170]]])
        color = cv2.cvtColor(blue, cv2.COLOR_HSV2BGR)[0, 0]
        copper = np.array(copper_bgr(), np.uint8)
        pixels = np.all(frame == copper, axis=2)
        frame[pixels] = color
        result, mask = BrownThreadTipTracker().detect(frame, self.params)
        self.assertIsNone(result)
        self.assertEqual(int(mask.sum()), 0)

    def test_column_profile_matches_per_column_median(self):
        rng = np.random.default_rng(3)
        binary = np.zeros((80, 90), np.uint8)
        binary[rng.random((80, 90)) > 0.82] = 255
        tracker = BrownThreadTipTracker()
        profile = tracker._column_profile(binary)
        self.assertIsNotNone(profile)
        x0, centerline, thickness, raw_thick = profile

        ys, xs = np.where(binary > 0)
        length = int(xs.max() - xs.min()) + 1
        med = np.full(length, np.nan)
        thick = np.zeros(length)
        for x in range(int(xs.min()), int(xs.max()) + 1):
            column = ys[xs == x]
            if len(column) == 0:
                continue
            med[x - x0] = float(np.median(column))
            thick[x - x0] = float(len(column))
        good = np.isfinite(med)
        columns = np.arange(length)
        filled = np.interp(columns, columns[good], med[good])
        expected = np.empty_like(filled)
        for index in range(length):
            expected[index] = np.median(filled[max(0, index - 4):min(length, index + 5)])
        expected_thickness = np.convolve(thick, np.ones(5) / 5.0, mode="same")
        expected_thickness[0] = thick[0]
        expected_thickness[1] = thick[1]
        expected_thickness[-1] = thick[-1]
        expected_thickness[-2] = thick[-2]
        np.testing.assert_allclose(centerline, expected, atol=1e-9)
        np.testing.assert_allclose(thickness, expected_thickness, atol=1e-9)
        np.testing.assert_array_equal(raw_thick, thick)


class ThreadGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from pinhole.ui import MainWindow

        cls.app = QApplication.instance() or QApplication([])
        cls.window = MainWindow("", "")

    @classmethod
    def tearDownClass(cls):
        cls.window.close()

    def test_both_panes_have_independent_thread_controls(self):
        from PySide6.QtWidgets import QTabWidget

        self.assertEqual(
            [pane.role for pane in self.window.panes],
            ["side", "top"],
        )
        side, top = self.window.panes
        for pane in (side, top):
            tabs = pane.findChildren(QTabWidget)
            names = [tabs[0].tabText(i) for i in range(tabs[0].count())]
            self.assertIn("Benang", names)
            self.assertGreaterEqual(pane.view_mode.findText("Mask benang"), 0)
            self.assertGreaterEqual(
                pane.selection_mode.findText("Oval ujung benang"), 0
            )
            self.assertTrue(pane.checks["thread_on"].isChecked())
            self.assertTrue(pane.checks["thread_from_right"].isChecked())
            self.assertTrue(pane.checks["thread_oval_lock"].isChecked())

        side.checks["thread_on"].setChecked(False)
        self.assertFalse(side.parameters()["thread_on"])
        self.assertTrue(top.parameters()["thread_on"])
        side.checks["thread_on"].setChecked(True)

        side_p = side.parameters()
        top_p = top.parameters()
        self.assertAlmostEqual(side_p["cx"], 190.0)
        self.assertAlmostEqual(top_p["cx"], 175.5)
        validate_parameters(side_p, top=False)
        validate_parameters(top_p, top=True)
        self.assertIn("thread_on", side.profile()["parameters"])
        self.assertIn("th_hmin", top.profile()["parameters"])

    def test_overlay_readout_and_mask_view(self):
        pane = self.window.panes[1]
        frame = make_thread_frame(tip_y=210)
        p = pane.parameters()
        thread, mask = BrownThreadTipTracker().detect(frame, p)
        self.assertIsNotNone(thread)
        pane.packet = {
            "raw": frame,
            "adjusted": frame,
            "mask": np.zeros(frame.shape[:2], np.uint8),
            "gray": cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            "result": None,
            "thread": thread,
            "thread_mask": mask,
            "note": "uji",
            "index": 4,
            "ms": 1.5,
        }
        pane.checks["thread_on"].setChecked(False)
        pane.view_mode.setCurrentIndex(0)
        pane.render()
        hidden = pane.shown.copy()
        self.assertIn("deteksi mati", pane.thread_readout.text())

        pane.checks["thread_on"].setChecked(True)
        pane.render()
        shown = pane.shown
        self.assertTrue(np.any(shown != hidden))
        x, y = (int(round(v)) for v in thread["tip"])
        np.testing.assert_array_equal(shown[y, x], (0, 0, 255))
        text = pane.thread_readout.text()
        self.assertIn(f"{thread['tip'][0]:.1f}", text)
        self.assertIn(f"{thread['tip'][1]:.1f}", text)
        self.assertIn("ujung", pane.stats.text())
        self.assertIn("oval kunci", pane.thread_readout.text())

        roi = shown[20:60, 8:320]
        # Teks cyan di atas latar terang: B dan G naik, R turun.
        cyan = (roi[:, :, 0] > roi[:, :, 2] + 12) & (roi[:, :, 1] > roi[:, :, 2] + 12)
        self.assertGreater(int(cyan.sum()), 8)

        pane.view_mode.setCurrentIndex(pane.view_mode.findText("Mask benang"))
        pane.render()
        self.assertGreater(int(pane.shown.max()), 200)
        self.assertIn(f"{thread['tip'][0]:.1f}", pane.thread_readout.text())
        pane.view_mode.setCurrentIndex(0)

        original = pane.parameters()
        pane.frozen = frame.copy()
        try:
            pane.selection_mode.setCurrentText("Oval ujung benang")
            pane.select_region(300, 200, 316, 230)
            dragged = pane.parameters()
            self.assertTrue(dragged["thread_oval_lock"])
            self.assertAlmostEqual(dragged["thread_oval_along"], 16.0)
            self.assertAlmostEqual(dragged["thread_oval_across"], 30.0)
            self.assertAlmostEqual(dragged["thread_oval_cx"], 308.0)
            self.assertAlmostEqual(dragged["thread_oval_cy"], 215.0)

            pane.lock_thread_oval()
            locked = pane.parameters()
            self.assertAlmostEqual(
                locked["thread_oval_across"], thread["body_thickness"], places=4
            )
            self.assertAlmostEqual(
                locked["thread_oval_angle"], thread["angle"], delta=0.001
            )
            self.assertTrue(locked["thread_oval_lock"])
            fitted, _mask = BrownThreadTipTracker().detect(frame, locked)
            pane.packet["thread"] = fitted
            pane.frozen = None
            pane.view_mode.setCurrentIndex(0)
            pane.render()
            shown = pane.shown.copy()
        finally:
            pane.frozen = None
            pane.fill_parameters(original)

        out_path = Path(tempfile.gettempdir()) / "thread_overlay_top.png"
        self.assertTrue(cv2.imwrite(str(out_path), shown))


class ThreadCaptureTest(unittest.TestCase):
    def test_worker_publishes_thread_for_top_and_side(self):
        frame = make_thread_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "benang.avi")
            writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480)
            )
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(frame)
            writer.release()

            for role in ("top", "side"):
                packet = self._read_packet(path, defaults_for(role), role, True)
                self.assertIsNotNone(packet["thread"], role)
                self.assertGreater(int(packet["thread_mask"].sum()), 0)
                tip_x = packet["thread"]["tip"][0]
                self.assertGreater(tip_x, 280)
                self.assertLess(tip_x, 430)

            disabled = dict(defaults_for("top"))
            disabled["thread_on"] = False
            packet = self._read_packet(path, disabled, "top", False)
            self.assertIsNone(packet["thread"])
            self.assertGreater(int(packet["thread_mask"].sum()), 0)

    def _read_packet(self, path, params, role, need_tip):
        worker = CaptureWorker(path, cv2.CAP_ANY, (640, 480), 20, params, None, role)
        worker.start()
        packet = None
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                packet = worker.latest.get(timeout=0.3)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            if packet["index"] >= 1 and (packet["thread"] is not None or not need_tip):
                break
        worker.stop_event.set()
        worker.join(timeout=3)
        messages = []
        while True:
            try:
                messages.append(worker.events.get_nowait())
            except queue.Empty:
                break
        errors = [
            payload for kind, payload in messages
            if kind == "message" and str(payload).startswith("ERROR")
        ]
        self.assertEqual(errors, [], role)
        self.assertIsNotNone(packet, role)
        return packet


if __name__ == "__main__":
    unittest.main()
