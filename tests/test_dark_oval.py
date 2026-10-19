"""Lubang jarum samping: oval gelap, bukan logam terang."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pinhole.params import defaults_for
from pinhole.vision import SidePinholeTracker


def _gray_frame():
    return np.full((480, 640, 3), 160, np.uint8)


def _stamp_bright_patch(image, x, y, width=48, height=64):
    yy, xx = np.mgrid[0:height, 0:width]
    patch = (170 + (xx * 3) % 50).astype(np.uint8)
    patch[8:16, :] = 245
    image[y:y + height, x:x + width] = patch[:, :, None]


class DarkOvalTest(unittest.TestCase):
    def test_side_tracker_prefers_black_oval_over_bright_template(self):
        reference = _gray_frame()
        _stamp_bright_patch(reference, 168, 138)

        frame = _gray_frame()
        _stamp_bright_patch(frame, 40, 90)
        cv2.ellipse(frame, (250, 200), (8, 20), 0, 0, 360, (5, 5, 5), -1)

        tracker = SidePinholeTracker(reference)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, _note = tracker.detect(gray, defaults_for("side"))

        self.assertIsNotNone(result)
        cx, cy = result["center"]
        self.assertLess(abs(cx - 250), 12)
        self.assertLess(abs(cy - 200), 12)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.ellipse(mask, result["ellipse"], 255, -1)
        self.assertLess(float(gray[mask > 0].mean()), 40)

    def test_side_tracker_reports_nothing_without_a_dark_hole(self):
        reference = _gray_frame()
        _stamp_bright_patch(reference, 168, 138)
        frame = _gray_frame()
        _stamp_bright_patch(frame, 40, 90)
        tracker = SidePinholeTracker(reference)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, _note = tracker.detect(gray, defaults_for("side"))
        self.assertIsNone(result)

    def test_side_tracker_ignores_dark_blob_on_search_border(self):
        reference = _gray_frame()
        _stamp_bright_patch(reference, 168, 138)
        frame = _gray_frame()
        frame[76:100, 10:70] = 0
        tracker = SidePinholeTracker(reference)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, _note = tracker.detect(gray, defaults_for("side"))
        self.assertIsNone(result)
