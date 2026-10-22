"""Lubang jarum samping: template matching, sama seperti metode #8."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from pinhole.params import defaults_for
from pinhole.vision import SidePinholeTracker


def _reference():
    image = np.full((480, 640, 3), 160, np.uint8)
    yy, xx = np.mgrid[0:64, 0:48]
    patch = (40 + (xx * 4 + yy) % 180).astype(np.uint8)
    image[138:202, 168:216] = patch[:, :, None]
    return image


class SideTemplateTest(unittest.TestCase):
    def test_side_tracker_follows_the_shifted_template(self):
        reference = _reference()
        frame = np.full_like(reference, 160)
        frame[122:186, 192:240] = reference[138:202, 168:216]
        cv2.ellipse(frame, (80, 140), (8, 20), 0, 0, 360, (5, 5, 5), -1)

        tracker = SidePinholeTracker(reference)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, note = tracker.detect(gray, defaults_for("side"))

        self.assertIsNotNone(result)
        self.assertIn("TM", note)
        cx, cy = result["center"]
        self.assertLess(abs(cx - 214), 8)
        self.assertLess(abs(cy - 155), 8)
        self.assertGreater(result["score"], 0.9)

    def test_side_tracker_ignores_a_dark_blob_without_the_template(self):
        reference = _reference()
        frame = np.full_like(reference, 160)
        cv2.ellipse(frame, (250, 200), (8, 20), 0, 0, 360, (5, 5, 5), -1)

        tracker = SidePinholeTracker(reference)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result, note = tracker.detect(gray, defaults_for("side"))

        self.assertIsNone(result)
        self.assertNotIn("Oval gelap", note)
