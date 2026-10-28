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

    def test_high_template_score_without_the_hole_is_rejected(self):
        """Tepi di sekitar lubang masih mirip, tetapi jarumnya tidak ada."""
        params = defaults_for("side")
        params.update(
            cx=190,
            cy=175,
            axis_a=12,
            axis_b=16,
            angle=0,
            tx=150,
            ty=145,
            tw=90,
            th=70,
            rx0=0.05,
            ry0=0.10,
            rx1=0.70,
            ry1=0.70,
            min_match=0.58,
            scale_min=0.95,
            scale_max=1.05,
            scale_count=3,
        )
        reference = np.full((480, 640, 3), 100, np.uint8)
        xx = np.arange(640, dtype=np.uint8)
        yy = np.arange(480, dtype=np.uint8)
        reference[:, :, 0] = 80 + (xx * 3) % 90
        reference[:, :, 1] = (90 + (yy * 2) % 70)[:, None]
        reference[:, :, 2] = 70 + ((xx + yy[:, None]) * 2) % 80
        cv2.ellipse(reference, ((190, 175), (12, 16), 0), (20, 20, 40), -1)
        for index, y in enumerate(range(166, 186, 3)):
            color = (240, 240, 240) if index % 2 == 0 else (5, 5, 5)
            cv2.line(reference, (178, y), (202, y), color, 1)

        missing = reference.copy()
        hole = np.zeros(reference.shape[:2], np.uint8)
        cv2.ellipse(hole, ((190, 175), (12, 16), 0), 255, -1)
        shifted = np.roll(reference, -40, axis=1)
        missing[hole > 0] = shifted[hole > 0]

        gray = cv2.cvtColor(missing, cv2.COLOR_BGR2GRAY)
        result, note = SidePinholeTracker(reference).detect(gray, params)
        self.assertIsNone(result)
        self.assertIn("belum terpasang", note)

        present = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        kept, kept_note = SidePinholeTracker(reference).detect(present, params)
        self.assertIsNotNone(kept)
        self.assertGreater(kept["score"], 0.9)
        self.assertIn("TM", kept_note)
