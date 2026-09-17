"""
GUI dua sumber kamera/video untuk pelacakan pinhole.

Jalankan:
    python pinhole_gui.py --side samples/side.webm --top top.webm

Kiri  SIDE: template matching + ECC, kalibrasi samples/side.webm.
Kanan TOP: template matching multi-skala + ECC affine (algoritme tidak diubah).

Kalibrasi disimpan dalam koordinat referensi 640 x 480.
Diameter oval dan koordinat hasil merupakan piksel, bukan ukuran metrologi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pinhole.params import ROOT, SIDE_SAMPLE
from pinhole.ui import STYLE, MainWindow


def existing_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    alt = ROOT / value
    if alt.is_file():
        return str(alt.resolve())
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top",
        default="top.webm",
        help="Video awal panel TOP (kanan)",
    )
    parser.add_argument(
        "--side",
        default=str(SIDE_SAMPLE) if SIDE_SAMPLE.is_file() else "",
        help="Video awal panel SIDE (kiri)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    window = MainWindow(existing_path(args.top), existing_path(args.side))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
