# Pelacak lubang jarum (samping)

Pelacak oval untuk rekaman kamera samping pada benda kerja logam. Skrip ini mengikuti alur asli template matching + ECC, dengan kalibrasi ulang untuk `side.webm`: lubang jarum adalah bukaan tabung gelap di ujung kiri permukaan yang mengkilap.

Oval referensi dikalibrasi secara visual pada bukaan, bukan ukuran metrologi. Jika video atau kamera berubah, kalibrasi ulang `REFERENCE_ELLIPSE`, `TEMPLATE_BOX`, dan `calib/template_side.png`.

## Dependensi

- Python 3.10+
- `numpy`, `opencv-python`
- `ffmpeg` dan `ffprobe` di PATH

```bash
python -m pip install -r requirements.txt
```

## Jalankan

```bash
python pinhole_presisi.py samples/side.webm hasil.mp4
```

Keluaran:

- `hasil.mp4` — video anotasi (oval hijau, titik tengah merah, stempel koordinat)
- `hasil.json` — pusat, sumbu, sudut, dan skor per frame

Frame tanpa lubang (awal rekaman, kamera bergeser, atau buram) ditandai `PINHOLE: tidak terdeteksi`.

Template kustom:

```bash
python pinhole_presisi.py rekaman.webm keluar.mp4 --template calib/template_side.png
```

## Kalibrasi

Konstanta di `pinhole_presisi.py` berlaku untuk resolusi **640×480**. Panjang sumbu oval adalah diameter OpenCV.

| Konstanta | Nilai samping (side.webm) |
| --- | --- |
| `REFERENCE_ELLIPSE` | pusat `(190, 171)`, sumbu `(24, 44)`, sudut `-2°` |
| `TEMPLATE_BOX` | `(168, 138, 48, 64)` |
| `SEARCH_BOX` | kiri-tengah frame, tempat bukaan biasanya muncul |

`calib/template_side.png` adalah crop grayscale bukaan yang jelas (sekitar t=15s). Tracker **tidak** mengambil template dari frame pertama, karena pada video samping lubang belum terlihat di awal.

Overlay kalibrasi: `calib/reference_overlay.png`.
