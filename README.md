# Pelacak lubang jarum

Monitor dua panel: **kiri SIDE** (samping), **kanan TOP** (atas). Deteksi oval memakai template matching multi-skala + ECC. Koordinat dan diameter dalam piksel pada referensi 640×480, bukan ukuran metrologi.

```
pinhole_gui.py          # titik masuk GUI
pinhole/
  params.py             # field, default, validasi
  vision.py             # pemrosesan citra + PinholeTracker
  capture.py            # thread kamera/video
  ui.py                 # PySide6
pinhole_presisi.py      # batch CLI untuk rekaman samping
```

Algoritme panel TOP tidak diubah. Panel SIDE memakai kelas yang sama, dengan kalibrasi `samples/side.webm` dan penolak loncatan pusat.

## Dependensi

Python 3.10+, `ffmpeg`/`ffprobe` (untuk CLI batch), dan:

```bash
python -m pip install -r requirements.txt
```

## GUI

```bash
python pinhole_gui.py --side samples/side.webm --top top.webm
```

- Mulai masing-masing panel, atau kamera V4L2.
- Tab Citra / HSV / Tracking / Kalibrasi / Kamera.
- Bekukan frame, drag oval, template, dan area pencarian, lalu terapkan.
- Simpan/muat profil JSON, snapshot, dan rekam overlay.

Jika `top.webm` belum ada, isi path di panel kanan atau biarkan kosong sampai ada sumber.

## Batch SIDE

```bash
python pinhole_presisi.py samples/side.webm hasil.mp4
```

Menulis `hasil.mp4` dan `hasil.json`.

## Kalibrasi SIDE

| Konstanta | Nilai (`samples/side.webm`) |
| --- | --- |
| Oval | pusat `(190, 171)`, sumbu `(24, 44)`, sudut `-2°` |
| Template | `(168, 138, 48, 64)` |
| Search | `(0.00, 0.16)–(0.52, 0.55)` |

Referensi penuh: `calib/reference_side.png`. Kalibrasi ulang jika kamera, fokus, atau geometri pengambilan berubah.
