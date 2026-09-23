# Pelacak lubang jarum

Monitor dua panel: **kiri SIDE** (samping), **kanan TOP** (atas). Deteksi oval memakai template matching multi-skala + ECC. Koordinat dan diameter dalam piksel pada referensi 640×480, bukan ukuran metrologi.

```
pinhole_gui.py          # titik masuk GUI
pinhole/
  params.py             # field, default, validasi
  vision.py             # pemrosesan citra + PinholeTracker
  thread_tip.py         # ujung bebas benang coklat
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
- Tab Citra / HSV / Tracking / Benang / Kalibrasi / Kamera.
- Bekukan frame, drag oval, template, dan area pencarian, lalu terapkan.
- Simpan/muat profil JSON, snapshot, dan rekam overlay.
- Tab Benang menyetel HSV, sisi masuk, dan perataan ujung. Tampilan Mask benang menunjukkan segmentasinya.

## Ujung benang

Kedua panel melacak ujung bebas benang coklat/tembaga yang masuk dari tepi gambar. Titik ukur subpiksel ada di mask sebelum penutupan morfologi, jadi kernel tidak menggeser ujung.

Saat perpindahan di bawah radius diam (default 3.2 px/frame, dan 1.6 px untuk perataan lebih kuat), koordinat diratakan supaya marker tidak bergetar. Di atas radius itu, marker langsung mengikuti pengukuran baru.

| Konstanta | Default |
| --- | --- |
| HSV | H 0–22, S 60–255, V 18–230 |
| Luas / lebar minimum | 250 px / 30 px |
| Batas sisi masuk | 0.50 (komponen melewati tengah, ke arah tepi masuk) |
| Rasio tebal ujung | 0.30 |

Default menganggap benang masuk dari tepi kanan. Untuk kamera yang benangnya masuk dari kiri, matikan centang itu di tab Benang. Profil JSON versi 3 menyimpan parameter ini per panel.

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
