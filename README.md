# Pelacak lubang jarum

Monitor dua panel: **kiri SIDE** (samping), **kanan TOP** (atas). Kedua panel memakai template matching multi-skala + ECC. Koordinat dan diameter dalam piksel pada referensi 640×480, bukan ukuran metrologi.

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

Algoritme panel TOP tidak diubah. Panel SIDE memakai template matching yang sama, dengan kalibrasi `samples/side.webm` dan penolak loncatan pusat.

## Dependensi

Python 3.10+, `ffmpeg`/`ffprobe` (untuk CLI batch), dan:

```bash
python -m pip install -r requirements.txt
```

## GUI

```bash
python pinhole_gui.py --side samples/side_train.mp4 --top top.webm
```

- Mulai masing-masing panel, atau kamera V4L2.
- Tab Citra / HSV / Tracking / Benang / Kalibrasi / Kamera.
- Bekukan frame, drag oval, template, dan area pencarian, lalu terapkan.
- Simpan/muat profil JSON, snapshot, dan rekam overlay.
- Tab Benang menyetel HSV, sisi masuk, dan perataan ujung. Tampilan Mask benang menunjukkan segmentasinya.

## Ujung benang

Kedua panel melacak ujung bebas benang coklat/tembaga yang masuk dari tepi gambar. Titik ukur subpiksel ada di mask sebelum penutupan morfologi, jadi kernel tidak menggeser ujung. Pada SIDE, kawat yang tipis sering tertutup: pecahan sebaris disambung sampai celah 48 px, sementara ujung tetap diukur pada piksel benang yang terlihat. Ujung yang dicari adalah benang cokelat yang gelap, bukan tepi logam yang lebih terang. Saat aplikasi dibuka, sumber SIDE bawaan adalah `samples/side_train.mp4`; template lubang dipotong dari frame pertama video itu.

Saat perpindahan di bawah radius diam (default 3.2 px/frame, dan 1.6 px untuk perataan lebih kuat), koordinat diratakan supaya marker tidak bergetar. Di atas radius itu, panel TOP langsung mengikuti pengukuran baru. Panel SIDE menggeser ujung pelan: 4 px pada frame pertama, lalu paling banyak 16 px per frame bila ujung yang sama terukur lagi.

| Konstanta | Default |
| --- | --- |
| HSV TOP | H 0–22, S 60–255, V 18–230 |
| HSV SIDE | H 0–28, S 40–255, V 30–210 |
| Luas / lebar minimum | TOP 250 px / 30 px, SIDE 40 px / 12 px |
| Batas sisi masuk | 0.50 (komponen melewati tengah, ke arah tepi masuk) |
| Rasio tebal ujung | 0.30 |

Default menganggap benang masuk dari tepi kanan. Untuk kamera yang benangnya masuk dari kiri, matikan centang itu di tab Benang. Profil JSON versi 3 menyimpan parameter ini per panel.

Oval ujung dikunci ke lebar kalibrasi karena lebar benang tidak berubah. Defaultnya ramping dan tegak: lebar 8 px, panjang 22 px, sudut 90 derajat pada referensi 640×480. Posisi oval tetap mengikuti ujung. Sesuaikan dengan drag **Oval ujung benang** di tab Kalibrasi, atau tombol **Kunci oval ke lebar benang**. Angka oval memakai referensi 640×480, sama seperti oval lubang.

Pada overlay, garis putus-putus hijau adalah bukaan jalur jarum. Garis putus-putus kuning adalah jalur benang, dan oval kuning yang ramping menandai ujung. Garis tegak hijau ada di sisi lubang yang menghadap benang. Insertion siap bila jalur kuning berada di dalam jalur hijau (`JALUR SELARAS`). Jika kuning keluar dari hijau, ujung menabrak dinding lubang (`BENANG MELEWATI JARUM`). Centang di tab Benang bisa mematikan garis ini.

Jika `top.webm` belum ada, isi path di panel kanan atau biarkan kosong sampai ada sumber.

## Batch SIDE

```bash
python pinhole_presisi.py samples/side_train.mp4 hasil.mp4
```

Menulis `hasil.mp4` dan `hasil.json`. CLI memakai template matching yang sama dengan panel SIDE.

## Kalibrasi SIDE

| Konstanta | Nilai (`samples/side.webm`) |
| --- | --- |
| Oval | pusat `(190, 171)`, sumbu `(24, 44)`, sudut `-2°` |
| Template | `(168, 138, 48, 64)` |
| Search | `(0.00, 0.16)–(0.52, 0.55)` |

Referensi penuh: `calib/reference_side.png`. Kalibrasi ulang jika kamera, fokus, atau geometri pengambilan berubah.
