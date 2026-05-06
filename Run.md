# Panduan Menjalankan Model Deteksi Penyakit Cabai (CNN)

Dokumen ini berisi tahapan lengkap untuk menyiapkan lingkungan (environment) dan menjalankan model deteksi penyakit daun cabai.

---

## 1. Struktur Folder Proyek

Proyek ini menggunakan struktur modular untuk kerapian dan kemudahan manajemen:
```text
Model/
├── data/               # Folder dataset utama
│   └── Pepper_Dataset/
├── models/             # Folder penyimpanan file model (.keras)
├── src/                # Kode sumber modul (Data Loader, Arsitektur)
├── tests/              # Script untuk pengujian otomatis
├── uploads/            # Folder sementara untuk file yang diunggah API
├── app.py              # Server API (Flask)
├── train_model.py      # Script untuk melatih model
├── predict.py          # Script untuk prediksi via CLI
└── evaluate_model.py   # Script untuk evaluasi performa model
```

---

## 2. Persiapan Lingkungan (Setup Environment)

Pastikan Anda sudah menginstal **Python 3.8 - 3.11**.

### a. Buat & Aktifkan Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### b. Instal Library
```bash
pip install -r requirements.txt
```

---

## 3. Menjalankan Training

Jika ingin melatih ulang model:
```bash
python train_model.py
```
*Hasil akan disimpan di folder `models/` sebagai `best_pepper_model.keras`.*

---

## 4. Melakukan Prediksi (CLI)

Gunakan `predict.py` untuk menguji satu gambar:
```bash
python predict.py <path_ke_gambar_anda.jpg>
```
**Contoh:**
```bash
python predict.py data/Pepper_Dataset/Healthy/healthy_1.jpg
```

---

## 5. Menjalankan API Server (Flask)

Untuk integrasi dengan aplikasi eksternal:
```bash
python app.py
```
- **Endpoint:** `http://localhost:5000/predict`
- **Method:** `POST`
- **Body:** `form-data` dengan key `file`.

---

## 6. Evaluasi & Testing

- **Evaluasi Performa:**
  ```bash
  python evaluate_model.py
  ```
- **Unit Testing:**
  ```bash
  pytest tests/test_model.py
  ```

---

### Tips Troubleshooting
1. **Model Tidak Ditemukan:** Pastikan file `.keras` ada di dalam folder `models/`.
2. **Dataset Tidak Ditemukan:** Pastikan folder dataset ada di `data/Pepper_Dataset/`.
3. **Error Import:** Selalu jalankan script dari folder utama (root) agar modul di `src/` terdeteksi.
