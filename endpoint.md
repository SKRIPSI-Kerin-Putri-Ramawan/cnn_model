# Dokumentasi API - Pepper Disease Detection

API ini digunakan untuk mendeteksi penyakit pada daun cabai menggunakan model CNN yang telah dilatih.

## Base URL
```
http://localhost:5000
```

---

## 1. Health Check
Digunakan untuk memastikan server API berjalan dengan baik dan mengecek penggunaan hardware (CPU/GPU).

- **Endpoint**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "active",
    "device": "cuda",
    "cuda_available": true
  }
  ```

---

## 2. Prediksi Penyakit
Endpoint utama untuk mengunggah gambar daun dan mendapatkan hasil klasifikasi penyakit.

- **Endpoint**: `/predict`
- **Method**: `POST`
- **Request Type**: `multipart/form-data`
- **Body**:
  - `file`: (File Image - .jpg, .jpeg, .png)

- **Response (Success)**:
  ```json
  {
    "class": "Leaf_Curl",
    "confidence": 0.985,
    "status": "success"
  }
  ```

- **Response (Error)**:
  ```json
  {
    "error": "Tidak ada file dikirim",
    "status": "failed"
  }
  ```

---

## Contoh Penggunaan (cURL)
Anda dapat mencoba API ini melalui terminal menggunakan cURL:

```bash
curl -X POST -F "file=@path/ke/gambar/anda.jpg" http://localhost:5000/predict
```

## Daftar Kelas Penyakit
Model ini dapat mendeteksi 4 kategori:
1. `Bacterial Spot`
2. `Cerespora`
3. `Healthy` (Sehat)
4. `Leaf_Curl` (Daun Keriting)
