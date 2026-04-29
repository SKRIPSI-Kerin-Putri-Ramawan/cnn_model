# Rencana Detail Milestone: Deteksi Penyakit Daun Paprika (CNN)

Berikut adalah detail tahapan pengembangan sistem deteksi penyakit daun paprika:

### Phase 1: Data Preparation (Persiapan Data)
Fase ini kritikal untuk memastikan kualitas data yang masuk ke model.
*   **Gather Data**: Mengumpulkan dataset citra daun paprika yang sudah terlabeli (misal: Sehat, Bercak Bakteri, dll).
*   **Preprocessing**:
    *   **Resize**: Menyeragamkan ukuran gambar (contoh: 224x224 px) agar model dapat memproses input secara konsisten.
    *   **Normalization**: Mengubah nilai pixel dari rentang 0-255 menjadi 0-1 untuk mempercepat konvergensi saat training.
*   **Data Augmentation**: Menerapkan transformasi acak seperti rotasi, flip horizontal/vertical, zoom, dan pergeseran menggunakan `ImageDataGenerator` untuk mencegah overfitting dan meningkatkan kemampuan generalisasi model.

---

### Phase 2: Building the Architecture (Pembangunan Arsitektur)
Membangun struktur CNN yang terdiri dari dua bagian utama: *Feature Extractor* dan *Classifier*.
*   **Convolutional Layer**: Menggunakan filter (kernel) untuk mendeteksi fitur lokal seperti tepi (edges), tekstur, dan pola visual pada daun.
*   **Activation Layer (ReLU)**: Memperkenalkan non-linearitas ke dalam jaringan menggunakan fungsi *Rectified Linear Unit* (ReLU).
*   **Pooling Layer (Max Pooling)**: Melakukan downsampling pada feature maps untuk mengurangi dimensi dan beban komputasi sambil tetap mempertahankan informasi penting.
*   **Flattening**: Mengonversi output 3D dari feature extractor menjadi vector 1D untuk masuk ke tahap klasifikasi.
*   **Fully Connected (Dense) Layer**: Melakukan penalaran tingkat tinggi dan memberikan klasifikasi akhir berdasarkan fitur yang telah diekstraksi.

---

### Phase 3: Training and Optimization (Latihan & Optimasi)
Proses melatih model agar mampu mengenali pola secara akurat.
*   **Compile the Model**:
    *   **Optimizer**: Menggunakan `Adam` untuk penyesuaian bobot secara efisien.
    *   **Loss Function**: Menggunakan `Categorical Cross-entropy` untuk klasifikasi multi-kelas.
    *   **Metrics**: Memantau tingkat akurasi (`Accuracy`).
*   **Train (Fit) the Model**: Menyuplai data melalui jaringan selama beberapa *epoch*. Menggunakan *Backpropagation* untuk memperbarui bobot berdasarkan error yang dihasilkan.
*   **Hyperparameter Tuning**: Melakukan penyesuaian pada jumlah filter, learning rate, atau batch size untuk mencapai performa optimal.

---

### Phase 4: Evaluation and Deployment (Evaluasi & Implementasi)
Mengukur keberhasilan model dan mempersiapkannya untuk penggunaan nyata.
*   **Evaluate**: Menguji model pada *unseen data* (data uji) untuk memastikan model tidak mengalami overfitting pada data latih.
*   **Visualization**: Memplot grafik training vs validation loss/accuracy untuk memantau progres dan mendeteksi anomali.
*   **Inference**: Menggunakan model yang telah terlatih untuk melakukan prediksi pada citra daun paprika baru di dunia nyata.

---

### Phase 5: API Integration & Deployment (Integrasi API & N8N)
Menyediakan akses model melalui jaringan agar bisa digunakan oleh aplikasi lain atau workflow automation.
*   **Setup API**: Membangun REST API menggunakan `Flask` atau `FastAPI` untuk melayani request prediksi.
*   **Endpoint /predict**: Membuat endpoint yang menerima unggahan gambar dan mengembalikan hasil prediksi dalam format JSON.
*   **N8N Integration**: Menghubungkan API dengan N8N menggunakan node *HTTP Request* untuk membuat workflow otomatis (misal: kirim gambar via Telegram -> API Deteksi -> Balas dengan hasil).

---
*Dokumen ini merupakan panduan langkah-demi-langkah. Detail implementasi kode akan mengikuti urutan fase di atas.*
