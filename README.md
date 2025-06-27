# Laporan Proyek Machine Learning - Irfan Fajar Muttaqin

## Domain Proyek

Anime merupakan salah satu industri hiburan yang berkembang pesat secara global, termasuk di Indonesia. MyAnimeList, sebagai salah satu platform terbesar untuk katalog anime dan manga, menyediakan data yang sangat kaya untuk dieksplorasi. Salah satu aspek yang menarik untuk diprediksi adalah **popularitas anime**, yang dalam hal ini direpresentasikan oleh jumlah anggota pengguna ("members") yang telah menambahkan suatu anime ke daftar mereka.

Dengan mengetahui karakteristik apa yang membuat sebuah anime menjadi populer, stakeholder seperti rumah produksi, distributor, hingga platform streaming dapat mengambil keputusan yang lebih tepat dalam proses kurasi konten.

**Referensi**:
- [JikanAPI - Unofficial MyAnimeList API](https://jikan.moe)
- [Analisis Popularitas Studio Anime dari Data MyAnimeList](https://www.jurnal.peneliti.net/index.php/JIWP/article/view/2401)
- [Multimodal Deep Learning for Predicting Anime Popularity Before Major Investments](https://www.researchgate.net/publication/381704609_Anime_Popularity_Prediction_Before_Huge_Investments_a_Multimodal_Approach_Using_Deep_Learning)

## Business Understanding

### Problem Statements
- Bagaimana memprediksi tingkat popularitas sebuah anime (dalam bentuk jumlah anggota) berdasarkan fitur-fiturnya?
- Fitur apa saja yang paling berpengaruh terhadap popularitas anime?

### Goals
- Membangun model terbaik prediksi jumlah anggota (members) dari sebuah anime berdasarkan data fitur seperti skor, genre dan musim tayang.
- Mengidentifikasi fitur-fitur yang memiliki pengaruh paling signifikan terhadap popularitas anime.


### Solution Statements
- Menggunakan tiga algoritma regresi: **Linear Regression**, **Ridge Regression**, dan **Random Forest Regressor** dan membandingkan menggunakan metrik regresi: MAE, RMSE, dan R2-score untuk menemukan hasil terbaik.
- Melakukan visualisasi feature importance pada model terbaik.

---

## Data Understanding

Data dikumpulkan dari MyAnimeList melalui **JikanAPI**. Jumlah data: **8329 baris × 9 kolom** anime dengan berbagai fitur.

[Link Referensi JikanAPI](https://jikan.moe)

### Variabel pada dataset:
- `Rank`: Ranking anime berdasarkan popularitas.
- `Title`: Judul anime.
- `Score`: Skor rating dari pengguna (0-10).
- `Episodes`: Jumlah episode.
- `Members`: Jumlah anggota (target).
- `Genre`: Genre anime.
- `Studio`: Studio produksi.
- `Status`: Status penayangan.
- `Premiered`: Musim dan tahun tayang.

Data disimpan di [Dataset Anime (Raw)](https://drive.google.com/file/d/1d0M5wgBNhgbfIKsc9WWNLFmhi4HEdUlk/view?usp=sharing)

Menambahkan data secara manual yang masih kosong khususnya di kolom episode. Berikut data yang saya tambahkan:

- One Piece : 1126 Episode
- Detective Conan : 1159 Episode
- Crayon Shinchan : 1267 Episode
- Nintama Rantarou : 2321 Episode
- Sazae-san : 7000 Episode
- Ojarumaru : 1767 Episode
- Doraemon : 1787 Episode
- Bonobono : 454 Episode

Tujuan agar data anime terkenal yang episode masih kosong masuk dalam faktor prediksi nantinya. Lalu disimpan lagi di [Dataset Anime](https://drive.google.com/file/d/14nMXWeRFbbCu_j9qk0OTHKbzI2vfNPhb/view?usp=sharing)

Pada data tersebut ada beberapan data `missing_value` berikut datanya:
| Kolom                 | Jumlah Data     | 
|----------------------|---------|
| rank    | 295   | 
| score     | 3284  | 
| episodes       | 357   | 
| genre       | 1272   | 
| studio       | 2767   | 
| premiered      | 531   | 

Jumlah data duplikat : 14

---

### Exploratory Data Analysis (EDA):
![Visualisasi Data](https://i.imgur.com/HFJFhNS.png)

- `score` punya korelasi positif sedang dengan members (0.46) → sangat relevan untuk regresi.
- `rank` berkorelasi negatif kuat dengan score dan lumayan dengan members (-0.41) → bisa dipakai
- `episodes` tidak berkorelasi signifikan dengan members → kemungkinan bukan prediktor kuat.

---

![Visualisasi Data](https://i.imgur.com/s9oz3mr.png)

- Distribusinya right-skewed (miring ke kanan) → banyak anime punya sedikit anggota, dan hanya sebagian kecil yang sangat populer.
- Bisa dipertimbangkan log-transform members saat modeling untuk stabilkan varian.

---

![Visualisasi Data](https://i.imgur.com/VITnX1s.png)

- **Score vs Members** → terlihat adanya pola menaik, tapi menyebar → menunjukkan korelasi positif yang tidak terlalu linear.

---

![Visualisasi Data](https://i.imgur.com/kgcUz3c.png)

- **Episodes vs Members** → tidak ada pola jelas, dan tampak banyak outlier → bukan fitur penting.

---

![Visualisasi Data](https://i.imgur.com/UYCMmvh.png)

**Genre:**
- Action, Adventure, Fantasy dan Comedy, Romance punya members tertinggi.
- Genre bisa jadi fitur penting, terutama jika dikelola jadi fitur multi-label.

---

![Visualisasi Data](https://i.imgur.com/nQcNIqU.png)

**Premiered:**
- Tahun-tahun seperti Spring 2014, Spring 2016, Fall 2015 banyak menghasilkan anime populer.
- Bisa diolah menjadi fitur musim dan tahun.

---

![Visualisasi Data](https://i.imgur.com/wg3mXTH.png)

**Studio:**
- Studio seperti Madhouse, J.C.Staff, dan Pierrot menarik banyak members.
- Bisa encode top-N studio sebagai fitur dummy.

---

## Data Preparation

- Membuat data copy `anime_df_clean` agar data asli `anime_df` tetap utuh dan tidak terpengaruh oleh proses pembersihan data. 
- Menanganani missing value dengan:.
    - `score`: Menghapus kolom yang berisi nilai kosong.
    - `rank`: Missing value diisi dengan nilai -1, sebagai penanda bahwa data tersebut tidak tersedia atau tidak diketahui.
    - `episodes`: Diisi dengan nilai median. Median digunakan karena lebih tahan terhadap outlier daripada (mean).
    - `genre`, `studio`, dan `premiered`:  Diisi dengan label `'Unknown'`. Ini membantu menjaga konsistensi tipe data dan memungkinkan model atau analisis tetap mengenali bahwa nilai tersebut tidak tersedia.
- Menghapus nilai duplikat agar tidak mengganggu distribusi data dan menhindari terjadinya bias dalam regresi.
- Encoding fitur kategorikal (`genre`, `studio`) menggunakan `MultiLabelBinarizer` (karena data lebih dari 1), agar model bisa membaca variabel kategorikal.
- Membuat `anime_encoded_df` dan membuang fitur `genre` dan `studio` agar dikhususkan untuk persiapan data training.
- Melakukan pemisahan data dari kolom `premiered` menjadi `season` dan `year` agar data lebih variatif dan mudah dibaca.
- Mengganti nilai `"Unknown"` menjadi nilai 0 di fitur `year` sebagai penanda bahwa data tersebut tidak tersedia atau tidak diketahui.
- Encoding fitur kategorikal (`premiered`) menggunakan `OneHotEncoder` (karena data cuma ada 1), agar model bisa membaca variabel kategorikal.
- Membuang fitur `year`,`member`, dan `title` di `anime_encoded_df`.
- Melakukan transformasi log pada fitur `members` agar distribusi data jauh lebih baik.
- Split data menjadi training dan test set (80:20) untuk mengukur generalisasi model.

---

## Modeling

### Model 1: Linear Regression

#### Cara Kerja  
Linear Regression adalah metode yang digunakan untuk memodelkan hubungan linear antara satu atau lebih variabel independen (fitur) dengan variabel dependen (target). Tujuan model adalah meminimalkan total galat kuadrat (Sum of Squared Errors) antara nilai prediksi dan nilai aktual dengan pendekatan **Ordinary Least Squares (OLS)**.

Persamaan model:
> 𝑦̂ = β₀ + β₁𝑥₁ + β₂𝑥₂ + ... + βₙ𝑥ₙ

Fungsi loss:
> Loss = ∑(𝑦ᵢ − 𝑦̂ᵢ)²

Linear Regression bekerja optimal saat:
- Terdapat hubungan linear antar variabel.
- Tidak terjadi multikolinearitas antar fitur.
- Error terdistribusi normal dan homoskedastik.

#### Parameter (Default)
- `fit_intercept=True`: Model menghitung nilai intercept (β₀) untuk menggeser garis regresi dari titik nol.
- `copy_X=True`: Data input `X` akan disalin untuk mencegah modifikasi data asli.
- `n_jobs=None`: Menggunakan satu core CPU saat fitting model.
- `positive=False`: Koefisien regresi diizinkan bernilai negatif.

#### Kelebihan & Kekurangan
- ✅ Mudah dipahami, cepat dilatih, dan sangat cocok untuk baseline model.
- ❌ Tidak mampu menangkap hubungan non-linear dan sensitif terhadap outlier.

---

### Model 2: Random Forest Regressor

#### Cara Kerja  
Random Forest adalah algoritma ensemble berbasis decision tree yang menggunakan pendekatan **bagging (bootstrap aggregation)** untuk menghasilkan prediksi yang lebih stabil dan akurat. Setiap pohon dilatih pada subset acak dari data dan subset acak dari fitur. Prediksi akhir diambil dari rata-rata seluruh pohon.

Langkah-langkah umum:
1. Bangun beberapa pohon keputusan dari data bootstrap (dengan pengembalian).
2. Di setiap node, pemisahan dipilih dari subset acak fitur (`max_features`).
3. Prediksi regresi adalah rata-rata dari semua pohon.

Keunggulannya terletak pada:
- Kemampuan menangkap hubungan non-linear.
- Robust terhadap overfitting pada dataset besar.
- Tidak memerlukan scaling fitur.

#### Parameter (Default)
- `n_estimators=100`: Jumlah pohon dalam ensemble.
- `random_state=42`: Seed untuk menjamin hasil yang konsisten.

#### Parameter (Tuning dengan RandomizedSearchCV)
- `n_estimators`: `[100, 200, 300]`  
  → Menentukan berapa banyak pohon yang akan dibuat dalam forest.
- `max_depth`: `[None, 10, 20, 30]`  
  → Kedalaman maksimum pohon. `None` berarti pohon dibiarkan tumbuh sampai leaf node murni.
- `min_samples_split`: `[2, 5, 10]`  
  → Jumlah minimum sampel yang dibutuhkan untuk memisahkan node.
- `min_samples_leaf`: `[1, 2, 4]`  
  → Minimum jumlah sampel untuk menjadi sebuah leaf node.
- `max_features`: `['auto', 'sqrt', 'log2']`  
  → Jumlah fitur yang dipertimbangkan saat mencari pemisahan terbaik.
  - `'auto'` atau `None`: Semua fitur digunakan.
  - `'sqrt'`: Akar kuadrat dari total fitur.
  - `'log2'`: Log basis 2 dari jumlah fitur.

#### Kelebihan & Kekurangan
- ✅ Akurat, menangani non-linearitas, tidak terlalu rentan terhadap overfitting.
- ❌ Interpretasi sulit, waktu prediksi relatif lambat.

---

### Model 3: Ridge Regression

#### Cara Kerja  
Ridge Regression adalah pengembangan dari Linear Regression dengan menambahkan penalti **L2 regularization** pada koefisien. Tujuannya adalah untuk menghindari overfitting dan mengatasi multikolinearitas antar fitur dengan cara mengecilkan koefisien yang terlalu besar.

Fungsi loss Ridge:
> Loss = ∑(𝑦ᵢ − 𝑦̂ᵢ)² + α ∑(βⱼ²)

- `α` (alpha) mengatur kekuatan regularisasi.
- Ridge masih mempertahankan semua fitur dalam model, tapi dengan koefisien yang diperkecil.

#### Parameter (Tuning dengan GridSearchCV)
- `alpha`: `[0.01, 0.1, 1, 10, 100]`  
  → Semakin besar nilainya, semakin kuat regularisasi yang diberikan.
- `fit_intercept`: `[True, False]`  
  → Menentukan apakah akan menghitung intercept pada model atau tidak.

#### Kelebihan & Kekurangan
- ✅ Cocok untuk data dengan multikolinearitas tinggi, lebih stabil dari Linear Regression.
- ❌ Tidak menangani non-linearitas, dan performa menurun jika regularisasi tidak dikalibrasi dengan baik.

---

### Model Terbaik  
Dari hasil evaluasi menggunakan metrik regresi (seperti MSE atau R²), model dengan performa terbaik adalah **Random Forest Regressor dengan hyperparameter tuning**, karena mampu menangkap kompleksitas data dan menghasilkan generalisasi yang baik di data validasi.

---

## Evaluation

### Metrik Evaluasi

Dalam proyek ini, tiga metrik utama digunakan untuk mengevaluasi performa model regresi:

- **MAE (Mean Absolute Error)**  
  Mengukur rata-rata kesalahan absolut antara nilai aktual dan nilai prediksi. Cocok untuk mengukur seberapa jauh prediksi dari kenyataan tanpa memperhitungkan arah deviasi.

- **RMSE (Root Mean Square Error)**  
  Mengkuadratkan selisih antara nilai aktual dan prediksi sebelum dihitung rata-ratanya, lalu diakarkan. Memberikan penalti lebih besar terhadap error besar. Cocok untuk situasi di mana kesalahan besar sangat tidak diinginkan.

- **R² Score (Koefisien Determinasi)**  
  Mengukur seberapa baik model menjelaskan variasi data. Nilai 1 menunjukkan model sempurna, sementara nilai 0 berarti model tidak menjelaskan variasi data sama sekali.

Ketiga metrik ini relevan karena target yang diprediksi bersifat kontinu, yaitu jumlah anggota (members) dari sebuah anime.

#### Rumus Metrik:
- **MAE** = (1 / n) * Σ |yᵢ - ŷᵢ|  
- **RMSE** = sqrt( (1 / n) * Σ (yᵢ - ŷᵢ)² )  
- **R² Score** = 1 - [ Σ (yᵢ - ŷᵢ)² / Σ (yᵢ - ȳ)² ]

---

### Hasil Evaluasi

| Model                         | MAE     | RMSE    | R² Score |
|------------------------------|---------|---------|----------|
| Linear Regression            | 0.8609  | 1.1052  | 0.7159   |
| Ridge Regression             | 0.8715  | 1.1138  | 0.7114   |
| Random Forest (Before Tuning)| 0.7067  | 0.9204  | 0.8029   |
| Random Forest (After Tuning) | 0.7047  | 0.9175  | 0.8042   |

Dari tabel di atas dapat dilihat bahwa model **Random Forest Regressor setelah tuning** memiliki performa terbaik:

- Nilai **MAE dan RMSE paling rendah**, artinya prediksi paling mendekati nilai aktual.
- Nilai **R² tertinggi (0.8042)** menunjukkan bahwa model menjelaskan lebih dari 80% variasi dalam data.

---

### Interpretasi Fitur Paling Berpengaruh

![Visualisasi Data](https://i.imgur.com/kjs1zv3.png)

Berdasarkan fitur importance dari model Random Forest, beberapa fitur paling berpengaruh adalah:

- `rank`: Korelasi terbalik dengan jumlah anggota, semakin tinggi ranking (semakin kecil nilainya), semakin populer animenya.
- `score`: Rating anime yang mencerminkan kualitas menurut penonton.
- `year`: Tahun rilis, menunjukkan efek tren atau potensi jangkauan penonton (platform digital mempercepat distribusi).
- `episodes`: Jumlah episode berpotensi memperpanjang durasi eksposur.

#### Genre Berpengaruh:
- Genre seperti `Romance`, `Action`, `Fantasy`, `Ecchi`, `Suspense`, `Drama`, dan `Comedy` menunjukkan korelasi tinggi dengan popularitas. Genre ini umum di anime populer dan menunjukkan adanya tren preferensi dari penonton.

---

### Hubungan dengan Business Understanding

**Tujuan 1: Membangun model terbaik untuk prediksi jumlah anggota anime.**  
✅ Tujuan ini berhasil dicapai. Random Forest Regressor (After Tuning) memberikan performa prediksi terbaik dengan error rendah dan koefisien determinasi tinggi. Model ini dapat digunakan untuk **memprediksi popularitas anime baru**, misalnya sebelum rilis berdasarkan fitur yang diketahui.

**Tujuan 2: Mengidentifikasi fitur paling berpengaruh terhadap popularitas anime.**  
✅ Tujuan ini juga tercapai. Model berhasil menunjukkan fitur dan genre mana yang paling berpengaruh terhadap jumlah anggota. Informasi ini sangat penting untuk:  
- Studio/Produser dalam menentukan tema dan waktu rilis anime.  
- Platform distribusi (seperti streaming) untuk rekomendasi atau promosi konten.  

**Apakah menjawab problem statement?**  
✔ Ya. Model menjawab pertanyaan “faktor apa yang paling berkontribusi pada popularitas sebuah anime?”

**Apakah setiap solusi berdampak?**  
✔ Ya. Model ini bisa digunakan untuk mendukung pengambilan keputusan bisnis berbasis data (data-driven strategy) dalam industri anime atau platform digital.

---

### Kesimpulan

Model **Random Forest Regressor dengan hyperparameter tuning** merupakan solusi paling optimal karena memberikan hasil prediksi terbaik dan mampu mengidentifikasi fitur penting yang mempengaruhi popularitas anime. Model ini tidak hanya akurat, tapi juga memberikan insight berharga yang dapat digunakan secara langsung oleh stakeholder.


---

