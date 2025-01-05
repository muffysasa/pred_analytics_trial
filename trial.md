# Proyek 1 Predictive Analytics - Credit Score Risk
## 1. Domain Project

![Image](./img/Businessman%20pushing%20credit%20score%20speedometer%20from%20poor%20to%20good.jpg)
*all source pictures here from [Freepik](https://www.freepik.com/)*

Kredit bank merupakan penyediaan uang atau tagihan, berdasarkan persetujuan atau kesepakatan pinjam-meminjam antara bank dan debitur, yang mewajibkan pihak debitur untuk melunasi hutangnya setelah jangka waktu tertentu yang disepakati. Namun dalam aktivitas kredit bank terdapat resiko kredit yang disebabkan faktor dari debitur tersebut untuk menunda pelunasan hutang berujung melakukan gagal bayar, dimana hal tersebut berujung merugikan pihak bank [Resiko Kredit](https://www.ocbc.id/id/article/2022/02/24/risiko-kredit-adalah?_gl=1*1ihxzca*_gcl_au*MTIzMzU0NDI2MS4xNzI4Nzk5ODEy). Analisis resiko kresit pada Bank merupakan aktivitas penilaian kredit Bank terhadap debitur untuk menghindari kerugian bank terhadap debitur yang memiliki potensi resiko kredit [Analisis Resiko Kredit](https://www.ocbc.id/id/article/2022/11/15/analisis-kredit-adalah). Jurnal: [Jurnal Analisis Resiko Kredit](https://journal.metansi.unipol.ac.id/index.php/jurnalmetansi/article/view/87)


## 2. Business Understanding

**- Problem Statement**

Dalam dunia perbankan dan lembaga keuangan, pengelolaan risiko kredit merupakan aspek penting untuk mendapatkan profitabilitas suatu lembaga. Masalah umum pada kredit bank sebagian besar berikut:
1. Meningkatnya jumlah pinjaman yang diberikan kepada debitur dari berbagai latar belakang
2. Kurangnya penyesuaian kontrak kredit pada tanpa analisis profil debitur
3. Profitabilitas perusaahaan menurun karena banyaknya debitur yang gagal bayar dimana kurangnya bank untuk menargetkan analisis resiko kredit.

Berdasarkan latar belakang tersebut project ini memiliki batasan masalah sebagai berikut:
1. **Bagaimana langkah-langkah untuk menurunkan debitur yang berpotensi gagal bayar?**
  --> Melakukan metode sederhana CRISP-DM (Cross-Industry Standard Process for Data Mining) yaitu melakukan proses analitik pada data history/profil kreditur bank tersebut untuk predictive analytics pada skor nilai resiko kredit.
2. **Apa variabel pada profil debitur yang mempengaruhi debitur gagal bayar?**
  --> Menganalisis data dan mencari korelasi dari data history terhadap variabel skor resiko kredit
3. **Bagaimana cara membangun model Machine Learning sebagai solusi untuk analisis resiko kredit pada nasabah menggunakan data history profil debitur?**
  --> Analisis data dan membangun model Predictive Analytics dangan eksplorasi algoritma Machine Learning Regresi dengan melihat pada MSE, MAE dan R2 sebagai parameter algoritma model yang memiliki error paling kecil


**- Goals**

Analisis kredit sangat penting dalam dunia keuangan, terutama bagi lembaga perbankan dan pemberi pinjaman. bertujuan untuk:
1.   **Prediksi dini pada calon debitur yang berpotensi gagal bayar** dimana analisis kredit dilakukan untuk menilai risiko dari calon debitur atau nasabah. Dengan prediksi nilai kredit pada masing-masing debitur. Sehingga pihak bank dapat memastikan bahwa mereka meminjamkan uang kepada individu atau perusahaan yang mampu membayar kembali pinjaman tersebut. Dengan analisis kredit yang baik, risiko kredit macet (non-performing loans) dapat diminimalisir.
2.   **Prediksi skor penilaian debitur**, analisis ini bertujuan untuk memahami sejauh mana kemampuan calon peminjam dalam membayar kembali pinjaman. Ini melibatkan analisis terhadap pendapatan, stabilitas keuangan, dan arus kas dari calon debitur. Penilaian ini membantu lembaga keuangan dalam menentukan besarnya pinjaman yang layak diberikan
3.  **Menurunkan jumlah debitur yang melakukan gagal bayar dan menyesuaikan kontrak terhadap calon debitur**, Berdasarkan hasil analisis kredit, lembaga keuangan dapat menyesuaikan kontrak dengan nasabah seperti tingkat suku bunga yang sesuai bagi calon debitur. Sehingga kontrak terhadap debitur seperti suku bunga dapat menyesuaikan debitur untuk menghindari potensi gagal bayar. Peminjam dengan profil risiko yang rendah biasanya mendapatkan suku bunga yang lebih rendah yang dapat memudahkan debitur melunaskan hutangnya, sedangkan yang berisiko tinggi mendapatkan suku bunga yang lebih tinggi yang akan dihindari bank.

**- Solution statements**
Menggunakan algoritma regresi dengan mengeksplore algortima regresi dengan hyperparameter default dari library sklearn yang dapat terukur baik menggunakan MSE, MAE dan R2

## 3. Data Understanding
Data yang digunakan bersumber [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), data tersebut berisi profil nasabah pada lembaga keuangan/bank yang memiliki history kredit bank, berjumlah 32581 baris dengan 12 kolom. Dan berikut penjelasan kolom-kolom pada data berikut:

- person_age	= Age (Usia Nasabah)
- person_income	= Annual Income (Pendapatan per bulan) --> diasumsi kan menggunakan Dollar
- person_home_ownership	= Home ownership (Kepemilikan tempat tinggal)
- person_emp_length	= Employment length in years (Lama pengalaman pekerjaan dalam tahunan)
- loan_intent	= Loan intent (Kebutuhan peminjaman)
- loan_grade	= Loan grade (Tingkat peminjaman)
- loan_amnt	= Loan amount (Jumlah peminjaman)
- loan_int_rate	= Interest rate (Rate bunga)
- loan_status =	Loan status, 0 is non default 1 is default (Status peminjaman, 1 = gagal bayar dan 0 = sukses bayar)
- loan_percent_income	= Percent income (persen pendapatan untuk peminjaman)
- cb_person_default_on_file	= Historical default (rekap gagal bayar = 1, lunas = 0)
- cb_preson_cred_hist_length = 	Credit history length --> credit score, bigger meaning to bad score (nilai kredit, semakin besar semakin buruk nilai kredit tersebut)

##### 3.1 Mengambil sumber data
Sumber data menggunakan API dari Kaggle dan membuat variabel untuk mengkonversi data csv ke data frame (menggunakan pandas) untuk memudahkan dalam mengelola dan menganalisis data dalam format tabular

```sh
#mendownload data dari Kaggle
!kaggle datasets download -d laotse/credit-risk-dataset
!unzip credit-risk-dataset -d '/content/dataset'
```
```sh
#import library pandas dan membaca data csv
import pandas as pd
data = pd.read_csv('/content/dataset/credit_risk_dataset.csv')
```

##### 3.2 Melihat nilai unik data kategorikal
- kolom person_home_ownership terdiri dari: 'RENT' 'OWN' 'MORTGAGE' 'OTHER'
- kolom loan_intent terdiri dari: PERSONAL' 'EDUCATION' 'MEDICAL' 'VENTURE' 'HOMEIMPROVEMENT' 'DEBTCONSOLIDATION'
- kolom loan_grade: 'A' 'B' 'C' 'D' 'E' 'F' 'G'
- kolom cb_person_default_on_file terdiri dari: 'Y' 'N'

##### 3.3 Melihat jumlah data pada tiap kolom

 - 0   person_age                  32581 non-null  int64
 - 1   person_income               32581 non-null  int64
 - 2   person_home_ownership       32581 non-null  object
 - 3   **person_emp_length           31686 non-null  float64**
 - 4   loan_intent                 32581 non-null  object
 - 5   loan_grade                  32581 non-null  object
 - 6   loan_amnt                   32581 non-null  int64
 - 7   **loan_int_rate               29465 non-null  float64**
 - 8   loan_status                 32581 non-null  int64
 - 9   loan_percent_income         32581 non-null  float64
 - 10  cb_person_default_on_file   32581 non-null  object
 - 11  cb_person_cred_hist_length  32581 non-null  int64

##### 3.4 Melihat nilai null
Jumlah nilai null adalah 3943 pada kolom person_emp_length sejumlah 895 dan loan_int_rate sejumlah 3116

##### 3.5 Melihat nilai duplikat
- Kondisi data terebut memiliki nilai kolom null sebanyak 3943 baris dan 165. Untuk mengurangi bias data maka dilakukan drop nilai null dan duplikat

- Berdasarkan tujuan yaitu predictive analytics maka dilakukan EDA berfokus pada data yang memiliki riwayat dan kredit gagal bayar

##### 3.6 EDA-Melihat data pada riwayat gagal bayar
Data yang akan digunakan akan berfokus pada debitur yang memiliki riwayat gagal bayar. Step ini akan menganalisis apakah debitur yang memiliki riwayat gagal bayar memiliki potensi status kredit gagal bayar atau lunas?

- Melihat data dengan status kredit berdasarkan debitur yang memiliki riwayat kredit (pada kolom cb_person_default_on_file)
    1. Berdasarkan yang memiliki riwayat gagal  bayar (kolom cb_person_default_on_file = 'Y') dan status kredit (kolom loan_status = 1 dan 0), data menunjukkan total debitur yang memiliki riwayat gagal bayar 5475 dengan status kredit gagal bayar 2172 sedangkan kredit lunas 3573.

    2. Berdasarkan yang memiliki riwayat gagal  bayar (kolom cb_person_default_on_file = 'N') dan status kredit (kolom loan_status = 1 dan 0), data menunjukkan total debitur dengan riwayat lunas 26836 dengan status kredit gagal bayar 4936 sedangkan kredit lunas 21900.

**- Berdasarkan data yang memiliki riwayat gagal bayar memiliki potensi kredit gagal bayar dan lunas, sedangkan yang memiliki riwayat lunas memiliki potensi lunas yang tinggi namun ada kemungkinan gagal bayar.**

**- Berdasarkan analisis sementara tersebut maka riwayat lunas dan memiliki potensi kredit lunas data tersebut akan di drop (kolom cb_person_default_on_file = N dan loan_status = 0). Bertujuan mengurangi bias data yang memiliki profil kredit yang lunas.**

Data awal berjumlah 32581, didrop data riwayat lunas dan memiliki potensi kredit lunas sejumlah 21900. Data menjadi 10681

##### 3.7 EDA -  Melihat data tingkat kredit (kolom loan_grade dari A - G) dan status kredit gagal bayar dan lunas (kolom loan_status = 0 dan 1)
status kredit gagal bayar dan lunas (kolom loan_status = 0 dan 1)
Data saat ini berfokus pada profil kreditur yang memiliki riwayat gagal bayar dan potensi kredit gagal bayar. Pengklasifikasian tingkat kredit berdasarkan riwayat kredit, dimana debitur yang memiliki riwayat gagal bayar diklasifikasikan status kredit tidak aman. Step ini melihat apakah status loan_grade berpotensi kredit gagal bayar?

- Melihat klasifikasi kredit pada debitur yang memiliki status kredit gagal bayar dan lunas
    1. Debitur yang memiliki status kredit gagal bayar adalah sejumlah klasifikasi kredit sebagai berikut:
        - Tingkat A: 0
        - Tingkat B: 0
        - Tingkat C: 2615
        - Tingkat D: 749
        - Tingkat E: 174
        - Tingkat F: 34
        - Tingkat G: 1
    2. Debitur yang memiliki status kredit lunas adalah sejumlah klasifikasi kredit sebagai berikut:
        - Tingkat A: 1073
        - Tingkat B: 1701
        - Tingkat C: 1339
        - Tingkat D: 2141
        - Tingkat E: 621
        - Tingkat F: 170
        - Tingkat G: 63

**- Berdasarkan data yang memiliki riwayat kredit gagal bayar, data kredit yang gagal bayar terklasifikasi loan_grade : C, D ,E, F, G sedangkan kredit lunas terklasifikasi A, B, C, D, E, F, G**

**- Analisis sementara: loan_grade C, D, E, F, G berpotensi kredit gagal bayar, sehingga data loan_grade A dan B akan didrop. Bertujuan mengurangi bias data yang memiliki tingkat kredit aman**

Data berjumlah 10681, akan didrop dengan data status kredit lunas dan tingkat kredit aman yaitu A sejumlah 1073 dan B sejumlah 1701. Data menjadi 10681.

Sehingga kondisi data saat debitur dengan riwayat kredit gagal bayar, dan klasifikasi kredit tidak aman (C - D) dengan jumlah baris 7909

##### 3.8 Membuang data yang tidak digunakan -> kolom data yang tidak digunakan, data null dan data duplikat
Setelah melakukan drop data yang memiliki riwayat lunas dan status kredit aman, pada step 3.4 dan 3.5 data terdapat null dan duplikat. Sehingga perlu untuk drop kondisi data yang berpotensi bias dan tidak digunakan. Yaitu drop data null dan duplikat juga dilakukan drop kolom yang tidak akan digunakan. Jumlah data setelah null dan duplikat didrop menjadi 7000.  Selanjutnya drop kolom yang tidak digunakan yaitu kolom 'loan_percent_income', 'loan_grade', 'loan_status', 'cb_person_default_on_file' sehingga jumlah kolom menjadi 8 kolom.

##### 3.9 Mengecek nilai null dan duplikat dan melihat jumlah data masing-masing kolom not null dan tidak terduplikat
Kondisi setelah pembersihan data adalah jumlah baris dan kolom (7000, 8)
- 0   person_age                  7000 non-null   int64
- 1   person_income               7000 non-null   int64
- 2   person_home_ownership       7000 non-null   object
- 3   person_emp_length           7000 non-null   float64
- 4   loan_intent                 7000 non-null   object
- 5   loan_amnt                   7000 non-null   int64
- 6   loan_int_rate               7000 non-null   float64
- 7   cb_person_cred_hist_length  7000 non-null   int64

##### 3.10 Memperbaiki nama kolom dan menyesuaikan urutan kolom
Tujuan step ini untuk mempermudah dalam analisis data tabular secara penamaan kolom dan urutan kolom.
Merubah nama kolom sebagai berikut:
- person_home_ownership': 'home_status',
- 'person_emp_length': 'experience_work',
- 'loan_amnt': 'loan_amount',
- 'cb_person_cred_hist_length' : 'credit_score'

Dan mengurutkan kolom dari persona nasabah lalu variabel kredit
- persona nasabah: person_age, experience_work, person_income, home_status
- variabel kredit: loan_amount, loan_int_rate, loan_intent, credit_score

##### 3.11 Melihat rentang data dengan analisis statistik deskriptif
Berdasarkan deskripsi nilai max dan Q3 person_age 70 dan experience work 123, nilai-nilai tersebut memiliki persebaran kurang tepat pada kondisi kolom-kolom tersebut menunjukkan terdapat outlier sehingga yang diperlukan pembersihan data lagi

##### 3.12 Visualisasi data boxplot-> melihat outlier pada kolom age dan experience_work
Visualisasi boxplot menampilkan data outlier pada kolom person_age rentang 40 - 70 dan experience_work hanya 120

##### 3.13 Visualisasi data boxplot -> Membuang outlier dan melihat data tanpa outlier
Step membuang data outlier yang digunakan ada lah teknik IQR=Q3-Q1. Setelah dilakukan pembersihan data outlier maka rentang kolom person_age 20-40 dan rentang kolom experience_work 0-16
Jumlah baris dan kolom data menjadi (6613, 8)

##### 3.14 Visualisasi Data Univariat- Kategorikal
Jumlah dominasi debitur memiliki rumah dengan tanpa kepemilikan yaitu kepemilikan sewa mencapai 61.6% dan hipotek sebesar 32.1%. Nasabah yang mengajukan kredit untuk memiliki kebutuhan paling tinggi adalah Medis namun juga disusul kebutuhan lain pendidikan. Dan paling rendah untuk kebutuhan perbaikan rumah

##### 3.15 Visualisasi Data Univariat- Kategorikal
Jumlah nasabah mengajukan kredit adalah debitur yang berumur rentaang 23-25 mencapai lebih dari 300. Banyak debitur mengajukan kredit yang masih memiliki pengalaman kerja < 5 tahun. Dan jumlah yng tinggi pada pendapatan debitur pertahun adalah < 200000

##### 3.16 Visualisasi data multivariat
Berdasarkan visualisasi color grade menunjukkan Ascending

- Pada kolom home_status, rata-rata credit_score menunjukan sama masing-masing home_status selain color grade terendah. Rentang home_status terhadap rata-rata credit score 5-6. Grade tertinggi yaitu grade kreditur memiliki kepemilikan rumah hipotik.
Dapat disimpulkan kolom home_status memiliki pengaruh kecil karna rata-rata yang tidak ada cenderung pada masing-masing nilai home_status
- Pada kolom loan_intent, rata-rata loan_intent menunjukan rata-rata credit score yang memiliki rentang sama yaitu 5-6. Rentang loan_intent terhadap rata-rata credit score 5-6. Grade tertinggi yaitu grade kreditur memiliki tujuan kredit untuk Medis.
Dapat disimpulkan kolom loan_intent memiliki pengaruh kecil karna rata-rata yang tidak ada cenderung pada masing-masing nilai loan_intent

- Tidak ada korelasi kuat selain 0.84 pada korelasi kuat antara person_age dan credit_score
- Tidak ada korelasi kuat pada kolom credit_score hanya memiliki korelasi kuat pada person_age dimana persebaran kearah kiri. Riwayat skor kredit semakin tinggi terhadap usia kreditur yang semakin tua
- Pada visualisasi korelasi tersebut yang memiliki korelasi kuat pada kolom target "credit_score" hanya kolom "person_age". Dimana algoritma yang akan digunakan adalah algoritma non-linear/berbasis jarak

## 4. Data Preparation
Kondisi data terebut memiliki nilai kolom null sebanyak 3943 baris dan 165. Untuk mengurangi bias data maka dilakukan drop nilai null dan duplikat

Berdasarkan tujuan yaitu predictive analytics maka dilakukan EDA berfokus pada data yang memiliki riwayat dan kredit gagal bayar

##### 4.1 Labelling Data - melakukan label data kategorikal ke data angka
Untuk membangun model data input secara keseluruhan adalah numerik. Pada data kategorikal data type string perlu dikonversi ke numerik. Metode yang akan digunakan menggunakan Label Encoder dimana nilai kategorikal akan berupa numerik.
- Informasi label untuk kolom home_status -> nama kolom setelah konversi Label Encoder home_status_en:

    |Nama Unik| |
    |---------|-|
    |MORTGAGE	|0|
    |OTHER  	|1|
    |OWN      |2|
    |RENT     |3|


- Informasi label untuk kolom loan_intent -> nama kolom setelah konversi Label Encoder loan_intent_en:
  |Nama Unik         |	 |
  |------------------|---|
  |DEBTCONSOLIDATION |	0|
  |EDUCATION         |  1|
  |HOMEIMPROVEMENT   |	2|
	|MEDICAL           |  3|
	|PERSONAL        	 |  4|
	|VENTURE	         |  5|

##### 4.2 Data splitting -> rasio 90:10 dan menentukan kolom target dan kolom variabel dan melakukan sampling data training menggunakan SMOTE
Pembagian data akan menggunakan rasio 90:10 dimana data training berjumlah 5951 dan data testing 662

##### 4.3 Balancing Data --> Menggunakan teknik SMOTE
Berdasarkan visualisasi step 3. 15 persebaran data pada nilai-nilai kolom credit_score tidak seimbang (imbalance data) dimana, skor nilai yang semakin tinggi menunjukkan jumlah data yang kecil. Project ini berfokus prediksi terhadap variabel yang memiliki nilai kredit yang tinggi untuk menghindari potensi debitur gagal bayar. **SMOTE** bekerja dengan cara menciptakan contoh sintetis baru dari kelas minoritas dengan mengambil dua atau lebih contoh minoritas yang ada dan menginterpolasi di antara mereka. Sehingga diperlukan teknik **SMOTE** pada data agar data debitur yang memiliki skor kredit tinggi dapat seimbang dengan lainnya.

Jumlah data untuk training adalah setelah dilakukan SMOTE:
- Total data keseluruhan: 6613
- Total data training setelah SMOTE: 18816
- Total data testing: 662

##### 4.4 Standarisasi Data --> Standarisasi Data
Tujuan standarisasi adalah proses transformasi data agar memiliki rata-rata (mean) 0 dan standar deviasi 1. Step ini membentuk data lebih seragam, sehingga setiap fitur memiliki skala yang sebanding.

## 5. Modelling menggunakan algoritma Machine Learning Regressor
Berdasarkan pemrosesan data, data tidak ada korelasi kuat antar masing-masing fitur sehingga implementasi algortima akan menggunakan Regresi Non-Linear dan jarak untuk mengenali fitur satu antar lainnya, parameter algortima akan mengggunakan default dari library sklearn
1. KNN : KNN bekerja berbasis jarak dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Algoritma KNN akan mencari/membentuk kedekatan fitur satu dengan lainnya.
  - Pada implementasi akan menggunakan parameter:
    - K yaitu K=10, maka prediksi dari suatu data baru adalah rata-rata nilai target dari 10 titik data terdekat,

2. Gradient Boosting : Boosting bekerja dengan membangun model dari data latih dengan pengoptimalan dengan menggunakan loss function untuk meminimalisir kesalahan secara iteratif. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.
  - Pada implementasi akan menggunakan parameter:
    - n_estimator: 300,  jumlah weak learners atau pohon keputusan yang akan digunakan,
    - learning_rate: 0.1 nilai yang lebih rendah seperti 0.01 atau 0.1 memberikan hasil yang lebih stabil tetapi memerlukan lebih banyak n_estimators, sehingga menggunakan 0.1 karena nilai n_estimator tidak banyak
    - max_depth = 3 menentukan kedalaman maksimum setiap pohon keputusan, semakin besar nilai ini, semakin kompleks pohonnya dan semakin tinggi potensi overfitting


3. AdaaBoost: AdaBoost adalah salah satu algoritma boosting dan bagging yang bekerja dengan cara membangun beberapa model yang lemah (weak learners, sering kali kombinasi algoritma Decision Trees sederhana) secara bertahap secara Bootstrap sampling dan agregrasi prediksi.
Setiap model baru berusaha memperbaiki kesalahan yang dibuat oleh model sebelumnya dengan memberikan bobot lebih besar pada kesalahan tersebut. Model akhir adalah kombinasi berbobot dari semua weak learners.
  - Pada implementasi akan menggunakan parameter:
    - learning_rate: 0.5 nilai yang lebih rendah seperti 0.01 atau 0.1 memberikan hasil yang lebih stabil tetapi memerlukan lebih banyak n_estimators, sehingga menggunakan 0.1 karena nilai n_estimator tidak banyak.
    - random_state: 55 mengontrol pengacakan data dan penentuan nilai awal sehingga hasilnya konsisten jika kode dijalankan kembali


4. HistGradient Boosting: HistGradientBoostingRegressor adalah varian dari Gradient Boosting yang lebih efisien dan dioptimalkan untuk dataset besar.
Algoritma ini mengubah data input menjadi bentuk histogram, yang kemudian digunakan untuk mempercepat proses pembagian dan pelatihan.
  - Pada implementasi akan menggunakan parameter:
    - learning_rate: 0.05 nilai yang lebih rendah seperti 0.01 atau 0.1 memberikan hasil yang lebih stabil tetapi memerlukan lebih banyak n_estimators, sehingga menggunakan 0.1 karena nilai n_estimator tidak banyak.
    - max_iter: 500 menentukan jumlah weak learners atau iterasi boosting yang akan dibangun.
    - max_depth:5 menentukan kedalaman maksimum dari setiap pohon sebagai cara lain untuk mengontrol kompleksitas pohon
    - min_samples_leaf: 10 menentukan jumlah minimum sampel yang diperlukan untuk membentuk sebuah daun.
    - max_bins: 255 menentukan jumlah maksimum bin yang digunakan untuk menyimpan data namun nilai yang lebih tinggi dapat meningkatkan resolusi split
    - l2_regularization: 0.1 menentukan kekuatan regularisasi L2 untuk mengurangi kompleksitas model dan menghindari overfitting
    - random_state: 64 mengontrol pengacakan data dan penentuan nilai awal sehingga hasilnya konsisten jika kode dijalankan kembali
    - early_stopping: True, untuk menghentikan pelatihan lebih awal jika performa tidak meningkat pada data validasi, jika disetel ke True, pelatihan akan berhenti ketika tidak ada perbaikan selama 10 iterasi; False berarti tidak ada penghentian awa
    - random_state: 42 mengontrol pengacakan data dan penentuan nilai awal sehingga hasilnya konsisten jika kode dijalankan kembali


5. DTR + Bagging: Bagging adalah menggabungkan hasil dari beberapa model dasar (base learners) yang dilatih pada subset acak dari data pelatihan.
  - Pada implementasi akan menggunakan parameter:
    - base_estimator: DecisionTreeRegressor default estimatornya
    - n_estimator yaitu 42,  jumlah weak learners atau pohon keputusan yang akan digunakan,
    - random_state: 64 mengontrol pengacakan data dan penentuan nilai awal sehingga hasilnya konsisten jika kode dijalankan kembali

6. Support Vector: SVR dapat menggunakan kernel non-linear seperti Radial Basis Function (RBF) untuk menangkap hubungan yang lebih kompleks.
  - Pada implementasi akan menggunakan parameter:
    - C:100 parameter unutk mengontrol trade-off antara minimization error pada data pelatihan dan kompleksitas model
    - epsilon:0.1  menentukan margin di sekitar prediksi di mana tidak ada penalti untuk kesalahan, contohnya jika epsilon diset ke 0.1, model akan mengabaikan kesalahan prediksi yang berada dalam jarak 0.1 dari nilai aktual.
    - kernel = 'rbf' Tipe kernel yang digunakan untuk mengubah data ke dalam bentuk yang memungkinkan untuk regresi non-linear
    - gamma:0.1 untuk mengatur parameter untuk kernel RBF, yang mengontrol jangkauan dari pengaruh titik pelatihan

7. Random Forest  metode ensemble dari banyak Decision Trees, yang bisa lebih kuat dalam memprediksi target dari data yang kompleks.
  - Pada implementasi akan menggunakan parameter:
    - n_estimator yaitu 50,  jumlah weak learners atau pohon keputusan yang akan digunakan,
    - max_depth = 16 menentukan kedalaman maksimum setiap pohon keputusan, semakin besar nilai ini, semakin kompleks pohonnya dan semakin tinggi potensi overfitting
    - random_state: 55 mengontrol pengacakan data dan penentuan nilai awal sehingga hasilnya konsisten jika kode dijalankan kembali

8. Decision Tree: Merupakan algortima non-linier dengan model pohon keputusan seperti Decision Tree Regressor mampu menangkap hubungan non-linier antar variabel.
  - Pada implementasi akan menggunakan parameter:
    - max_depth = 4 menentukan kedalaman maksimum setiap pohon keputusan, semakin besar nilai ini, semakin kompleks pohonnya dan semakin tinggi potensi overfitting
    - min_samples_split: 5 jumlah minimum sampel yang diperlukan untuk membagi suatu node
    - min_samples_leaf: 3 jumlah minimum sampel yang diperlukan untuk berada dalam suatu daun
    - max_features: None menentukan jumlah fitur yang akan dipertimbangkan saat mencari pemisahan terbaik, default berati menggunakan semua fitur


## 6. Evaluasi model -> MSE, MAE, R2
![Image](./img/data-management-perfomance-graph-concept.jpg)
Evaluasi hasil model Regresi Non-linier akan menggunakan:
- MSE: Mengukur rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Nilai MSE yang lebih rendah menunjukkan model yang lebih baik
- MAE: Mengukur rata-rata dari nilai absolut selisih antara nilai aktual dan nilai prediksi. Nilai MAE rendah akan menunjukan nilai kesalahan yang rendah.
- R2: Mengukur seberapa baik model memprediksi variasi dalam data. Nilainya berkisar antara 0 dan 1. Nilai R² yang lebih mendekati 1 menunjukkan model yang lebih baik.

Nilai MSE, MAE dan R2 training masing-masing algoritma:
|No.|Algoritma         |MSE Train |MAE Train|R2 Train|
|---|------------------|----------|---------|-------|
|1. |Adaptive Boosting |2.884691|1.426624|0.86425|
|2. |DecisionTree      | 2.834147| 1.409014 |0.866628
|3. |GradientBoost| 2.107652| 1.205095| 0.900816
|4. |Random Forest| 0.533179| 0.521623| 0.974909
|5. |HistGradientBoost| 1.34575| 0.930221| 0.936671
|6. |**Bagging Regressor**| **0.135168**| **0.249788**| **0.993639**
|7. |Support Vector Regressor| 2.991074| 1.264286| 0.859244
|8. |KNearestNeighbor| 2.041922| 0.997013| 0.90391

Nilai MSE, MAE dan R2 testing masing-masing algoritma:

|No.|Algoritma         |MSE Train |MAE Train|R2 Train|
|---|------------------|----------|---------|-------|
|1. |Adaptive Boosting | 1.749497| 1.069593| 0.84433|
|2. |DecisionTree      | 1.803272| 1.07573| 0.839545|
|3. |GradientBoost     |1.93315| 1.110244| 0.827988|
|4. |Random Forest     |1.964606| 1.12127|t 0.825189|
|5. |HistGradientBoost| 2.004605| 1.129062| 0.82163|
|6. |**Bagging Regressor**| **2.073779**| **1.148535**| **0.815475**|
|7. |Support Vector Regressor| 3.948882| 1.516849| 0.648629|
|8. |KNearestNeighbor| 4.054033| 1.550302| 0.639272|


Hasil evaluasi matriks -> data prediksi yang digunakan merupakan data continue dan bersifat regresi dengan nilai mse dan mae lebih rendah menunjukkan nilai kesalahan lebih rendah dan R2 yang lebih tinggi menunjukkan model memiliki varians banyak dan model mendekati kesalahan rendah.
1. Bagging Regressor menunjukkan kinerja terbaik di data pelatihan dengan MSE terendah (0.135168), MAE terendah (0.249788), dan R² tertinggi (0.993639). Namun, di data pengujian performanya menurun, tetapi masih relatif baik.

2. Random Forest juga memiliki kinerja baik, dengan R² yang tinggi (0.974909) di pelatihan, tetapi sedikit lebih rendah di pengujian (0.825189).

3. Gradient Boost dan K-Nearest Neighbor menunjukkan kinerja yang rendah di pelatihan maupun pengujian.

4. Support Vector Regressor dan K-Nearest Neighbor memiliki kinerja terburuk di pengujian dan R² yang jauh lebih rendah.

Nilai evaluasi matriks melihat MSE, MAE terendah dan R2 tertinggi
- Training Bagging Regressor: MSE 0.135168, MAE 0.249788, R2 0.993639
- Testing Bagging Regressor: MSE 2.073779, MAE 1.148535, R2 0.815475

**Model dengan evaluasi matriks dengan minimal error adalah: implementasi algortima br(BaggingRegressor)**


![Image](./img/2210.i109.083.F.m004.c9.crisis%20management%20isometric.jpg)

Hasil pengembangan/proyek ini dapat menjawab probelm statement yang dijelaskan pada Business Undestanding sebagai berikut:

1. Analisis resiko kredit merupakan langkah penting untuk mengurangi debitur gagal bayar dimana pihak bank dapat menyetujui kredit atau menyesuaikan kontrak kredit dengan nasabah yang berpotensi gagal bayar. Dengan metode sederhana  CRISP-DM (Cross-Industry Standard Process for Data Mining) yaitu melakukan proses analisis pada data history/profil kreditur bank tersebut, melakukan teknik-teknik feature engineering bertujuan untuk mengkonversi data-data agar bisa diproses model algoritma dan untuk menentukan jenis algoritma predictive analytics pada skor nilai resiko kredit berdasarkan evaluasi matriks. Model predictive analytics yang sudah dilatih dan ditest inilah dapat memprediksi skor kredit untuk menilai kemampuan debitur terhadap kreditnya.

2. Berdasarkan proses analisis data history bank menunjukkan korelasi yang sedikit pada latar belakang persona debitur dan variabel kredit yang diajukan. Hal tersebut mengindikasikan debitur mengajukan kredit dengan latar belakang persona apa saja dan kebutuhan yang beragam. Sehingga hal ini sulit jika dilakukan analisis dan prediksi manual pada resiko kredit. Solusi permasalahan tersebut adalah menggunakan prediksi skor kredi untuk analisis resiko kredit dengan pendekatan algoritma Machine Learning Regresi Non-Linier.

3. Berdasarkan pemrosesan CRISP-DM hasil test MAE, MSE, R2 maka algoritma **Bagging Regressor** menunjukkan prediksi dengan target memiliki nilai eror yang minimal. Dimana prediksi dapat membantu menentukan skor dari persona dan kategori kredit yang akan diajukan debitur. Sehingga pihak bank dapat menganalisis potensi dari calon debitur dan dapat meminimalisir debitur yang berpotensi gagal bayar. Dengan prediksi skor kredit pada calon debitur berdasarkan latar belakang persona debitur dan variabel kredit yang diajukan dapat membantu salah satu penilaian calon debitur terhadap pihak bank.

Poin-poin diatas dapat menjawab goals dimana pihak bank dapat prediksi dini untuk skor penilaian debitur. Dan juga dapat menyesuaikan bunga atau kontrak kredit berdasarkan prediksi dini skor kredit sehingga dapat meminimalisir debitur yang berpotensi gagal bayar.

Solusi proses pengembangan/penilitian yang diterapkan projek ini berdampak besar pada sektor financial dan banking dimana prediksi skor kredit dapat sebagai salah satu parameter penilaian aktivitas keuangan nasabah terhadap bank/lembaga keuangan. Selain itu prediksi pendekatan Machine Learning dapat dikembangkan berbagai parameter sektor keuangan misalnya prediksi kerugian pengeluaran, keuntungan income, prediksi harga, pengklasifikasian nasabah dsb.

























