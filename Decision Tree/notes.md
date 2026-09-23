## Dataset

- 6819 perusahaan, 95 fitur rasio keuangan, target `Bankrupt?` (0 = tidak bangkrut, 1 = bangkrut).
- Yang bangkrut hanya 220 (3.2%), data **imbalanced**.
- Tidak ada missing value, tidak ada duplikat, semua fitur sudah angka.

## Steps

- **EDA:** boxplot 6 fitur dengan korelasi tertinggi. Perusahaan bangkrut punya profit (ROA) lebih rendah dan utang lebih tinggi.
- **Hapus `Net Income Flag`:** nilainya sama di semua baris
- **Tidak ada imputasi dan encoding:** karena tidak ada missing value dan semua fitur sudah numerik.
- **Split 80:20 dengan `stratify=y`:** supaya proporsi bangkrut (3.2%) sama di train dan test.
- **Tidak pakai scaling:** tree hanya membandingkan nilai dengan batas ("lebih besar dari X?"), jadi skala fitur tidak berpengaruh.

## Konsep Penting

**Gini dan Entropy**

- Keduanya mengukur seberapa "campur" kelas di satu node.
- Node berisi satu kelas saja = murni = nilai 0. Campuran 50:50 = nilai paling tinggi.
- Tree memilih split yang membuat node paling murni.
- Cara kerjanya mirip, jadi hasilnya biasanya hampir sama.

**Overfitting**

- Model menghafal data train, bukan belajar pola.
- Ciri: akurasi train terus naik, akurasi test tetap atau turun.

**Underfitting**

- Model terlalu sederhana untuk belajar pola.
- Ciri: akurasi train dan test sama-sama rendah atau model hanya menebak satu kelas.

**Kenapa accuracy tidak cukup**

- Menebak "tidak bangkrut" untuk semua data sudah menghasilkan akurasi 96.8%.
- Jadi yang dilihat adalah metrik untuk kelas bangkrut:
  - **Precision:** dari yang ditebak bangkrut, berapa yang benar.
  - **Recall:** dari yang benar-benar bangkrut, berapa yang tertebak.
  - **F1:** gabungan precision dan recall. Dipakai untuk memilih model terbaik.

**Confusion matrix**

- Tabel 2x2 aktual vs prediksi.
- Kanan bawah (bangkrut, ditebak bangkrut) = benar, ingin besar.
- Kiri bawah (bangkrut, ditebak tidak bangkrut) = kesalahan paling berbahaya.

**Feature importance**

- Seberapa besar suatu fitur membantu membuat node lebih murni. Makin tinggi, makin berpengaruh.

---

## Hasil

**Gini vs Entropy** (tanpa batas kedalaman)

| Model   | Accuracy | Precision | Recall | F1     |
| ------- | -------- | --------- | ------ | ------ |
| Gini    | 0.9575   | 0.3409    | 0.3409 | 0.3409 |
| Entropy | 0.9589   | 0.3571    | 0.3409 | 0.3488 |

Hasilnya hampir sama. Keduanya menemukan 15 dari 44 perusahaan bangkrut. Entropy sedikit lebih baik (2 false positive lebih sedikit), jadi Entropy dipakai untuk eksperimen max_depth.

**max_depth** (Entropy)

| max_depth | Train Acc | Test Acc | Test F1   |
| --------- | --------- | -------- | --------- |
| 1         | 0.9677    | 0.9677   | 0.000     |
| 2         | 0.9677    | 0.9677   | 0.000     |
| 3         | 0.9683    | 0.9699   | 0.226     |
| 5         | 0.9743    | 0.9663   | **0.439** |
| 10        | 0.9930    | 0.9604   | 0.357     |
| None      | 1.0000    | 0.9589   | 0.349     |

Depth 1–2: underfitting, tidak pernah menebak bangkrut.
Depth 3: akurasi test tertinggi.
Depth 5: F1 tertinggi, paling baik menemukan perusahaan bangkrut.
Depth 10 dan None: overfitting, train naik sampai 100% tetapi test turun.

**Fitur paling penting**

- `Borrowing dependency` (0.215), jauh di atas fitur lain. Ini juga split pertama di tree.
- Berikutnya: Continuous interest rate (after tax), Persistent EPS in the Last Four Seasons, Net Value Growth Rate, Quick Ratio.
- Fitur ROA tidak semuanya di atas karena saling mirip. Setelah satu dipakai, yang lain tidak menambah informasi.

## notes

- Pemilihan Gini vs Entropy memakai test set. Cara yang lebih benar adalah cross-validation di data train.
- Test set hanya punya 44 perusahaan bangkrut, jadi selisih kecil antar model belum tentu berarti.
- Feature importance diambil dari tree tanpa batas (overfitting). Hanya beberapa fitur teratas yang bisa dipercaya.
- Model Gini dan Entropy memakai setting default agar perbandingan adil.
- `class_weight='balanced'`: agar model lebih memperhatikan kelas bangkrut.
- `ccp_alpha` atau `min_samples_leaf`: cara lain mencegah overfitting.
- `GridSearchCV` dengan `cv=5` dan `scoring='f1'`: mencari hyperparameter terbaik tanpa memakai test set.
- Random Forest: gabungan banyak tree, biasanya lebih stabil.
