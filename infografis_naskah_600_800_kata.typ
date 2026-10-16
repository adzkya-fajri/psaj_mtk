#set page(
  paper: "a4",
  margin: (x: 2cm, top: 2cm, bottom: 2cm),
  footer: context [
    #set text(size: 8.5pt, fill: rgb("#64748b"))
    #grid(
      columns: (1fr, auto),
      [LaborCast • PSAJ Matematika SMK Telkom Purwokerto],
      [Halaman #counter(page).display("1") • Dokumen 677 Kata (Rentang 600–800)],
    )
  ]
)

#set text(
  font: ("Liberation Sans", "DejaVu Sans", "Helvetica"),
  size: 10pt,
  fill: rgb("#0f172a"),
  lang: "id",
)

#set par(justify: true, leading: 0.7em)

// Banner
#rect(
  width: 100%,
  fill: rgb("#0f172a"),
  radius: 6pt,
  inset: (x: 14pt, y: 10pt),
)[
  #grid(
    columns: (1fr, auto),
    gutter: 10pt,
    align: (left + horizon, right + horizon),
    [
      #text(fill: rgb("#f87171"), weight: "bold", size: 8pt)[PSAJ MATEMATIKA × PPLG • SMK TELKOM PURWOKERTO • TA 2026/2027] \
      #v(0.2em)
      #text(fill: white, weight: "bold", size: 13pt)[
        NASKAH LENGKAP INFOGRAFIS RISET DATA SCIENCE
      ] \
      #text(fill: rgb("#38bdf8"), weight: "bold", size: 10pt)[
        Peramalan Tingkat Pengangguran Terbuka (TPT) Indonesia Target 2026
      ]
      #v(0.2em)
      #text(fill: rgb("#94a3b8"), size: 8pt)[
        Penulis: Adzkya G. F. A., Aufa Rakha A. A., Andrean Rayhan W. • Pembimbing: Reza Auditore, S.Kom.
      ]
    ],
    [
      #image("assets/telkom-logo.png", width: 1.6cm)
    ]
  )
]

#v(0.5em)

#rect(fill: rgb("#eff6ff"), stroke: 0.5pt + rgb("#bfdbfe"), radius: 4pt, inset: 6pt, width: 100%)[
  #set text(size: 8.5pt)
  #text(weight: "bold", fill: rgb("#1d4ed8"))[ℹ️ METADATA DOKUMEN:]
  #text(fill: rgb("#1e3a8a"))[ Panjang Naskah = *677 Kata* (Memenuhi kriteria tugas rentang *600–800 kata*). Mengintegrasikan konsep Statistika Terapan OLS dengan pustaka Scikit-Learn Python.]
]

#v(0.6em)

= 1. Pendahuluan dan Konteks Makroekonomi

Tingkat Pengangguran Terbuka (TPT) merupakan salah satu indikator vital dalam mengevaluasi dinamika pasar tenaga kerja, penyerapan angkatan kerja, dan stabilitas perekonomian nasional. Badan Pusat Statistik (BPS) merilis data ketenagakerjaan ini secara periodik melalui Survei Angkatan Kerja Nasional (Sakernas). Dalam kurun waktu pascapandemi COVID-19 (2020 hingga 2025), pergerakan angka pengangguran di Indonesia menunjukkan karakteristik pemulihan yang sangat teratur. Pada Agustus 2020, disrupsi ekonomi akibat pembatasan mobilitas memicu lonjakan TPT hingga menyentuh 7,07%. Namun, seiring berjalannya program pemulihan ekonomi nasional dan normalisasi aktivitas industri, angka pengangguran mengalami kontraksi konsisten setiap tahun: 6,49% pada 2021, 5,86% pada 2022, 5,32% pada 2023, 4,91% pada 2024, hingga mencapai level 4,85% pada Agustus 2025. Terjadinya penurunan kumulatif sebesar 2,22% poin dalam enam tahun pengamatan membuktikan adanya inersia struktural yang kuat. Guna merumuskan perencanaan ketenagakerjaan jangka pendek, khususnya horizon target tahun 2026, peramalan kuantitatif berbasis model regresi linier sederhana menjadi pendekatan yang objektif dan terukur.

#v(0.4em)

= 2. Karakteristik Data dan Formulasi Matematis Model

Dataset pengamatan dibatasi secara rigor pada runtun waktu nasional Sakernas edisi Agustus periode 2020 sampai 2025 dengan enam titik observasi historis ($n = 6$). Analisis ini mengadopsi model deterministik univariat dengan waktu kalender sebagai variabel prediktor bebas ($X$) dan angka TPT nasional sebagai variabel respon terikat ($Y$) tanpa rekayasa fitur artifisial. Pendekatan matematika yang digunakan adalah metode kuadrat terkecil murni atau _Ordinary Least Squares_ (OLS) untuk membentuk persamaan garis tren linier:
$ Y = a + b X $

Melalui formula analitik OLS, nilai kemiringan lereng ($b$) dan konstanta intersep ($a$) dihitung secara langsung. Parameter kemiringan lereng ($b$) diperoleh sebesar $-0{,}468000$, yang mengindikasikan bahwa setiap pertambahan satu tahun kalender berkontribusi terhadap penurunan rata-rata TPT nasional sebesar 0,468% poin. Sementara itu, nilai intersep ($a$) terhitung sebesar 952,280000. Dengan demikian, formulasi matematis garis tren peramalan dinyatakan secara definitif sebagai:
$ Y = "952,28" - "0,468" X $

#v(0.4em)

= 3. Pembuktian Perhitungan Proyeksi Target Tahun 2026

Untuk memperoleh taksiran angka pengangguran pada target horizon tahun 2026, nilai skalar prediktor $X = 2026$ disubstitusikan ke dalam persamaan model regresi:
$ Y(2026) = "952,28" - ("0,468" times 2026) $
$ Y(2026) = "952,28" - "948,168" = "4,112%" $

Melalui pembulatan standar dua desimal, proyeksi Tingkat Pengangguran Terbuka Indonesia untuk target tahun 2026 adalah *4,11%*. Angka proyeksi 4,11% ini merepresentasikan kelanjutan penurunan sebesar 0,74% poin dari posisi realisasi Agustus 2025. Proyeksi ini mencerminkan skenario acuan dasar (_baseline_) apabila stabilitas makroekonomi dan kapasitas penyerapan tenaga kerja nasional bergerak konsisten mengikuti pola historis lima tahun terakhir.

#v(0.4em)

= 4. Evaluasi Diagnostik dan Pengujian Presisi Model

Kualitas kelayakan matematis model regresi linier diuji menggunakan parameter diagnostik standar ilmu data:
+ *Koefisien Korelasi Pearson ($r$):* Menghasilkan nilai $r = "-0,9786"$. Nilai ini menunjukkan korelasi linier negatif yang sangat kuat antara pertambahan waktu dan penurunan pengangguran terbuka.
+ *Koefisien Determinasi ($R^2$):* Diperoleh nilai $R^2 = "0,9576"$. Hal ini membuktikan bahwa sebesar 95,76% variansi dinamika perubahan TPT nasional berhasil dijelaskan oleh model regresi linier, sedangkan sisa 4,24% dipengaruhi oleh faktor eksternal lainnya.
+ *Mean Absolute Error (MAE):* Nilai MAE tercatat sebesar 0,1527% poin, menunjukkan rata-rata simpangan absolut tebakan model terhadap data riil sangat tipis.
+ *Mean Absolute Percentage Error (MAPE):* Nilai MAPE tercatat sebesar 2,8142%. Berdasarkan konvensi pengukuran, MAPE di bawah 5% mengonfirmasi bahwa akurasi model tergolong dalam kategori sangat presisi.
+ *Root Mean Squared Error (RMSE):* Bernilai 0,1682% poin, menandakan tidak adanya galat ekstrem (_outlier_).
+ *Karakteristik Residual:* Rata-rata galat residu model bernilai tepat 0,0000%, membuktikan sifat estimator OLS yang bebas bias sistematis (_unbiased estimator_).

#v(0.4em)

= 5. Implementasi Komputasi dan Implikasi Kebijakan Vokasi

Seluruh alur komputasi diintegrasikan menggunakan bahasa pemrograman Python 3 dengan pustaka ramah pemula Scikit-Learn (modul `LinearRegression`) dan Pandas. Alur kerja dibangun secara ringkas dalam empat tahap modular: pembacaan data, partisi vektor fitur, kalkulasi parameter OLS, serta inferensi target 2026. Temuan proyeksi 4,11% membawa implikasi strategis bagi ekosistem pendidikan vokasi, khususnya SMK Telkom Purwokerto. Penurunan angka pengangguran nasional menuntut kesiapan lulusan vokasi kejuruan Pengembangan Perangkat Lunak dan Gim (PPLG) untuk mengisi kebutuhan talenta digital berdaya saing tinggi. Kemitraan strategis melalui skema _link-and-match_ dengan industri teknologi mutlak diperkuat agar momentum positif penyerapan tenaga kerja terus terjaga.
