// ============================================================
// INFOGRAFIS PROYEK PSAJ MATEMATIKA × PPLG
// SMK TELKOM PURWOKERTO - TAHUN AJARAN 2026/2027
// ============================================================

#set page(
  paper: "a4",
  margin: (x: 1.15cm, top: 0.95cm, bottom: 0.95cm),
  header: none,
  footer: none,
)

#set text(
  font: ("Liberation Sans", "DejaVu Sans", "Helvetica"),
  size: 8.5pt,
  fill: rgb("#0f172a"),
  lang: "id",
)

#set par(justify: false, leading: 0.52em)

// --- Palette Warna ---
#let c-primary    = rgb("#1e3a8a") // Deep Navy
#let c-accent     = rgb("#2563eb") // Royal Blue
#let c-sky        = rgb("#0284c7") // Sky Blue
#let c-crimson    = rgb("#dc2626") // Telkom Red
#let c-emerald    = rgb("#16a34a") // Emerald Green
#let c-amber      = rgb("#d97706") // Amber
#let c-bg-card    = rgb("#ffffff") // White Card
#let c-bg-soft    = rgb("#f8fafc") // Light Slate
#let c-border     = rgb("#cbd5e1") // Border Grey
#let c-muted      = rgb("#64748b") // Muted Text
#let c-dark       = rgb("#0f172a") // Deep Dark

// --- Helper Functions ---
#let badge(teks, bg: rgb("#eff6ff"), fg: rgb("#1d4ed8")) = {
  box(
    fill: bg,
    radius: 3pt,
    inset: (x: 6pt, y: 2.5pt),
    baseline: 0%,
    text(fill: fg, weight: "bold", size: 6.8pt)[#teks]
  )
}

#let card(isi, bg: c-bg-card, border: c-border, radius: 5pt, inset: 7pt) = {
  rect(
    width: 100%,
    fill: bg,
    stroke: 0.6pt + border,
    radius: radius,
    inset: inset,
    isi
  )
}

#let section-title(nomor, judul, tag: "") = {
  v(0.35em)
  grid(
    columns: (auto, 1fr, auto),
    gutter: 6pt,
    align: (left + horizon, left + horizon, right + horizon),
    box(
      fill: c-primary,
      radius: 3pt,
      inset: (x: 5.5pt, y: 2.5pt),
      text(fill: white, weight: "bold", size: 8pt)[#nomor]
    ),
    text(fill: c-primary, weight: "bold", size: 10.5pt)[#judul],
    if tag != "" { badge(tag, bg: rgb("#f1f5f9"), fg: c-muted) } else { [] }
  )
  v(0.12em)
  line(length: 100%, stroke: 0.7pt + rgb("#cbd5e1"))
  v(0.28em)
}

// ============================================================
// HALAMAN 1: TREN DATA & PROYEKSI TPT 2026
// ============================================================

// ─── Header Banner ──────────────────────────────────────────
#rect(
  width: 100%,
  fill: rgb("#0f172a"),
  radius: 6pt,
  inset: (x: 12pt, y: 9pt),
)[
  #grid(
    columns: (1fr, auto),
    gutter: 10pt,
    align: (left + horizon, right + horizon),
    [
      #grid(
        columns: (auto, auto, auto),
        gutter: 5pt,
        badge("PSAJ MATEMATIKA × PPLG", bg: rgb("#b91c1c"), fg: white),
        badge("SMK TELKOM PURWOKERTO", bg: rgb("#1e293b"), fg: rgb("#cbd5e1")),
        badge("TAHUN AJARAN 2026/2027", bg: rgb("#1e293b"), fg: rgb("#94a3b8")),
      )
      #v(0.35em)
      #text(fill: white, weight: "bold", size: 13.5pt)[
        INFOGRAFIS PERAMALAN TINGKAT PENGANGGURAN TERBUKA (TPT)
      ] \
      #text(fill: rgb("#38bdf8"), weight: "bold", size: 11pt)[
        PROYEKSI TARGET TAHUN 2026 BERBASIS REGRESI LINIER SEDERHANA (OLS)
      ]
      #v(0.25em)
      #text(fill: rgb("#94a3b8"), size: 7.5pt)[
        *Tim Pengembang:* Adzkya G. F. A. | Aufa Rakha A. A. | Andrean Rayhan W.  •  *Guru Pembimbing:* Reza Auditore, S.Kom.
      ]
    ],
    [
      #image("assets/telkom-logo.png", width: 1.8cm)
    ]
  )
]

#v(0.35em)

// ─── 4 Key Highlight Cards ──────────────────────────────────
#grid(
  columns: (1.25fr, 1fr, 1fr, 1fr),
  gutter: 7pt,
  [
    #rect(
      width: 100%,
      fill: rgb("#fef2f2"),
      stroke: 1.2pt + c-crimson,
      radius: 6pt,
      inset: (x: 8pt, y: 7pt),
    )[
      #text(fill: c-crimson, weight: "bold", size: 7pt)[🎯 PROYEKSI TARGET 2026] \
      #v(0.08em)
      #text(fill: c-crimson, weight: "bold", size: 20pt)[4,11%] \
      #text(fill: rgb("#991b1b"), size: 7pt)[
        *Turun -0,74% poin* dari posisi Agustus 2025 (4,85%)
      ]
    ]
  ],
  [
    #card(
      bg: rgb("#eff6ff"),
      border: c-accent,
      [
        #text(fill: c-accent, weight: "bold", size: 7pt)[📉 LAJU TREN TAHUNAN] \
        #v(0.08em)
        #text(fill: c-primary, weight: "bold", size: 17pt)[-0,468%] \
        #text(fill: c-muted, size: 7pt)[
          Gradien lereng ($b$) penurunan TPT per tahun
        ]
      ]
    )
  ],
  [
    #card(
      bg: rgb("#f0fdf4"),
      border: c-emerald,
      [
        #text(fill: c-emerald, weight: "bold", size: 7pt)[📊 DETERMINASI MODEL ($R^2$)] \
        #v(0.08em)
        #text(fill: c-emerald, weight: "bold", size: 17pt)[95,76%] \
        #text(fill: c-muted, size: 7pt)[
          Proporsi variansi yang sukses dijelaskan OLS
        ]
      ]
    )
  ],
  [
    #card(
      bg: rgb("#fefce8"),
      border: c-amber,
      [
        #text(fill: c-amber, weight: "bold", size: 7pt)[⚡ TINGKAT GALAT (MAPE)] \
        #v(0.08em)
        #text(fill: rgb("#b45309"), weight: "bold", size: 17pt)[2,81%] \
        #text(fill: c-muted, size: 7pt)[
          Kategori *Sangat Presisi* (Standar MAPE < 5%)
        ]
      ]
    )
  ],
)

#v(0.15em)

// ─── Section 1: Data Historis & Visualisasi Tren ────────────
#section-title("01", "Dinamika Data Historis Lengkap BPS Sakernas (2015–2025) & Tren OLS", tag: "Observasi Lengkap 11 Tahun")

#grid(
  columns: (1fr, 1.35fr),
  gutter: 8pt,
  [
    #card(
      inset: 4.5pt,
      [
        #text(weight: "bold", fill: c-primary, size: 8pt)[Tabel Runtun Waktu Lengkap BPS (Agustus)]
        #v(0.12em)
        #table(
          columns: (1fr, 1.15fr, 1.15fr, 1.7fr),
          stroke: 0.35pt + c-border,
          inset: 2.2pt,
          align: (center, center, center, left),
          fill: (col, row) => if row == 0 { rgb("#f1f5f9") } else if row == 6 { rgb("#fee2e2") } else if row == 11 { rgb("#f0fdf4") } else { none },
          table.header(
            strong[Tahun], strong[TPT (%)], strong[YoY], strong[Dinamika Pasar Kerja]
          ),
          [2015], [6,18%], [—], [Normalisasi Pasca-BBM],
          [2016], [5,61%], [-0,57%], [Penurunan Bertahap],
          [2017], [5,50%], [-0,11%], [Ekspansi Manufaktur],
          [2018], [5,34%], [-0,16%], [Kondisi Stabil Pra-Pandemi],
          [2019], [5,23%], [-0,11%], [Titik Terendah Pra-Pandemi],
          [2020], [*7,07%*], [*+1,84%*], [⚡ *Syok Lonjakan COVID-19*],
          [2021], [6,49%], [-0,58%], [Awal Pemulihan Industri],
          [2022], [5,86%], [-0,63%], [Rebound Mobilitas Warga],
          [2023], [5,32%], [-0,54%], [Normalisasi Lapangan Kerja],
          [2024], [4,91%], [-0,41%], [Ekspansi Sektor Formal],
          [2025], [*4,85%*], [-0,06%], [Stabilisasi Makroekonomi],
        )
        #v(0.12em)
        #rect(
          fill: rgb("#f8fafc"),
          stroke: 0.5pt + rgb("#cbd5e1"),
          radius: 3pt,
          inset: 3.5pt,
          width: 100%,
        )[
          #text(size: 6.5pt, fill: c-dark)[
            📌 *Pola Dinamika Data 11 Tahun:* \
            • *2015–2019:* Penurunan stabil pra-pandemi (6,18% $arrow$ 5,23%) \
            • *2020:* Syok eksternal COVID-19 meledak ke puncak 7,07% (+1,84%) \
            • *2020–2025:* Pemulihan linier (-2,22% poin) di-fit OLS ke *4,11% (2026)*
          ]
        ]
      ]
    )
  ],
  [
    #card(
      inset: 5pt,
      [
        #image("assets/infografis_tren_chart.png", width: 100%)
      ]
    )
  ]
)

#v(0.15em)

// ─── Section 2: Wawasan Kunci Ekstrapolasi ──────────────────
#section-title("02", "Peta Keterhubungan Lintas Dataset (Waktu x Spasial x Target 2026)", tag: "Integrasi Komprehensif")

#grid(
  columns: (1fr, 1.05fr, 0.95fr),
  gutter: 7pt,
  [
    #card(
      bg: rgb("#f8fafc"),
      border: c-border,
      inset: 5.5pt,
      [
        #text(weight: "bold", fill: c-primary, size: 7.8pt)[1. Deret Waktu (2015–2025)] \
        #v(0.12em)
        #text(size: 6.8pt, fill: rgb("#334155"))[
          Merekam 11 tahun historis BPS (dua gelombang: Feb & Agu). Menangkap dinamika pra-pandemi (5,23%), syok lonjakan 2020 (7,07%), hingga pemulihan bertahap ke 4,85% (Agu 2025).
        ]
      ]
    )
  ],
  [
    #card(
      bg: rgb("#eff6ff"),
      border: rgb("#93c5fd"),
      inset: 5.5pt,
      [
        #text(weight: "bold", fill: c-accent, size: 7.8pt)[2. Titik Temu 2025 (Dataset Y & X)] \
        #v(0.12em)
        #text(size: 6.8pt, fill: rgb("#1e3a8a"))[
          Tahun 2025 terhubung langsung ke *Dataset Y (38 Provinsi: 1,49% Bali s.d. 6,96% Papua; Jateng 4,66%)* dan *Dataset X (Pendidikan SMA+)* yang memetakan disparitas serapan tenaga kerja.
        ]
      ]
    )
  ],
  [
    #card(
      bg: rgb("#fef2f2"),
      border: rgb("#fca5a5"),
      inset: 5.5pt,
      [
        #text(weight: "bold", fill: c-crimson, size: 7.8pt)[3. Proyeksi Target (2026)] \
        #v(0.12em)
        #text(size: 6.8pt, fill: rgb("#7f1d1d"))[
          Model OLS pascapandemi ($Y = "952,28" - "0,468" X$) memproyeksikan TPT nasional Agustus 2026 ke *4,11%* (dan Feb 2026 ke *4,02%*), menembus rekor terendah dalam 12 tahun.
        ]
      ]
    )
  ],
)

#v(0.3em)
#align(center)[
  #text(fill: c-muted, size: 7pt)[Halaman 1 dari 2 • Infografis PSAJ SMK Telkom Purwokerto — Lanjut ke Metodologi & Diagnostik Validasi]
]

#pagebreak()

// ============================================================
// HALAMAN 2: BEDAH MATEMATIKA, DIAGNOSTIK & IMPLEMENTASI
// ============================================================

// ─── Header Mini Halaman 2 ──────────────────────────────────
#rect(
  width: 100%,
  fill: rgb("#1e293b"),
  radius: 5pt,
  inset: (x: 10pt, y: 6pt),
)[
  #grid(
    columns: (1fr, auto),
    gutter: 10pt,
    align: (left + horizon, right + horizon),
    [
      #text(fill: white, weight: "bold", size: 10.5pt)[
        METODOLOGI FORMULASI OLS, DIAGNOSTIK VALIDASI & ALUR PIPELINE
      ] \
      #text(fill: rgb("#38bdf8"), size: 7.5pt)[
        Pembuktian Matematis, Verifikasi Residu, dan Integrasi Perangkat Lunak Python (Scikit-Learn)
      ]
    ],
    [
      #badge("BAGIAN TEKNIS & KOMPUTASI", bg: rgb("#334155"), fg: white)
    ]
  )
]

#v(0.25em)

// ─── Section 3: Bedah Matematis OLS ─────────────────────────
#section-title("03", "Formulasi Matematis: Ordinary Least Squares (OLS)", tag: "Konsep Dasar Matematika")

#grid(
  columns: (1.2fr, 1fr),
  gutter: 8pt,
  [
    #card(
      inset: 7pt,
      [
        #text(weight: "bold", fill: c-primary, size: 8.5pt)[Perumusan Parameter Garis Tren Linier] \
        #v(0.2em)
        Persamaan regresi linier memodelkan hubungan fungsional:
        #align(center)[
          #rect(fill: rgb("#f8fafc"), stroke: 0.5pt + rgb("#cbd5e1"), radius: 3pt, inset: (x: 8pt, y: 4pt))[
            #text(size: 9.5pt, weight: "bold", fill: c-primary)[
              $ hat(Y) = a + b X $
            ]
          ]
        ]
        #v(0.1em)
        Estimasi parameter diperoleh dengan meminimalkan kuadrat simpangan vertikal:
        #grid(
          columns: (1fr, 1fr),
          gutter: 6pt,
          [
            #text(size: 7.2pt)[*Kemiringan Lereng ($b$):*]
            #align(center)[
              #text(size: 7.5pt)[
                $ b = (n sum X Y - sum X sum Y) / (n sum X^2 - (sum X)^2) $
              ]
            ]
            #align(center)[#badge("b = -0,468000", bg: rgb("#eff6ff"), fg: c-accent)]
          ],
          [
            #text(size: 7.2pt)[*Titik Potong ($a$):*]
            #align(center)[
              #text(size: 7.5pt)[
                $ a = overline(Y) - b overline(X) $
              ]
            ]
            #align(center)[#badge("a = 952,280000", bg: rgb("#f0fdf4"), fg: c-emerald)]
          ]
        )
      ]
    )
  ],
  [
    #card(
      bg: rgb("#fef2f2"),
      border: rgb("#fca5a5"),
      inset: 7pt,
      [
        #text(weight: "bold", fill: c-crimson, size: 8.5pt)[Perhitungan Eksplisit Proyeksi 2026] \
        #v(0.2em)
        #text(size: 7.5pt)[
          Substitusi nilai prediktor $X = 2026$ ke dalam persamaan model:
        ]
        #v(0.25em)
        #rect(fill: white, stroke: 0.5pt + rgb("#f87171"), radius: 3pt, inset: 6pt, width: 100%)[
          #text(size: 8pt)[
            $ hat(Y)_(2026) = "952,28" - ("0,468" times 2026) $ \
            $ hat(Y)_(2026) = "952,28" - "948,168" $ \
            $ bold(hat(Y)_(2026) = "4,112%" approx "4,11%") $
          ]
        ]
        #v(0.2em)
        #text(size: 7pt, fill: rgb("#991b1b"))[
          *Interpretasi:* Setiap bertambah 1 tahun, TPT nasional berkurang rata-rata sebesar *0,468% poin*.
        ]
      ]
    )
  ]
)

#v(0.18em)

// ─── Section 4: Diagnostik Evaluasi & Residual ──────────────
#section-title("04", "Validasi Model & Analisis Residu (In-Sample)", tag: "Diagnostik Presisi")

#grid(
  columns: (1.3fr, 1fr),
  gutter: 8pt,
  [
    #card(
      inset: 5pt,
      [
        #image("assets/infografis_residual_chart.png", width: 100%)
      ]
    )
  ],
  [
    #card(
      inset: 6pt,
      [
        #text(weight: "bold", fill: c-primary, size: 8.5pt)[Rangkuman Metrik Validasi]
        #v(0.25em)
        #grid(
          columns: (1fr, 1fr),
          gutter: 5pt,
          [
            #rect(fill: rgb("#f8fafc"), stroke: 0.4pt + c-border, radius: 3pt, inset: 4pt, width: 100%)[
              #text(size: 6.5pt, fill: c-muted)[KOEF. DETERMINASI] \
              #text(size: 9.5pt, weight: "bold", fill: c-primary)[$R^2 = "0,9576"$] \
              #text(size: 6.5pt, fill: c-emerald)[Sangat Kuat (95,8%)]
            ]
          ],
          [
            #rect(fill: rgb("#f8fafc"), stroke: 0.4pt + c-border, radius: 3pt, inset: 4pt, width: 100%)[
              #text(size: 6.5pt, fill: c-muted)[KORELASI PEARSON] \
              #text(size: 9.5pt, weight: "bold", fill: c-primary)[$r = "-0,9786"$] \
              #text(size: 6.5pt, fill: c-accent)[Linier Negatif Kuat]
            ]
          ],
          [
            #rect(fill: rgb("#f8fafc"), stroke: 0.4pt + c-border, radius: 3pt, inset: 4pt, width: 100%)[
              #text(size: 6.5pt, fill: c-muted)[DEV. ABSOLUT (MAE)] \
              #text(size: 9.5pt, weight: "bold", fill: c-primary)[0,1527%] \
              #text(size: 6.5pt, fill: c-emerald)[Simpangan Sangat Tipis]
            ]
          ],
          [
            #rect(fill: rgb("#f8fafc"), stroke: 0.4pt + c-border, radius: 3pt, inset: 4pt, width: 100%)[
              #text(size: 6.5pt, fill: c-muted)[AKAR KUADRAT (RMSE)] \
              #text(size: 9.5pt, weight: "bold", fill: c-primary)[0,1682%] \
              #text(size: 6.5pt, fill: c-emerald)[Bebas Outlier Ekstrem]
            ]
          ],
        )
        #text(size: 6.8pt, fill: rgb("#334155"))[
          *Sifat Residu OLS:* Rata-rata residual = *0,0000%* (bebas bias sistematis). Deviasi simpangan maksimum hanya *+0,186%* (2025) dan *-0,280%* (2023).
        ]
      ]
    )
  ]
)

#v(0.18em)

// ─── Section 5: Pipeline Komputasi & Implikasi ──────────────
#section-title("05", "Arsitektur Komputasi Python & Rekomendasi Kebijakan", tag: "Pipeline & Dampak Vokasi")

#grid(
  columns: (1.2fr, 1fr),
  gutter: 8pt,
  [
    #card(
      inset: 6pt,
      [
        #text(weight: "bold", fill: c-primary, size: 8.5pt)[Arsitektur Pipeline Scikit-Learn] \
        #v(0.2em)
        #grid(
          columns: (auto, 1fr),
          gutter: 5pt,
          align: (center + horizon, left + horizon),
          badge("1", bg: c-primary, fg: white),
          [#text(size: 7.2pt)[*Ingesti Data:* `pd.DataFrame()` membaca 6 observasi BPS.]],
          badge("2", bg: c-accent, fg: white),
          [#text(size: 7.2pt)[*Partisi Fitur:* Matriks fitur $X$ `[['Tahun']]` dan target $y$ `['TPT']`.]],
          badge("3", bg: c-emerald, fg: white),
          [#text(size: 7.2pt)[*Fitting OLS:* `LinearRegression().fit(X, y)` estimasi $a$ & $b$.]],
          badge("4", bg: c-crimson, fg: white),
          [#text(size: 7.2pt)[*Inferensi:* `model.predict([[2026]])` ekstrapolasi horizon 2026.]],
        )
        #v(0.25em)
        #rect(fill: rgb("#0f172a"), stroke: 0.5pt + rgb("#334155"), radius: 3pt, inset: (x: 6pt, y: 4pt), width: 100%)[
          #set text(size: 6.8pt, font: ("Liberation Mono", "DejaVu Sans Mono"))
          #text(fill: rgb("#94a3b8"))[\# Implementasi Ringkas Model Scikit-Learn] \
          #text(fill: rgb("#38bdf8"))[model = LinearRegression().fit(X, y)] \
          #text(fill: rgb("#4ade80"))[pred_2026 = model.predict([[2026]])  \# Hasil: 4.112%]
        ]
      ]
    )
  ],
  [
    #card(
      bg: rgb("#f8fafc"),
      border: c-border,
      inset: 6pt,
      [
        #text(weight: "bold", fill: c-primary, size: 8.5pt)[Rekomendasi Kebijakan Vokasi] \
        #v(0.2em)
        #text(size: 7.2pt, fill: rgb("#334155"))[
          + *Relevansi SMK Vokasi:* Penurunan TPT menuju 4,11% menuntut penyerapan lulusan SMK di sektor teknologi digital dan industri bernilai tambah tinggi.
          + *Skema Link-and-Match:* Mendorong sertifikasi kompetensi industri dan magang industri terstruktur untuk menekan pengangguran usia muda.
          + *Monitoring Triwulan:* Evaluasi realisasi Sakernas 2026 terhadap angka baseline model (4,11%) sebagai indikator peringatan dini kebijakan.
        ]
      ]
    )
  ]
)

#v(0.3em)

// ─── Footer Resmi ───────────────────────────────────────────
#rect(
  width: 100%,
  fill: rgb("#f1f5f9"),
  stroke: 0.5pt + rgb("#cbd5e1"),
  radius: 4pt,
  inset: (x: 8pt, y: 5pt),
)[
  #grid(
    columns: (1fr, auto),
    gutter: 10pt,
    align: (left + horizon, right + horizon),
    [
      #text(size: 6.8pt, fill: c-muted)[
        *Sumber Data:* Badan Pusat Statistik (BPS) RI, Survei Angkatan Kerja Nasional (Sakernas) Edisi Agustus 2020–2025. \
        *Dokumen Terkait:* Proposal Proyek PSAJ Matematika SMK Telkom Purwokerto (NIS: 541241008, 541241028, 54124018).
      ]
    ],
    [
      #text(size: 6.8pt, weight: "bold", fill: c-primary)[
        LaborCast © 2026 SMK Telkom Purwokerto
      ]
    ]
  )
]
