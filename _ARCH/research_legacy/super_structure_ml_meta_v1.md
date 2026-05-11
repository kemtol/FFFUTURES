# Plan: Super Structure ML Meta-Layer (v2)

## Objective
- Meningkatkan kualitas keputusan Super Structure dengan ML, tanpa merusak versi live yang sudah stabil.
- Fokus utamanya bukan mengganti strategi, tapi menambah lapisan probabilitas untuk menjawab: *trade core ini layak diambil atau tidak*.

## Prinsip Dasar
- **Core Integrity:** Strategy tetap utuh (SuperTrend + DEMA + ADX + CCI).
- **Meta-Layer:** ML hanya bertugas sebagai probability filter di atas signal core.
- **Isolation:** Live execution tidak disentuh selama fase riset.
- **Versioning:** Semua eksperimen (datamart & models) harus versioned.

## Struktur Kerja & Namespace
- **Research Namespace:** `pipeline/super_structure_ml/`
- **Datamart:** `data/Level_2_Datamart/super_structure_ml/` (Versioned)
- **Model Artifacts:** `model/SUPER_STRUCTURE/meta_v1/` (Versioned)

## Pipeline Roadmap

### 1. Dataset Generation (The "Truth")
- Extract trades dari `parity_super_structure.py` logic.
- Align antara Signal Entry (Live) vs UI Entry (Theoretical).
- Labeling berdasarkan hasil realita (PnL, Drawdown) sesuai target Topstep.

### 2. Feature Engineering (Entry-Only Context)
- Menggunakan data yang tersedia tepat saat signal muncul.
- Market context (Volatility, Trend Strength, Session Time).
- Indicator states (CCI level, ADX slope, DEMA distance).

### 3. Model Training & Validation
- Train meta-model (LightGBM/XGBoost) sebagai classifier.
- Target: Probabilitas trade mencapai TP sebelum SL/MLL.
- Validation: Walk-forward analysis untuk menghindari overfitting pada regime market tertentu.

### 4. Simulation & Policy
- Cari threshold probabilitas terbaik (e.g., "Only take trades with Prob > 0.6").
- Bandingkan baseline (all signals) vs ML-filtered signals.
- Metric Utama: Expectancy, Max Drawdown, dan Pass Rate simulasi Topstep.

## Expectations
- Versi live aman dan tidak berubah.
- Baseline data yang rapi untuk eksperimen berkelanjutan.
- Menghasilkan threshold filter yang objektif dan sistematis.

## Non-Goals
- Tidak mengubah logika entry `super_structure.py`.
- Tidak mengganti flow eksekusi ke Topstep.
- Tidak mengejar kompleksitas model di awal; prioritaskan validitas dan usability.
