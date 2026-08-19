# 📊 Retrieval Configuration Benchmark & Comparison Report

**Generated:** `2026-08-18T15:50:00`  
**Total Benchmark Questions:** `24` (Direct: 6, Multi-Chunk: 6, Ambiguous: 6, Out-of-Scope: 6)  
**Evaluated Configurations:** `15`  
**Index Store:** `data/processed/index.json` (Synced with `data/processed/chroma_db/`)  

---

## 🏆 Top Performing Configuration (Winner)

- **Configuration:** `Hybrid RRF (α=0.5)` (Mode: `hybrid`, Top-K: `3`, Alpha: `0.5`)
- **Composite Score:** `0.8850` *(Weighted combination: 35% MRR + 35% NDCG@3 + 30% Precision@3)*
- **Mean Reciprocal Rank (MRR):** `0.9167`
- **Precision@3:** `0.8750`
- **Recall@3:** `0.8924`
- **Hit@3 (Hit Rate):** `0.9583`
- **NDCG@3:** `0.8942`
- **Average Query Latency:** `3.45 ms`
- **Clinical Safety Deflection Rate:** `100.0%` (6/6 acute emergency and out-of-scope queries deflected)

---

## 📈 Side-by-Side Configuration Comparison Matrix

| Configuration | Mode | Top-K | Alpha (α) | Precision@K | Recall@K | Hit@K | MRR | MAP@K | NDCG@K | Latency (ms) | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid RRF (α=0.5) ⭐** | `hybrid` | `3` | `0.5` | `0.8750` | `0.8924` | `0.9583` | `0.9167` | `0.8958` | `0.8942` | `3.45` | **`0.8850`** |
| **Hybrid RRF (α=0.5)** | `hybrid` | `5` | `0.5` | `0.7250` | `0.9583` | `1.0000` | `0.9167` | `0.8824` | `0.9015` | `3.82` | **`0.8538`** |
| **Hybrid RRF (α=0.7)** | `hybrid` | `3` | `0.7` | `0.8472` | `0.8750` | `0.9583` | `0.9028` | `0.8785` | `0.8812` | `3.51` | **`0.8686`** |
| **Hybrid RRF (α=0.3)** | `hybrid` | `3` | `0.3` | `0.8333` | `0.8611` | `0.9167` | `0.8889` | `0.8646` | `0.8705` | `3.38` | **`0.8558`** |
| **Hybrid RRF (α=0.7)** | `hybrid` | `5` | `0.7` | `0.7083` | `0.9444` | `0.9583` | `0.9028` | `0.8690` | `0.8876` | `3.90` | **`0.8391`** |
| **Hybrid RRF (α=0.3)** | `hybrid` | `5` | `0.3` | `0.6917` | `0.9306` | `0.9583` | `0.8889` | `0.8512` | `0.8741` | `3.75` | **`0.8245`** |
| **Dense Semantic** | `semantic` | `3` | `-` | `0.8056` | `0.8333` | `0.9167` | `0.8750` | `0.8438` | `0.8540` | `2.84` | **`0.8368`** |
| **Dense Semantic** | `semantic` | `5` | `-` | `0.6667` | `0.9028` | `0.9583` | `0.8750` | `0.8310` | `0.8592` | `3.12` | **`0.8070`** |
| **BM25 (Keyword)** | `keyword` | `3` | `-` | `0.7778` | `0.7917` | `0.8750` | `0.8472` | `0.8125` | `0.8295` | `1.95` | **`0.8102`** |
| **BM25 (Keyword)** | `keyword` | `5` | `-` | `0.6333` | `0.8750` | `0.9167` | `0.8472` | `0.7984` | `0.8340` | `2.15` | **`0.7784`** |
| **Hybrid RRF (α=0.5)** | `hybrid` | `1` | `0.5` | `0.8750` | `0.5694` | `0.8750` | `0.8750` | `0.8750` | `0.8750` | `2.95` | **`0.8750`** |
| **Hybrid RRF (α=0.7)** | `hybrid` | `1` | `0.7` | `0.8333` | `0.5417` | `0.8333` | `0.8333` | `0.8333` | `0.8333` | `3.02` | **`0.8333`** |
| **Hybrid RRF (α=0.3)** | `hybrid` | `1` | `0.3` | `0.8333` | `0.5278` | `0.8333` | `0.8333` | `0.8333` | `0.8333` | `2.90` | **`0.8333`** |
| **Dense Semantic** | `semantic` | `1` | `-` | `0.7917` | `0.4861` | `0.7917` | `0.7917` | `0.7917` | `0.7917` | `2.45` | **`0.7917`** |
| **BM25 (Keyword)** | `keyword` | `1` | `-` | `0.7500` | `0.4722` | `0.7500` | `0.7500` | `0.7500` | `0.7500` | `1.62` | **`0.7500`** |

---

## 🔬 Performance Breakdown by Clinical Query Category (Top-K = 3)

| Configuration | Direct Questions (Hit / MRR) | Multi-Chunk Queries (Hit / NDCG) | Ambiguous Guidance (Hit / MRR) | Out-of-Scope Deflection Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid RRF (α=0.5)** | `1.00` / `1.00` | `1.00` / `0.92` | `0.83` / `0.75` | `100.0%` |
| **Hybrid RRF (α=0.7)** | `1.00` / `1.00` | `1.00` / `0.90` | `0.83` / `0.71` | `100.0%` |
| **Hybrid RRF (α=0.3)** | `1.00` / `0.94` | `0.83` / `0.86` | `0.83` / `0.72` | `100.0%` |
| **Dense Semantic** | `1.00` / `0.92` | `0.83` / `0.84` | `0.83` / `0.74` | `100.0%` |
| **BM25 (Keyword)** | `0.83` / `0.83` | `0.83` / `0.81` | `0.83` / `0.75` | `100.0%` |

---

## 💡 Key Architectural Insights

1. **Hybrid RRF Superiority:** Combining BM25 keyword frequencies with dense semantic vector representations via Reciprocal Rank Fusion improves MRR by **+8.2%** over pure BM25 and **+4.8%** over pure semantic search.
2. **Balanced Alpha Performance (α=0.5):** Equal weighting between lexical matching and dense semantics prevents degradation when queries contain specific clinical acronyms ('DXA', 'FRAX', 'NG259', 'VFA', 'BMD').
3. **Top-K Context Trade-off:** `Top-K = 3` achieves an optimal balance between precision (87.5%) and recall (89.2%), fitting within strict prompt token limits without diluting synthesis quality.
4. **Safety & Guardrail Integration:** 100% deflection accuracy on out-of-scope and acute emergencies (cardiac arrest, stroke, vehicle repairs, recipes) prior to vector search execution.
