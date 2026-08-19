# 🔬 Multi-Dimensional Retrieval Experimentation Matrix & Benchmark Report

**Generated:** `2026-08-18T15:52:00`  
**Total Hyperparameter Configurations Evaluated:** `96`  
**Total Benchmark Questions:** `24` (Direct: 6, Multi-Chunk: 6, Ambiguous: 6, Out-of-Scope: 6)  
**Evaluated Grid Space:**  
- **Chunk Sizes (Tokens):** `[128, 256, 400, 512]`  
- **Chunk Overlaps (Tokens):** `[0, 20, 50, 100]`  
- **Embedding Models:** `['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']`  
- **Search Modes:** `['keyword', 'semantic', 'hybrid (α=0.3, 0.5, 0.7)']`  
- **Top-K Retrieval Depths:** `[1, 3, 5, 10]`  

---

## 🏆 Global Winner: Best Performing Retrieval Configuration

- **Configuration ID:** `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.5`
- **Embedding Model:** `all-MiniLM-L6-v2`
- **Chunk Size:** `400 tokens`
- **Chunk Overlap:** `50 tokens`
- **Search Type:** `HYBRID (RRF, α=0.5)`

### Rationale for Winner Selection:
The best configuration is selected using a multi-metric optimization objective:
$$\text{Composite Score} = 0.35 \cdot \text{Recall@3} + 0.35 \cdot \text{MRR} + 0.20 \cdot \text{Precision@3} + 0.10 \cdot \text{Recall@5} - \text{Latency Penalty}$$

1. **High Recall Without Dilution:** Captures **89.24%** of relevant guideline evidence at $K=3$ and **95.83%** at $K=5$.
2. **Top-Rank Precision (MRR: 0.9167):** Consistently places the primary guideline answer in rank 1.
3. **Sub-4ms Latency:** Highly responsive retrieval ($3.45\text{ ms}$) without GPU or remote API dependencies.
4. **Acronym & Semantic Fusion:** Fuses lexical exact matching ('FRAX', 'DXA', 'NG259') with dense semantic vector scoring.

### Winner Metrics Summary:
- **Recall@1:** `0.5694` | **Recall@3:** `0.8924` | **Recall@5:** `0.9583` | **Recall@10:** `0.9850`
- **Mean Reciprocal Rank (MRR):** `0.9167`
- **Precision@1:** `0.8750` | **Precision@3:** `0.8750` | **Precision@5:** `0.7250` | **Precision@10:** `0.4750`
- **Average Similarity:** `0.8245` | **Top-1 Similarity:** `0.8870`
- **Average Retrieval Latency:** `3.45 ms`
- **Composite Score:** **`0.8850`**

---

## 📊 Side-by-Side Configuration Comparison Matrix (Sorted by Performance)

| Rank | Configuration ID | Chunk (Tokens) | Overlap | Embedding Model | Search Mode | α | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Prec@3 | Avg Sim | Latency | Composite Score |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.5 ⭐` | `400` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `0.569` | `0.892` | `0.958` | `0.985` | `0.917` | `0.875` | `0.825` | `3.5ms` | **`0.8850`** |
| **2** | `sz256_ov50_bge-small-en-v1.5_hybrid_a0.5` | `256` | `50` | `bge-small-en-v1.5` | `hybrid` | `0.5` | `0.556` | `0.882` | `0.958` | `0.985` | `0.910` | `0.861` | `0.831` | `3.6ms` | **`0.8774`** |
| **3** | `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.7` | `400` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.7` | `0.542` | `0.875` | `0.944` | `0.975` | `0.903` | `0.847` | `0.819` | `3.5ms` | **`0.8686`** |
| **4** | `sz400_ov100_all-MiniLM-L6-v2_hybrid_a0.5` | `400` | `100` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `0.542` | `0.889` | `0.958` | `0.985` | `0.896` | `0.847` | `0.821` | `3.7ms` | **`0.8651`** |
| **5** | `sz256_ov50_all-MiniLM-L6-v2_hybrid_a0.5` | `256` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `0.528` | `0.868` | `0.944` | `0.975` | `0.896` | `0.847` | `0.815` | `3.4ms` | **`0.8643`** |
| **6** | `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.3` | `400` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.3` | `0.528` | `0.861` | `0.931` | `0.965` | `0.889` | `0.833` | `0.805` | `3.4ms` | **`0.8558`** |
| **7** | `sz512_ov50_all-MiniLM-L6-v2_hybrid_a0.5` | `512` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `0.514` | `0.847` | `0.917` | `0.955` | `0.882` | `0.819` | `0.798` | `3.6ms` | **`0.8471`** |
| **8** | `sz400_ov50_bge-small-en-v1.5_semantic` | `400` | `50` | `bge-small-en-v1.5` | `semantic` | `-` | `0.500` | `0.840` | `0.917` | `0.955` | `0.882` | `0.819` | `0.834` | `2.9ms` | **`0.8459`** |
| **9** | `sz400_ov50_all-MiniLM-L6-v2_semantic` | `400` | `50` | `all-MiniLM-L6-v2` | `semantic` | `-` | `0.486` | `0.833` | `0.903` | `0.950` | `0.875` | `0.806` | `0.812` | `2.8ms` | **`0.8368`** |
| **10** | `sz400_ov50_all-MiniLM-L6-v2_keyword` | `400` | `50` | `all-MiniLM-L6-v2` | `keyword` | `-` | `0.472` | `0.792` | `0.875` | `0.925` | `0.847` | `0.778` | `0.750` | `2.0ms` | **`0.8102`** |
| **11** | `sz128_ov20_all-MiniLM-L6-v2_hybrid_a0.5` | `128` | `20` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `0.444` | `0.778` | `0.861` | `0.910` | `0.833` | `0.764` | `0.775` | `3.1ms` | **`0.7961`** |

---

## 📈 Key Findings & Parameter Sensitivity

1. **Chunk Sizing Trade-Offs:**
   - **128 tokens**: Suffers from contextual truncation; splits diagnostic criteria across chunk boundaries.
   - **400 tokens**: Achieves the highest MRR (`0.9167`) and Recall@3 (`0.8924`), providing sufficient sentence context for clinical recommendations without semantic dilution.
   - **512 tokens**: Shows slight precision degradation due to extraneous non-relevant guideline text.

2. **Overlap Impact:**
   - A **50-token overlap** consistently boosts Recall@3 by **+8.5%** over 0 overlap by preserving boundary criteria.

3. **Search Type Superiority:**
   - **Hybrid RRF (α=0.5)** outperforms standalone BM25 (+8.2% MRR) and standalone Semantic (+4.8% MRR) by fusing lexical guideline terms with semantic concepts.

---

## 📁 Generated Artifacts

- **Raw Result Rows:** [`data/eval_results/evaluation_results.csv`](file:///E:/Nadod/Osteoporosis_RAG/data/eval_results/evaluation_results.csv)
- **Configuration Summary:** [`data/eval_results/evaluation_summary.csv`](file:///E:/Nadod/Osteoporosis_RAG/data/eval_results/evaluation_summary.csv)
- **Structured JSON:** [`data/eval_results/evaluation_results.json`](file:///E:/Nadod/Osteoporosis_RAG/data/eval_results/evaluation_results.json)
- **Visual Charts:** `data/eval_results/plots/`
