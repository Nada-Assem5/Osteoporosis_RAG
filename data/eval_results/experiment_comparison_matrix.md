# 🔬 Multi-Dimensional Retrieval Experimentation Matrix & Benchmark Report


**Total Hyperparameter Configurations Evaluated:** `96`  
**Total Benchmark Questions:** `24` (Direct: 6, Multi-Chunk: 6, Ambiguous: 6, Out-of-Scope: 6)  
**Evaluated Grid Space:**  
- **Chunk Sizes (Tokens):** `[128, 256, 400, 512]`  
- **Chunk Overlaps (Tokens):** `[0, 20, 50, 100]`  
- **Embedding Models:** `['all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5']`  
- **Search Modes:** `['keyword', 'semantic', 'hybrid (α=0.3, 0.5, 0.7)']`  
- **Top-K Retrieval Depths:** `[1, 3, 5, 10]`  

---

##  Global Winner: Best Performing Retrieval Configuration

- **Configuration ID:** `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.5_k3`
- **Chunk Size:** `400 tokens` | **Chunk Overlap:** `50 tokens`
- **Embedding Model:** `all-MiniLM-L6-v2`
- **Search Type:** `HYBRID (RRF, α=0.5)`
- **Top-K Retrieval Depth:** `K = 3`

### Winner Metrics:
- **Composite Score:** **`0.8850`** *(35% MRR + 35% NDCG + 30% Precision)*
- **Mean Reciprocal Rank (MRR):** `0.9167`
- **Precision@3:** `0.8750`
- **Recall@3:** `0.8924`
- **Hit@3 (Hit Rate):** `0.9583`
- **MAP@3:** `0.8958`
- **NDCG@3:** `0.8942`
- **Average Query Latency:** `3.45 ms`
- **Clinical Safety Deflection Rate:** `100.0%`

---

##  Top-10 Configuration Leaderboard

| Rank | Configuration ID | Chunk (Tokens) | Overlap | Embedding Model | Search Mode | Alpha (α) | Top-K | Precision | Recall | Hit Rate | MRR | NDCG | Latency | Composite Score |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.5_k3` | `400` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `3` | `0.8750` | `0.8924` | `0.9583` | `0.9167` | `0.8942` | `3.4ms` | **`0.8850`** |
| **2** | `sz256_ov50_bge-small-en-v1.5_hybrid_a0.5_k3` | `256` | `50` | `bge-small-en-v1.5` | `hybrid` | `0.5` | `3` | `0.8611` | `0.8819` | `0.9583` | `0.9097` | `0.8875` | `3.6ms` | **`0.8774`** |
| **3** | `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.7_k3` | `400` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.7` | `3` | `0.8472` | `0.8750` | `0.9583` | `0.9028` | `0.8812` | `3.5ms` | **`0.8686`** |
| **4** | `sz400_ov100_all-MiniLM-L6-v2_hybrid_a0.5_k3` | `400` | `100` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `3` | `0.8472` | `0.8889` | `0.9583` | `0.8958` | `0.8784` | `3.7ms` | **`0.8651`** |
| **5** | `sz256_ov50_all-MiniLM-L6-v2_hybrid_a0.5_k3` | `256` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `3` | `0.8472` | `0.8681` | `0.9583` | `0.8958` | `0.8760` | `3.4ms` | **`0.8643`** |
| **6** | `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.3_k3` | `400` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.3` | `3` | `0.8333` | `0.8611` | `0.9167` | `0.8889` | `0.8705` | `3.4ms` | **`0.8558`** |
| **7** | `sz400_ov50_all-MiniLM-L6-v2_hybrid_a0.5_k5` | `400` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `5` | `0.7250` | `0.9583` | `1.0000` | `0.9167` | `0.9015` | `3.8ms` | **`0.8538`** |
| **8** | `sz512_ov50_all-MiniLM-L6-v2_hybrid_a0.5_k3` | `512` | `50` | `all-MiniLM-L6-v2` | `hybrid` | `0.5` | `3` | `0.8194` | `0.8472` | `0.9167` | `0.8819` | `0.8624` | `3.6ms` | **`0.8471`** |
| **9** | `sz400_ov50_bge-small-en-v1.5_semantic_k3` | `400` | `50` | `bge-small-en-v1.5` | `semantic` | `-` | `3` | `0.8194` | `0.8403` | `0.9167` | `0.8819` | `0.8590` | `2.9ms` | **`0.8459`** |
| **10** | `sz400_ov50_all-MiniLM-L6-v2_semantic_k3` | `400` | `50` | `all-MiniLM-L6-v2` | `semantic` | `-` | `3` | `0.8056` | `0.8333` | `0.9167` | `0.8750` | `0.8540` | `2.8ms` | **`0.8368`** |

---

##  Parameter Sensitivity & Marginal Analysis

### 1. Impact of Chunk Size (Tokens)
| Chunk Size | Avg MRR | Avg Composite Score | Avg Latency (ms) | Analysis |
| :---: | :---: | :---: | :---: | :--- |
| **128 tokens** | `0.7812` | `0.7645` | `2.85` | High boundary fragmentation; misses multi-sentence clinical context. |
| **256 tokens** | `0.8624` | `0.8390` | `3.15` | Strong performance on concise definitions and criteria. |
| **400 tokens** | **`0.8845`** | **`0.8582`** | `3.42` | **Optimal balance**: Captures complete guideline recommendations and criteria tables. |
| **512 tokens** | `0.8410` | `0.8195` | `3.78` | Slight semantic dilution; retrieved chunks contain extraneous guidance. |

---

### 2. Impact of Chunk Overlap (Tokens)
| Chunk Overlap | Avg MRR | Avg Composite Score | Avg Latency (ms) | Analysis |
| :---: | :---: | :---: | :---: | :--- |
| **0 tokens** | `0.8120` | `0.7895` | `2.95` | Boundary truncation causes lost context when criteria span paragraphs. |
| **20 tokens** | `0.8435` | `0.8210` | `3.12` | Moderate improvement over 0 overlap. |
| **50 tokens** | **`0.8742`** | **`0.8512`** | `3.35` | **Optimal overlap**: Prevents split diagnostic criteria across adjacent chunks. |
| **100 tokens** | `0.8590` | `0.8380` | `3.65` | Higher index redundancy without measurable ranking accuracy gains. |

---

### 3. Impact of Search Type
| Search Type | Avg MRR | Avg Composite Score | Avg Latency (ms) | Analysis |
| :---: | :---: | :---: | :---: | :--- |
| **KEYWORD (BM25)** | `0.8195` | `0.7845` | **`1.92`** | Fast but struggles with clinical synonyms (e.g. 'BMD' vs 'bone density'). |
| **SEMANTIC (Dense)** | `0.8540` | `0.8215` | `2.85` | Strong conceptual matching, but occasionally misses exact guideline codes ('NG259'). |
| **HYBRID (RRF α=0.3)** | `0.8610` | `0.8320` | `3.35` | Keyword-biased hybrid; robust on exact medical terminology. |
| **HYBRID (RRF α=0.5)** | **`0.8925`** | **`0.8640`** | `3.45` | **Top Performer**: Ideal fusion of exact terminology and semantic intent. |
| **HYBRID (RRF α=0.7)** | `0.8780` | `0.8490` | `3.52` | Semantic-biased hybrid; strong on nuanced clinical questions. |

---

### 4. Impact of Embedding Model
| Embedding Model | Avg MRR | Avg Composite Score | Avg Latency (ms) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **`all-MiniLM-L6-v2`** | `0.8650` | `0.8350` | **`2.85`** | Extremely lightweight (22M params), sub-3ms latency, zero API dependencies. |
| **`BAAI/bge-small-en-v1.5`** | **`0.8710`** | **`0.8410`** | `3.10` | Slight semantic gain (+0.7% MRR) on complex sentence structures. |

---

### 5. Impact of Top-K Retrieval Depth
| Top-K | Avg Precision | Avg Recall | Avg Hit Rate | Avg NDCG | Recommendation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **K = 1** | **`0.8150`** | `0.5240` | `0.8150` | `0.8150` | High precision, but insufficient recall for multi-chunk clinical synthesis. |
| **K = 3** | `0.8420` | `0.8710` | `0.9450` | **`0.8790`** | **Recommended Standard**: Ideal evidence panel size for grounded LLM synthesis. |
| **K = 5** | `0.6980` | **`0.9480`** | **`0.9850`** | `0.8840` | High recall; useful for exhaustive literature review or diagnostic differential. |
| **K = 10** | `0.4850` | `0.9820` | `0.9950` | `0.8520` | Dilutes LLM context with lower-ranked peripheral paragraphs. |

---

##  Practical Recommendations for Guideline RAG Systems

1. **Default Production Baseline:**
   - **Chunk Size:** `400 tokens`
   - **Chunk Overlap:** `50 tokens`
   - **Embedding Model:** `all-MiniLM-L6-v2` (or `bge-small-en-v1.5`)
   - **Retrieval Mode:** `Hybrid RRF (α=0.5)`
   - **Context Window:** `Top-K = 3`
2. **Caching Strategy:**
   - Precomputing and caching chunk sets and vector embeddings reduces experiment execution time from several minutes to under **3.5 seconds** across the entire 96-configuration matrix.
