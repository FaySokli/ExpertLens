# ExpertLens

**ExpertLens** is a post-n Information Retrieval (IR) framework that combines **Mixture of Experts (MoE)** bi-encoders with **DeepView** visualization techniques. It enables interpretable dense document ranking by routing queries and documents through specialized expert networks, then visualizing the resulting embedding space with relevance highlighting.

---

## Overview

Dense retrieval models embed queries and documents into a shared vector space and rank by similarity. ExpertLens extends this paradigm with:

- **MoE Bi-Encoder**: Multiple specialized sub-networks (experts) that learn domain-specific representations. Each query/document is routed through relevant experts via a learned gating mechanism.
- **DeepView Visualizations**: Low-dimensional projections (UMAP) of the embedding space, with queries highlighted in yellow and relevant documents outlined in red — enabling visual inspection of retrieval quality and expert specialization.
- **Expert Analysis**: Textual and structural analysis of which documents each expert specializes in (vocabulary richness, readability, entropy, etc.).

Both **supervised** (with pre-computed relevance signals) and **unsupervised** (with Sparse Auto-Encoders) training pipelines are provided.

---

## Project Structure

```
ExpertLens/
├── ExpertLens/
│   └── DeepViewIR.py               # IR-specific DeepView visualization class
├── supervised/                     # Supervised training pipeline
│   ├── conf/                       # Hydra configuration files
│   │   ├── config.yaml             # Main config with defaults
│   │   ├── dataset/                # Dataset-specific configs (msmarco, hotpotqa, nq-train, ...)
│   │   ├── model/                  # Model configs (bert-base, colbert, contriever, ...)
│   │   ├── training/               # Optimizer and training hyperparameters
│   │   └── testing/                # Evaluation configs
│   └── src/
│       ├── 1_train_new_moe.py      # Step 1: Train MoE bi-encoder
│       ├── 2_create_embedding_moe.py  # Step 2: Generate corpus embeddings
│       ├── 3_test_biencoder_moe.py    # Step 3: Rank documents and evaluate
│       ├── 3_test_knn.py              # KNN-based retrieval baseline
│       ├── 4_expert_cluster_analysis.py  # Step 4: Analyze expert specialization
│       ├── 5_train_dv_cls.py          # Step 5: Train DeepView classifier
│       ├── 7_evaluate_deepview.py     # Step 7: Visualize and evaluate
│       ├── dataloader/dataloader.py   # Dataset loading utilities
│       ├── model/
│       │   ├── models.py             # MoEBiEncoder, DeepViewClassifier, Specializer
│       │   ├── loss.py               # MultipleRankingLossBiEncoder
│       │   └── utils.py              # Seed and misc utilities
│       ├── visualizations.py         # UMAP plotting utilities
│       ├── text_analysis.py          # Readability and vocabulary metrics
│       └── data_preprocessing.py     # Data format conversion
├── unsupervised/                   # Unsupervised pipeline (includes SAE training)
│   ├── conf/
│   └── src/
│       └── 6_train_sae.py          # Sparse Auto-Encoder training
├── clean_dv_plots.py               # Utility to clean up orphaned plot files
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---|---|
| `transformers` | HuggingFace transformer models |
| `sentence-transformers` | Pre-trained bi-encoders |
| `beir` | IR benchmark datasets and evaluation |
| `hydra-core` | Composable configuration management |
| `ranx` | IR evaluation metrics (nDCG, MRR, MAP, Recall) |
| `indxr` | Fast JSONL corpus indexing |
| `umap-learn` | Dimensionality reduction for visualization |
| `scikit-learn` | ML utilities |

> PyTorch must be installed separately. See [pytorch.org](https://pytorch.org/get-started/locally/) for instructions.

---

## Configuration

ExpertLens uses [Hydra](https://hydra.cc/) for configuration. Configs are composed from three groups:

- **model**: `bert-base`, `bert-small`, `tinybert`, `colbert`, `contriever`
- **dataset**: `msmarco`, `hotpotqa`, `nq-train`, `computer-science`, `political-science`
- **training** / **testing**: hyperparameters and evaluation settings

Override any config value from the command line:

```bash
python src/1_train_new_moe.py model=bert-base dataset=msmarco training.batch_size=32
```

### Model config defaults

| Parameter | Default |
|---|---|
| Experts | 6 |
| Experts used per forward pass | 6 |
| Adapter latent size | 192 |
| Non-linearity | ReLU |
| Pooling | mean / cls |
| Specialization mode | `densec3_top1` |

### Training defaults

| Parameter | Default |
|---|---|
| Learning rate (base model) | 1e-6 |
| Learning rate (specializers) | 1e-4 |
| Batch size | 64 |
| Max epochs | 20 |
| Validation split | 5% |
| Random seed | 42 |

---

## Supported Datasets

| Dataset | Description |
|---|---|
| **MS MARCO** | Large-scale passage ranking; includes TREC 2019/2020 test sets |
| **HotpotQA** | Multi-hop question answering |
| **Natural Questions (NQ)** | Open-domain passage retrieval |
| **Computer Science** | Domain-specific document collection |
| **Political Science** | Domain-specific document collection |

Each dataset config specifies paths to the corpus JSONL, query JSONL, relevance labels (qrels TSV), and output directories.

---

## Pipeline

### Supervised

```
1_train_new_moe.py          →  Trained MoE bi-encoder checkpoint
2_create_embedding_moe.py   →  Corpus embedding tensors + ID mapping
3_test_biencoder_moe.py     →  TREC-format run file + IR metrics
4_expert_cluster_analysis.py→  Per-expert textual analysis logs
5_train_dv_cls.py           →  Lightweight DeepView classifier
7_evaluate_deepview.py      →  UMAP plots + visualization quality metrics
```

### Unsupervised

Same stages as supervised, with an additional step:

```
6_train_sae.py              →  Sparse Auto-Encoder for unsupervised gating
```

---

## Model Architecture

### `MoEBiEncoder`

The core model wraps any HuggingFace transformer with a mixture-of-experts layer:

```
Input text
    └─► Transformer encoder (BERT, ColBERT, Contriever, ...)
            └─► Pooling (mean / max / CLS / identity)
                    └─► Expert gating (learned classifier)
                            ├─► Specializer 1 (Linear → ReLU → Linear)
                            ├─► Specializer 2
                            │   ...
                            └─► Specializer N
                    └─► Weighted combination + residual connection
                            └─► Final embedding
```

Two inference modes:
- **`densec3_top1`**: Routes each sample through its single highest-scoring expert.
- **`densec3_w`**: Combines all expert outputs with softmax-weighted aggregation.

### `Specializer`

Lightweight adapter applied after pooling:
```
hidden_size → hidden_size/2 (ReLU) → hidden_size
```

### `DeepViewClassifier`

Small MLP trained via knowledge distillation from MoE logits for fast expert prediction during visualization:
```
hidden_size → hidden_size/2 (ReLU) → num_experts
```

---

## Evaluation Metrics

### Retrieval

Computed via [ranx](https://github.com/AmenRa/ranx):

- nDCG, nDCG@10, nDCG@100
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)
- Recall@k

### Visualization Quality

Computed on UMAP projections:

- **Trustworthiness** — how well local neighborhoods are preserved
- **Continuity** — how well inverse neighborhoods are preserved
- **Leave-one-out KNN distance error**

### Expert Specialization

Per-expert textual statistics (Bank et al. 2012):

| Metric | Description |
|---|---|
| RVoc | Vocabulary richness |
| CVoc | Vocabulary concentration |
| DVoc | Vocabulary dispersion |
| H | Token entropy |
| Flesch-Kincaid | Readability grade level |
| Avg word length | Mean characters per word |
| Avg sentence length | Mean tokens per sentence |

---

## DeepView Visualizations

The `DeepViewIR` class (in `deepviewIR/DeepViewIR.py`) extends base DeepView with IR-specific annotations:

- **Query**: highlighted in yellow with a distinct border
- **Relevant documents**: outlined in red
- **Documents**: colored/grouped by expert assignment

Visualizations are generated in step 7 and saved as PNG files to the configured output directory.

---

## Usage Example

```bash
cd supervised

# 1. Train the MoE bi-encoder
python src/1_train_new_moe.py model=bert-base dataset=msmarco

# 2. Embed the corpus
python src/2_create_embedding_moe.py model=bert-base dataset=msmarco

# 3. Rank and evaluate
python src/3_test_biencoder_moe.py model=bert-base dataset=msmarco

# 4. Analyze expert specialization
python src/4_expert_cluster_analysis.py model=bert-base dataset=msmarco

# 5. Train DeepView classifier
python src/5_train_dv_cls.py model=bert-base dataset=msmarco

# 6. Generate visualizations and evaluate
python src/7_evaluate_deepview.py model=bert-base dataset=msmarco
```

---

## Citation

If you use ExpertLens in your research, please cite the relevant work (paper link TBD).
