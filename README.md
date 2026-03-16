# ExpertLens

In this paper, we introduce **ExpertLens**, a post-hoc explainability framework for MoE-enhanced dense retrievers. 
ExpertLens leverages discriminative embedding space visualizations derived from DeepView (https://github.com/LucaHermes/DeepView), a framework based on Discriminative Dimensionality Reduction that projects high-dimensional representations into interpretable, discriminative views.
We investigate how retrieval decisions arise from the geometry of learned embedding spaces and the specialization of individual experts.
Our main contributions are:
(1) We leverage DeepView to analyze the embedding spaces produced by MoE-enhanced dense retrievers, providing interpretable post-hoc insights into expert routing, subspace structure, and query-document alignment across domains and IR tasks.
(2) We conduct qualitative and quantitative analyses of expert-induced subspaces, characterizing each expert's specialization in terms of linguistic properties and dominant semantic themes.
(3) We use automatically extracted Concept Activation Vectors (CAVs) to associate semantic attributes with individual experts, linking expert behavior to interpretable conceptual features. 
ExpertLens provides explainable insights into both the organization of the embedding space and the specialization of experts.


## Installation

```bash
pip install -r requirements.txt
```

### Training defaults

| Parameter | Default |
|---|---|
| Learning rate (base model) | 1e-6 |
| Learning rate (experts) | 1e-4 |
| Batch size | 64 |
| Max epochs | 30 |
| Validation split | 5% |
| Random seed | 42 |

---

## Supported Datasets

| Dataset | Description |
|---|---|
| **MS MARCO** | Large-scale passage ranking; includes TREC 2019/2020 test sets |
| **HotpotQA** | Multi-hop question answering |
| **Natural Questions (NQ)** | Open-domain passage retrieval |

Each dataset config specifies paths to the corpus JSONL, query JSONL, relevance labels (qrels TSV), and output directories.

---

## Pipeline

```
1_train_new_moe.py          →  Trained MoE bi-encoder checkpoint
2_create_embedding_moe.py   →  Corpus embedding tensors + expert ID mapping
3_test_biencoder_moe.py     →  TREC-format run file + IR metrics
4_expert_cluster_analysis.py→  Per-expert textual analysis logs
5_train_dv_cls.py           →  Lightweight DeepView classifier
6_train_sae.py              →  Sparse Auto-Encoder for CAVs
7_evaluate_deepview.py      →  UMAP and DeepView plots + visualization quality metrics
```

## Model Architecture

### `MoEBiEncoder`

The core model wraps any HuggingFace transformer with a mixture-of-experts layer:

```
Input text
    └─► Transformer encoder (BERT, ColBERT, ...)
            └─► Pooling (mean)
                    └─► Expert gating (learned classifier)
                            ├─► Expert 1 (Linear → ReLU → Linear)
                            ├─► Expert 2
                            │   ...
                            └─► Expert N
                    └─► Weighted combination + residual connection
                            └─► Final embedding
```

Inference modes:
- **`sbmoe_all``densec3_w`**: Combines all expert outputs with softmax-weighted aggregation.

### `DeepViewClassifier`

Small MLP trained via knowledge distillation from MoE logits for fast expert prediction during visualization:
```
hidden_size → hidden_size/2 (ReLU) → num_experts
```

---

## Evaluation Metrics

### Visualization Quality

Computed on UMAP projections:

- **Trustworthiness** — how well local neighborhoods are preserved
- **Continuity** — how well inverse neighborhoods are preserved
- **Leave-one-out KNN distance error**

### Expert Specialization

Per-expert textual statistics:

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
../ExpertLens/supervised/src/pipeline.sh
../ExpertLens/unsupervised/src/pipeline.sh

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
