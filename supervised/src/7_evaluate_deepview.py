import json
import logging
import os

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from indxr import Indxr
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer

from deepview.DeepViewIR import DeepViewIR
from deepview.evaluate import evaluate_inv_umap, evaluate_umap, leave_one_out_knn_dist_err
from model.models import DeepViewClassifier, MoEBiEncoder
from model.utils import seed_everything

logger = logging.getLogger(__name__)

# ANALYSIS_QIDS = ["test646", "test1937", "test2225"]
ANALYSIS_QIDS = ["5a7b5af7554299042af8f752", "5a7199725542994082a3e88f", "5ae2eda355429928c4239570"]
LAMBDAS = [round(l * 0.1, 1) for l in range(11)]   # [0.0, 0.1, ..., 1.0]


def _setup_eval_logger(eval_dir: str) -> logging.Logger:
    eval_logger = logging.getLogger("eval")
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False
    fh = logging.FileHandler(os.path.join(eval_dir, "eval_results.log"), mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    eval_logger.addHandler(fh)
    eval_logger.addHandler(ch)
    return eval_logger


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = cfg.model.init.device
    num_experts = cfg.model.adapters.num_experts
    embedding_size = cfg.model.init.embedding_size
    batch_size = cfg.training.batch_size

    seed_everything(cfg.general.seed)

    eval_dir = os.path.join(cfg.dataset.output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    eval_log = _setup_eval_logger(eval_dir)
    eval_log.info("=" * 70)
    eval_log.info("DeepView evaluation")
    eval_log.info("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    config = AutoConfig.from_pretrained(cfg.model.init.doc_model)
    config.num_experts = num_experts
    config.adapter_residual = cfg.model.adapters.residual
    config.adapter_latent_size = cfg.model.adapters.latent_size
    config.adapter_non_linearity = cfg.model.adapters.non_linearity
    config.use_adapters = cfg.model.adapters.use_adapters
    config.num_experts_to_use = cfg.model.adapters.num_experts_to_use
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model, config=config)
    model = MoEBiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=num_experts,
        max_tokens=cfg.model.init.max_tokenizer_length,
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.pooling_mode,
        use_adapters=cfg.model.adapters.use_adapters,
        device=device,
    )
    moe_ckpt = (
        f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}'
        f'_experts{num_experts}-{cfg.model.init.specialized_mode}.pt'
    )
    model.load_state_dict(torch.load(moe_ckpt, map_location=device, weights_only=True))
    model.eval()
    eval_log.info(f"Loaded MoE model: {moe_ckpt}")

    doc_embedding = torch.load(
        f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}'
        f'_experts{num_experts}_fullrank.pt',
        map_location=device,
        weights_only=True,
    ).half()

    doc_logits = Indxr(cfg.testing.corpus_logits, key_id='_id')
    logits_map = {doc['_id']: doc['logits'] for doc in doc_logits}
    softmax_logits_map = {
        _id: torch.softmax(torch.tensor(logits, dtype=torch.float32) / 10, dim=-1)
        for _id, logits in logits_map.items()
    }
    with open(
        f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}'
        f'_experts{num_experts}_fullrank.json'
    ) as f:
        id_to_index = json.load(f)
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    sorted_doc_ids = [index_to_id[i] for i in range(len(index_to_id))]
    sorted_indices = [id_to_index[doc_id] for doc_id in sorted_doc_ids]
    doc_embedding_sorted = doc_embedding[sorted_indices]
    softmaxed_logits = torch.stack(
        [softmax_logits_map[doc_id] for doc_id in sorted_doc_ids]
    ).to(device)
    del softmax_logits_map, doc_embedding, doc_logits, logits_map

    # dv_cls 
    dv_dir = os.path.join(cfg.dataset.output_dir, "dv_plots")
    dv_cls = DeepViewClassifier(embedding_size, num_experts, device=device).to(device)
    dv_cls.load_state_dict(torch.load(
        os.path.join(dv_dir, "deepview_cls.pt"), map_location=device
    ))
    dv_cls.eval()
    eval_log.info(f"Loaded dv_cls from {dv_dir}/deepview_cls.pt")

    def _pred_wrapper(x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            return dv_cls(x).softmax(dim=1).cpu().numpy()

    queries = {q['_id']: q for q in Indxr(cfg.testing.query_path, key_id='_id')}
    classes = np.arange(num_experts)
    with open(cfg.testing.qrels_path) as f:
        qrels = json.load(f)

    # Per-query, per-lambda evaluation 
    for query_id in ANALYSIS_QIDS:
        if query_id not in queries:
            eval_log.warning(f"[SKIP] {query_id} not found in queries.")
            continue

        eval_log.info(f"\n{'='*70}\nQuery: {query_id}\n{'='*70}")
        with torch.no_grad():
            q_encoded = model.encoder_no_moe([queries[query_id]['text']])
            query_embedding = model.embedder_q(q_encoded).half()
            bert_scores = torch.einsum('nd,nd->n', doc_embedding_sorted, query_embedding)

        topk_indices = torch.argsort(bert_scores, descending=True)[:1000].tolist()
        top_doc_embeddings = doc_embedding_sorted[topk_indices]
        top_doc_expert_ids = torch.argmax(softmaxed_logits[topk_indices], dim=1).tolist()

        topk_doc_ids = [sorted_doc_ids[i] for i in topk_indices]
        topk_doc_id_to_pos = {doc_id: pos + 1 for pos, doc_id in enumerate(topk_doc_ids)}
        relevant_doc_ids = set(qrels.get(query_id, {}).keys())
        relevant_indices = [
            topk_doc_id_to_pos[doc_id]
            for doc_id in relevant_doc_ids
            if doc_id in topk_doc_id_to_pos
        ]
        eval_log.info(f"Relevant docs in top-1000: {len(relevant_indices)} / {len(relevant_doc_ids)}")

        all_embs = torch.cat([query_embedding, top_doc_embeddings], dim=0)
        X = all_embs.detach().cpu().float().numpy()
        # Query's true expert ID = top retrieved doc's expert.
        true_expert_ids = np.array([top_doc_expert_ids[0]] + top_doc_expert_ids)

        for lam in LAMBDAS:
            eval_log.info(f"\n--- lambda = {lam:.1f} ---")

            disc_dist = lam != 1.0
            deepview = DeepViewIR(
                _pred_wrapper, classes, len(X), batch_size, (embedding_size,),
                n=10, lam=lam, resolution=100, cmap="tab10",
                interactive=False, verbose=False,
                title=f"DeepViewIR sup — query {query_id} — λ={lam:.1f}",
                metric="cosine", disc_dist=disc_dist,
                relevant_docs=relevant_indices,
            )
            deepview.add_samples(X, true_expert_ids)
            deepview.show()

            # Save plot
            plot_path = os.path.join(
                eval_dir,
                f"deepview_{query_id}_lam{lam:.1f}.png",
            )
            deepview.fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(deepview.fig)
            eval_log.info(f"Saved plot: {plot_path}")

            q_knn_pred = leave_one_out_knn_dist_err(deepview.distances, deepview.y_pred)
            q_knn_true = leave_one_out_knn_dist_err(deepview.distances, deepview.y_true)
            eval_log.info(f"Q_knn (pred labels): {q_knn_pred:.4f}")
            eval_log.info(f"Q_knn (true labels): {q_knn_true:.4f}")

            deepview.y_true = true_expert_ids
            umap_results = evaluate_umap(
                deepview, return_values=True, compare_unsup=True, X=X, Y=true_expert_ids
            )
            for split, metrics in umap_results.items():
                for name, val in metrics.items():
                    eval_log.info(f"evaluate_umap  [{split}] {name}: {val:.4f}")

            inv_train_acc, inv_test_acc = evaluate_inv_umap(deepview, X, true_expert_ids)
            eval_log.info(f"evaluate_inv_umap  train acc: {inv_train_acc:.1f}%  test acc: {inv_test_acc:.1f}%")

    eval_log.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()