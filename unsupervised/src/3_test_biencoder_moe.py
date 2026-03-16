import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.models import MoEBiEncoder, DeepViewClassifier
from model.utils import seed_everything

from ranx import Run, Qrels, compare

logger = logging.getLogger(__name__)

# Query IDs for per-query analysis
ANALYSIS_QIDS = [
    "test1968", "test2932" ,"test3132"
]


def get_full_bert_rank(data, model, doc_embedding, id_to_index, k=1000):
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder([d['text']])

        bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}

    return bert_run


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)

    dataset_name = HydraConfig.get().runtime.choices.get('testing', 'unknown')

    # Optional: '+analysis.umap=true',...
    run_text_analysis  = bool(OmegaConf.select(cfg, 'analysis.text',     default=False))
    run_umap           = bool(OmegaConf.select(cfg, 'analysis.umap',     default=False))
    run_deepview       = bool(OmegaConf.select(cfg, 'analysis.deepview', default=False))
    run_corpus_metrics = bool(OmegaConf.select(cfg, 'analysis.corpus',   default=False))
    query_ids_override = OmegaConf.select(cfg, 'analysis.query_ids',     default=None)

    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_{dataset_name}_testing_biencoder.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    stats_logger = logging.getLogger('txt_stats')
    stats_logger.setLevel(logging.INFO)
    stats_logger.propagate = False
    _stats_fh = logging.FileHandler(
        os.path.join(cfg.dataset.logs_dir, f'txt_char_stats_{dataset_name}.log'), mode='a'
    )
    _stats_fh.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    stats_logger.addHandler(_stats_fh)

    seed_everything(cfg.general.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    config = AutoConfig.from_pretrained(cfg.model.init.doc_model)
    config.num_experts = cfg.model.adapters.num_experts
    config.adapter_residual = cfg.model.adapters.residual
    config.adapter_latent_size = cfg.model.adapters.latent_size
    config.adapter_non_linearity = cfg.model.adapters.non_linearity
    config.use_adapters = cfg.model.adapters.use_adapters
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)
    model = MoEBiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=cfg.model.adapters.num_experts,
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.aggregation_mode,
        use_adapters=cfg.model.adapters.use_adapters,
        device=cfg.model.init.device
    )

    if cfg.model.adapters.use_adapters:
        if cfg.model.init.specialized_mode in ("sbmoe_top1", "sbmoe_all"):
            ckpt = (
                f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}'
                f'_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt'
            )
        elif cfg.model.init.specialized_mode == "random":
            ckpt = (
                f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}'
                f'_experts{cfg.model.adapters.num_experts}-random.pt'
            )
        else:
            raise ValueError(f"Unknown specialized_mode: {cfg.model.init.specialized_mode}")
    else:
        ckpt = (
            f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}'
            f'_experts{cfg.model.adapters.num_experts}-ft.pt'
        )

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print(f"Loaded checkpoint: {ckpt}")

    if cfg.model.adapters.use_adapters:
        doc_embedding = torch.load(
            f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}'
            f'_experts{cfg.model.adapters.num_experts}_fullrank.pt',
            weights_only=True,
        ).to(cfg.model.init.device)
        with open(
            f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}'
            f'_experts{cfg.model.adapters.num_experts}_fullrank.json'
        ) as f:
            id_to_index = json.load(f)
    else:
        doc_embedding = torch.load(
            f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_ft_fullrank.pt',
            weights_only=True,
        ).to(cfg.model.init.device)
        with open(
            f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_ft_fullrank.json'
        ) as f:
            id_to_index = json.load(f)

    data = Indxr(cfg.testing.query_path, key_id='_id')

    bert_run = get_full_bert_rank(data, model, doc_embedding, id_to_index, 1000)
    ranx_qrels = Qrels.from_file(cfg.testing.qrels_path)
    ranx_run = Run(bert_run, 'FullRun')

    if cfg.model.adapters.use_adapters:
        run_save_path = (
            f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}'
            f'_experts{cfg.model.adapters.num_experts}'
            f'_biencoder-{cfg.model.init.specialized_mode}_{dataset_name}.json'
        )
    else:
        run_save_path = (
            f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}'
            f'_experts{cfg.model.adapters.num_experts}'
            f'_biencoder-ft_{dataset_name}.json'
        )
    ranx_run.save(run_save_path)

    evaluation_report = compare(
        ranx_qrels, [ranx_run],
        ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'],
    )
    print(evaluation_report)
    logging.info(
        f"Results for {cfg.model.init.save_model}"
        f"_experts{cfg.model.adapters.num_experts}_biencoder_{dataset_name}.json:\n"
        f"{evaluation_report}"
    )
    if not any([run_text_analysis, run_umap, run_deepview, run_corpus_metrics]):
        return

    corpus_metrics = None
    if run_corpus_metrics or run_text_analysis:
        from text_analysis import run_corpus_analysis
        corpus_metrics = run_corpus_analysis(
            cfg.testing.corpus_path, stats_logger, dataset_name
        )

    all_expert_ids = None
    if run_umap or run_deepview or run_text_analysis:
        import numpy as np
        prefix = "fullrank"
        if cfg.model.adapters.use_adapters:
            expert_ids_path = os.path.join(
                cfg.testing.embedding_dir,
                f"expert_ids_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy",
            )
        else:
            expert_ids_path = os.path.join(
                cfg.testing.embedding_dir,
                f"expert_ids_{cfg.model.init.save_model}_ft_{prefix}.npy",
            )
        all_expert_ids = np.load(expert_ids_path)

    dv_cls = None
    if run_deepview:
        dv_cls = DeepViewClassifier(
            hidden_size=cfg.model.init.embedding_size,
            num_classes=cfg.model.adapters.num_experts,
            device=cfg.model.init.device,
        )
        state_dict = torch.load(
            f'{cfg.dataset.output_dir}/dv_plots/deepview_cls_experts6.pt',
            map_location=cfg.model.init.device,
        )
        dv_cls.load_state_dict(state_dict)
        dv_cls = dv_cls.to(cfg.model.init.device)
        dv_cls.eval()

    corpus_index = None
    if run_text_analysis:
        corpus_index = Indxr(cfg.testing.corpus_path, key_id='_id')

    query_ids = list(query_ids_override or ANALYSIS_QIDS)
    umap_dir = os.path.join(cfg.dataset.output_dir, 'umap_plots')
    dv_dir   = os.path.join(cfg.dataset.output_dir, 'dv_plots')
    if run_umap:
        os.makedirs(umap_dir, exist_ok=True)
    if run_deepview:
        os.makedirs(dv_dir, exist_ok=True)

    relevants_dict = ranx_qrels.to_dict()

    for query_id in query_ids:
        print(f"\n{'='*60}\nProcessing query: {query_id}")
        query_data = data.get(query_id)
        if query_data is None:
            print(f"  [SKIP] Query ID {query_id} not found in data.")
            continue

        model.eval()
        with torch.no_grad():
            query_embedding = model.query_encoder([query_data['text']]).to(cfg.model.init.device)

        topk_ids = list(bert_run[query_id].keys())[:1000]
        topk_indices = [id_to_index[doc_id] for doc_id in topk_ids]
        top_doc_embeddings = doc_embedding[topk_indices]
        top_doc_expert_ids = [int(all_expert_ids[i]) for i in topk_indices]
        relevants = set(relevants_dict.get(query_id, {}).keys())
        relevant_indices = [
            i + 1
            for i, doc_id in enumerate(topk_ids)
            if doc_id in relevants
        ]

        if run_umap:
            from visualizations import visualize_umap
            visualize_umap(
                query_embedding, top_doc_embeddings, topk_ids,
                query_id, umap_dir, top_doc_expert_ids, relevants,
                use_adapters=cfg.model.adapters.use_adapters,
            )


        if run_deepview:
            from visualizations import visualize_deepview
            with torch.no_grad():
                query_expert_id = int(
                    dv_cls(query_embedding.float().to(cfg.model.init.device))
                    .argmax(-1).item()
                )
            true_expert_ids = [query_expert_id] + top_doc_expert_ids
            visualize_deepview(
                query_embedding, top_doc_embeddings, relevant_indices,
                dv_cls, cfg.model.init.device,
                cfg.model.adapters.num_experts,
                cfg.training.batch_size,
                cfg.model.init.embedding_size,
                query_id, dataset_name, dv_dir,
                true_expert_ids=true_expert_ids,
            )

        if run_text_analysis:
            from text_analysis import run_per_query_text_analysis
            run_per_query_text_analysis(
                query_id, topk_ids, top_doc_expert_ids, corpus_index,
                corpus_metrics, cfg.dataset.output_dir, dataset_name, stats_logger,
            )


if __name__ == '__main__':
    main()
