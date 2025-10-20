import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.models import MoEBiEncoder, DeepViewClassifier
from model.utils import seed_everything

from ranx import Run, Qrels, compare

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from deepview.DeepViewIR import DeepViewIR
from deepview.evaluate import evaluate_umap
from deepview.evaluate import leave_one_out_knn_dist_err

logger = logging.getLogger(__name__)

def visualize_tsne(query_embedding, top_doc_embeddings, top_doc_ids, query_id, output_dir, experts_used, relevants, use_adapters=True):
    all_embeddings = torch.cat([query_embedding, top_doc_embeddings], dim=0).cpu().numpy()
    relevant_set = set(relevants)

    tsne = TSNE(n_components=3, random_state=42, perplexity=30, init='pca')
    embeddings_3d = tsne.fit_transform(all_embeddings)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(embeddings_3d[0, 0], embeddings_3d[0, 1], embeddings_3d[0, 2], 
               c='black', label='query', marker='X', s=100)

    color_map = plt.cm.tab10
    safe_colors = [color_map(i) for i in range(color_map.N) if i != 3]
    marker_list = ['o', '^', 's', 'D', 'v', 'P', '*', 'X', '<', '>']

    for i, (x, y, z) in enumerate(embeddings_3d[1:], start=1):
        doc_id = top_doc_ids[i-1]
        is_relevant = doc_id in relevant_set

        if use_adapters:
            expert_id = experts_used[i-1]
            color = safe_colors[expert_id % len(safe_colors)]
            marker = marker_list[expert_id % len(marker_list)]
        else:
            color = 'blue'
            marker = 'o'

        if is_relevant:
            ax.scatter(x, y, z, edgecolors='red', facecolors='none', s=120, linewidths=2,
                       marker='o', label='relevant_docs' if i == 1 else "")
        else:
            ax.scatter(x, y, z, c=[color], marker=marker, alpha=0.6, s=40)

    # Legend
    handles = [mlines.Line2D([], [], color='black', marker='X', linestyle='None', markersize=10, label='query'),
               mlines.Line2D([], [], color='red', marker='o', markerfacecolor='none', linestyle='None', markersize=10, label='relevant_docs')]

    if use_adapters:
        unique_experts = sorted(set(experts_used))
        for expert_id in unique_experts:
            handles.append(
                mlines.Line2D([], [], color=safe_colors[expert_id % len(safe_colors)], marker=marker_list[expert_id % len(marker_list)],
                              linestyle='None', markersize=10, label=f'Expert {expert_id + 1}')
            )

    ax.legend(handles=handles, fontsize=20)
    # ax.set_title(f"3D t-SNE: Query {query_id} and Top 1000 Docs")
    plt.grid(True)
    plt.tight_layout()

    filename = f"tsne_query_{query_id}_3D_experts.png" if use_adapters else f"tsne_query_{query_id}_3D_NO_experts.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=900)
    plt.close()
    print(f"Saved t-SNE plot: {save_path}")


def get_bert_rerank(data, model, doc_embedding, bm25_runs, id_to_index):
    bert_run = {}
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder([d['text']])
            
        bm25_docs = list(bm25_runs[d['_id']].keys())
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[d['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
        
    return bert_run


def get_full_bert_rank(data, model, doc_embedding, softmaxed_logits, index_to_id, device, k=1000):
    bert_run = {}
    model.eval()
    for q in tqdm.tqdm(data, total=len(data)):
        with torch.no_grad():
            # with torch.autocast(device_type=device):
            q_encoded = model.encoder_no_moe([q['text']])

            if model.specialized_mode == 'blooms_top1':
                q_embedding = model.embedder_q_inf(q_encoded).half()
                q_embedding = torch.einsum('md,nm->nd', q_embedding.squeeze(0), softmaxed_logits.half()) + q_encoded
                q_embedding = q_embedding.half()
                
                del q_encoded
                torch.cuda.empty_cache()
                doc_embedding = doc_embedding.to(device)
                bert_scores = torch.einsum('nd,nd->n', doc_embedding, q_embedding)
                doc_embedding = doc_embedding.to("cpu")
                
            elif model.specialized_mode == 'blooms_all':
                q_embedding = model.embedder_q(q_encoded).half()
                doc_embedding = doc_embedding.to(device)
                bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
                doc_embedding = doc_embedding.to("cpu")

        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        bert_run[q['_id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}

        
    return bert_run
    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_testing_biencoder.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    config = AutoConfig.from_pretrained(cfg.model.init.doc_model)
    config.num_experts = cfg.model.adapters.num_experts
    config.num_experts_to_use = cfg.model.adapters.num_experts_to_use
    config.adapter_residual = cfg.model.adapters.residual
    config.adapter_latent_size = cfg.model.adapters.latent_size
    config.adapter_non_linearity = cfg.model.adapters.non_linearity
    config.use_adapters = cfg.model.adapters.use_adapters
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model)
    model = MoEBiEncoder(
        doc_model=doc_model,
        tokenizer=tokenizer,
        num_classes=cfg.model.adapters.num_experts,
        max_tokens=cfg.model.init.max_tokenizer_length,
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.pooling_mode,
        use_adapters = cfg.model.adapters.use_adapters,
        device=cfg.model.init.device
    )
    model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-{cfg.model.init.specialized_mode}.pt', weights_only=True))
    # model.load_state_dict(torch.load(f'output/msmarco/saved_models/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-{cfg.model.init.specialized_mode}.pt', weights_only=True))

    dv_cls = DeepViewClassifier(
        hidden_size=cfg.model.init.embedding_size,
        num_classes=cfg.model.adapters.num_experts,
        device=cfg.model.init.device,
    )
    state_dict = torch.load(f'{cfg.dataset.output_dir}/dv_plots/deepview_cls.pt', map_location=cfg.model.init.device)
    dv_cls.load_state_dict(state_dict)
    dv_cls = dv_cls.to(cfg.model.init.device)
    dv_cls.eval()

    torch.set_grad_enabled(False)
    doc_embedding = torch.load(f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.pt', weights_only=True).to("cpu")
    doc_embedding = doc_embedding.half()

    doc_logits = Indxr(cfg.testing.corpus_logits, key_id='_id')
    logits_map = {doc['_id']: doc['logits'] for doc in doc_logits}
    softmax_logits_map = {
    _id: torch.softmax(torch.tensor(logits, dtype=torch.float32)/10, dim=-1).to(cfg.model.init.device)
    for _id, logits in logits_map.items()
    }
    
    with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.json', 'r') as f:
        id_to_index = json.load(f)
    
    # with open(cfg.testing.bm25_run_path, 'r') as f:
    #     bm25_run = json.load(f)

    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    sorted_doc_ids = [index_to_id[i] for i in range(len(index_to_id))]
    sorted_indices = [id_to_index[doc_id] for doc_id in sorted_doc_ids]
    doc_embedding_sorted = doc_embedding[sorted_indices].to("cpu")
    softmaxed_logits = torch.stack([softmax_logits_map.get(doc_id) for doc_id in sorted_doc_ids])
    del softmax_logits_map, doc_embedding, doc_logits, logits_map
    torch.cuda.empty_cache()
    data = Indxr(cfg.testing.query_path, key_id='_id')
    if cfg.testing.rerank:
        bert_run = get_bert_rerank(data, model, doc_embedding_sorted, doc_logits, bm25_run, id_to_index)
    else:
        bert_run = get_full_bert_rank(data, model, doc_embedding_sorted, softmaxed_logits, index_to_id, cfg.model.init.device, 1000)
        
    
    # with open(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_biencoder.json', 'w') as f:
    #     json.dump(bert_run, f)
        
        
    ranx_qrels = Qrels.from_file(cfg.testing.qrels_path)
    
    if cfg.testing.rerank:
        ranx_run = Run(bert_run, 'ReRanker')
        ranx_bm25_run = Run(bm25_run, name='BM25')
        models = [ranx_bm25_run, ranx_run]
    else:
        ranx_run = Run(bert_run, 'FullRun')
        models = [ranx_run]
    
    ranx_run.save(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder-{cfg.model.init.specialized_mode}.json')
    
    evaluation_report = compare(ranx_qrels, models, ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'])
    print(evaluation_report)
    logging.info(f"Results for {cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder.json:\n{evaluation_report}")

    ############################
    # Create directory for t-SNE and DeepView plots
    # qids = ["test59", "test1513""test2514""test2519""test3185""test3227"]
    tsne_dir = os.path.join(cfg.dataset.output_dir, 'tsne_plots')
    os.makedirs(tsne_dir, exist_ok=True)
    relevants_dict = ranx_qrels.to_dict()

    dv_dir = os.path.join(cfg.dataset.output_dir, 'dv_plots')
    os.makedirs(dv_dir, exist_ok=True)
    
    ############################
    # DeepView
    ############################
    # for i in range(43):
    # import time
    # random.seed(time.time())  # Changes every run
    # random_query = random.choice(data)
    # query_id = random_query['_id']
    query_id = "527433"
    print(f"Selected query ID: {query_id}")
    query_data = data.get(query_id)
    if query_data is None:
        print(f"Query ID {query_id} not found in data, skipping.")

    # query
    model.eval()
    with torch.no_grad():
        q_encoded = model.encoder_no_moe([query_data['text']])
        query_embedding = model.embedder_q(q_encoded).half()

    # Get top 1000 docs
    topk_ids = list(bert_run[query_id].keys())[:1000]
    topk_indices = [id_to_index[doc_id] for doc_id in topk_ids]

    # Get embeddings and top-1 expert IDs for those docs
    top_doc_embeddings = doc_embedding_sorted[topk_indices].to(cfg.model.init.device)
    top_doc_logits = softmaxed_logits[topk_indices]
    top_doc_expert_ids = torch.argmax(top_doc_logits, dim=1).tolist()

    # Get relevant doc IDs
    relevants = set(relevants_dict.get(query_id, {}).keys())
    relevant_indices = [
        i+1
        for i, doc_id in enumerate(topk_ids)
        if doc_id in relevants
    ]

    # Generate and save DeepView plot
    all_embeddings = torch.cat([query_embedding, top_doc_embeddings], dim=0)
    with torch.no_grad():
        probs = dv_cls(all_embeddings.float().to(cfg.model.init.device)).softmax(dim=1).cpu().numpy()
    relevant_set = set(relevants)
    X = all_embeddings.detach().cpu().numpy()
    y = np.argmax(probs, axis=1)

    def pred_wrapper(x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(cfg.model.init.device)
            pred = dv_cls(x).softmax(dim=1).cpu().numpy()
        return pred
    
    # --- Deep View Parameters ----
    use_case = "nlp"
    classes = np.arange(cfg.model.adapters.num_experts)
    batch_size = cfg.training.batch_size
    max_samples = 1001  # including query
    data_shape = (cfg.model.init.embedding_size,)
    resolution = 100
    N = 10
    lam = .6
    cmap = 'tab10'
    metric = 'cosine'
    disc_dist = (
        False
        if lam == 1
        else True
    )
    # to make sure deepview.show is blocking,
    # disable interactive mode
    interactive = False
    my_title = "MOE Enhanced DRM - Deepview"

    deepview = DeepViewIR(pred_wrapper, classes, max_samples, batch_size, data_shape,
                                            N, lam, resolution, cmap, interactive, my_title, metric=metric,
                                            disc_dist=disc_dist, relevant_docs=relevant_indices)


    deepview.add_samples(X, y)
    deepview.show()
    # ipdb.set_trace()
    fig = plt.gcf()
    fig.savefig(os.path.join(dv_dir, f"TREC19 - deepviewIR_lam06_query_{query_id}_experts{cfg.model.adapters.num_experts}.png"), dpi=300)
    plt.close(fig)

    q_knn = leave_one_out_knn_dist_err(deepview.distances, deepview.y_pred)
    print('Lambda: %.2f - Pred. Val. Q_kNN: %.3f' % (lam, q_knn))

    q_knn = leave_one_out_knn_dist_err(deepview.distances, deepview.y_true)
    print('Lambda: %.2f - True Val. Q_kNN: %.3f' % (lam, q_knn))
    # ipdb.set_trace()
    # deepview.save_fig(os.path.join(dv_dir, f"deepview_query_{query_id}_experts{cfg.model.adapters.num_experts}.png"))


    ############################
    # 3D t-SNE
    ############################
    # # for i in range(40):
    # for query_id in qids:
    #     # import time
    #     # random.seed(time.time())  # Changes every run
    #     # random_query = random.choice(data)
    #     # query_id = random_query['_id']
    #     # # query_id = "2950442623"
    #     # # random_query = data.get(query_id)
    #     # print(f"Selected query ID: {query_id}")

    #     query_data = next((q for q in data if q['_id'] == query_id), None)
    #     if query_data is None:
    #         print(f"Query ID {query_id} not found in data, skipping.")
    #         continue
    #     print(f"Selected query ID: {query_id}")

    #     # query
    #     model.eval()
    #     with torch.no_grad():
    #         q_encoded = model.encoder_no_moe([query_data['text']])
    #         query_embedding = model.embedder_q(q_encoded).half()

    #     # Get top 1000 docs
    #     topk_ids = list(bert_run[query_id].keys())[:1000]
    #     topk_indices = [id_to_index[doc_id] for doc_id in topk_ids]

    #     # Get embeddings and top-1 expert IDs for those docs
    #     top_doc_embeddings = doc_embedding_sorted[topk_indices].to(cfg.model.init.device)
    #     top_doc_logits = softmaxed_logits[topk_indices]
    #     top_doc_expert_ids = torch.argmax(top_doc_logits, dim=1).tolist()

    #     # Get relevant doc IDs
    #     relevants = set(relevants_dict.get(query_id, {}).keys())

    #     # Generate and save t-SNE
    #     visualize_tsne(query_embedding, top_doc_embeddings, topk_ids, query_id, tsne_dir, top_doc_expert_ids, relevants, use_adapters=cfg.model.adapters.use_adapters)
    ############################

if __name__ == '__main__':
    main()
