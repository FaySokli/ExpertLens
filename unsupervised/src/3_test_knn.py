import json
import logging
import os
import ipdb
from collections import Counter, defaultdict

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
    marker_list = ['o', '^', 's', 'D', 'v', 'P', '*', 'X', '<', '>']

    for i, (x, y, z) in enumerate(embeddings_3d[1:], start=1):
        doc_id = top_doc_ids[i-1]
        is_relevant = doc_id in relevant_set

        if use_adapters:
            expert_id = experts_used[i-1]
            color = color_map(expert_id)
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
                mlines.Line2D([], [], color=color_map(expert_id), marker=marker_list[expert_id % len(marker_list)],
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

def get_full_bert_rank(data, model, model_name, doc_embedding, id_to_index, all_expert_ids, ranx_qrels, use_adapters, k=1000, output_dir=None):
    """
    Get ranking and compute knn + expert distributions
    """
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    
    # knn and expert distributions
    k_values = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    relevance_counts = {k: [] for k in k_values}
    doc_expert_counter = Counter()
    query_expert_counter = Counter()
    matching_queries = []
    mismatching_queries = []
    queries_without_relevants = []
    
    relevants_dict = ranx_qrels.to_dict()

    # Count document experts
    if use_adapters:
        for idx in range(len(all_expert_ids)):
            doc_expert = int(all_expert_ids[idx])
            doc_expert_counter[doc_expert] += 1
    
    model.eval()
    for d in tqdm.tqdm(data, total=len(data)):
        query_id = d['_id']
        
        with torch.no_grad():
            q_embedding = model.query_encoder([d['text']])
        
        # Get query expert if using adapters
        query_expert = None
        if use_adapters:
            with torch.no_grad():
                expert_probs = model.cls(q_embedding)  # [1, num_experts]
                query_expert = torch.argmax(expert_probs, dim=1).item()
                query_expert_counter[query_expert] += 1
        
        # Compute scores and ranking
        bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        bert_run[query_id] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}
        
        # Analyze relevant documents
        relevants = set(relevants_dict.get(query_id, {}).keys())
        
        if not relevants:
            queries_without_relevants.append(query_id)
            for k_val in k_values:
                relevance_counts[k_val].append(0)
            continue
        
        # Count relevant docs in top-k
        for k_val in k_values:
            topk_subset = bert_ids[:k_val]
            num_relevants = len([doc_id for doc_id in topk_subset if doc_id in relevants])
            relevance_counts[k_val].append(num_relevants)
            # ipdb.set_trace()
        
        # Check query vs top-1 relevant expert matching
        if use_adapters:
            top1_relevant_doc_id = None
            top1_relevant_expert = None
            
            for doc_id in bert_ids:
                if doc_id in relevants:
                    top1_relevant_doc_id = doc_id
                    doc_idx = id_to_index[doc_id]
                    top1_relevant_expert = int(all_expert_ids[doc_idx])
                    break
            
            if top1_relevant_expert is not None:
                if query_expert == top1_relevant_expert:
                    matching_queries.append(query_id)
                else:
                    mismatching_queries.append({
                        'query_id': query_id,
                        'query_expert': query_expert,
                        'top1_relevant_doc_id': top1_relevant_doc_id,
                        'top1_relevant_expert': top1_relevant_expert
                    })
    
    # Statistics
    relevance_stats = {}
    for k_val in k_values:
        precision = np.array(relevance_counts[k_val]) / k_val        
        relevance_stats[k_val] = {
            'precision': float(np.mean(precision)),
            'total_queries': len(precision),
            'queries_with_relevants': len(precision[precision > 0])
        }
    
    # Log KNN and expert distribution analysis
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics log
        with open(os.path.join(output_dir, f'{model_name}_knn_statistics.log'), 'w') as f:
            # Matching queries
            f.write(f"Total Matching Queries: {len(matching_queries)}\n")
            f.write("=" * 50 + "\n")
            for qid in matching_queries:
                f.write(f"{qid}\n")
            
            # Mismatching queries
            f.write(f"Total Mismatching Queries: {len(mismatching_queries)}\n")
            f.write("=" * 50 + "\n")
            for qid in mismatching_queries:
                f.write(f"{qid}\n")
            
            # Queries without relevants
            f.write(f"Total Queries Without Relevants: {len(queries_without_relevants)}\n")
            f.write("=" * 50 + "\n")
            for qid in queries_without_relevants:
                f.write(f"{qid}\n")

            f.write("KNN and Expert Distribution Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            for k_val in k_values:
                stats = relevance_stats[k_val]
                f.write(f"\nk={k_val}:\n")
                f.write(f"  Precision:    {stats['precision']:.4f}\n")
                f.write(f"  Total queries:         {stats['total_queries']}\n")
                f.write(f"  Queries w/ relevants:  {stats['queries_with_relevants']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("EXPERT DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write("\nDocument Expert Distribution:\n")
            for expert_id, count in sorted(doc_expert_counter.items()):
                f.write(f"  Expert {expert_id}: {count}\n")
            
            f.write("\nQuery Expert Distribution:\n")
            for expert_id, count in sorted(query_expert_counter.items()):
                f.write(f"  Expert {expert_id}: {count}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("EXPERT MATCHING SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Matching queries (query expert = top relevant doc expert): {len(matching_queries)}\n")
            f.write(f"Mismatching queries: {len(mismatching_queries)}\n")
            f.write(f"Queries without relevants: {len(queries_without_relevants)}\n")
            total_with_relevants = len(matching_queries) + len(mismatching_queries)
            if total_with_relevants > 0:
                match_rate = len(matching_queries) / total_with_relevants * 100
                f.write(f"Match rate: {match_rate:.2f}%\n")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("KNN AND EXPERT DISTRIBUTION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nMatching queries: {len(matching_queries)}")
    print(f"Mismatching queries: {len(mismatching_queries)}")
    print(f"Queries without relevants: {len(queries_without_relevants)}")
    
    print("\nKNN:")
    print("-" * 80)
    for k_val in k_values:
        stats = relevance_stats[k_val]
        print(f"k={k_val:4d}: precision={stats['precision']:6.2f}")
    
    print("\nExpert Distribution:")
    print("-" * 80)
    print(f"Document experts: {dict(doc_expert_counter)}")
    print(f"Query experts: {dict(query_expert_counter)}")
    print("=" * 80 + "\n")

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
        use_adapters = cfg.model.adapters.use_adapters,
        device=cfg.model.init.device
    )
    
    if cfg.model.adapters.use_adapters:
        if cfg.model.init.specialized_mode == "sbmoe_top1":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt', weights_only=True))
            model_name=f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1'
            print(f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt')
        elif cfg.model.init.specialized_mode == "sbmoe_all":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt', weights_only=True))
            model_name=f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1'
            print(f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-sbmoe_top1.pt')
        elif cfg.model.init.specialized_mode == "random":
            model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random.pt', weights_only=True))
            model_name=f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random'
            print(f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-random.pt')
    else:
        model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft.pt', weights_only=True))
        model_name=f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft'
        print(f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-ft.pt')
    
    # dv_cls = DeepViewClassifier(
    #     hidden_size=cfg.model.init.embedding_size,
    #     num_classes=cfg.model.adapters.num_experts,
    #     device=cfg.model.init.device,
    # )
    # state_dict = torch.load(f'{cfg.dataset.output_dir}/dv_plots/deepview_cls.pt', map_location=cfg.model.init.device)
    # dv_cls.load_state_dict(state_dict)
    # dv_cls = dv_cls.to(cfg.model.init.device)
    # dv_cls.eval()
    
    if cfg.model.adapters.use_adapters:
        doc_embedding = torch.load(f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.pt', weights_only=True).to(cfg.model.init.device)
        with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.json', 'r') as f:
            id_to_index = json.load(f)
    else:
        doc_embedding = torch.load(f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_ft_fullrank.pt', weights_only=True).to(cfg.model.init.device)
        with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_ft_fullrank.json', 'r') as f:
            id_to_index = json.load(f)
        
    data = Indxr(cfg.testing.query_path, key_id='_id')
    ############################
    tsne_dir = os.path.join(cfg.dataset.output_dir, 'tsne_plots')
    os.makedirs(tsne_dir, exist_ok=True)

    dv_dir = os.path.join(cfg.dataset.output_dir, 'dv_plots')
    os.makedirs(dv_dir, exist_ok=True)

    # Load .npy embedding and label files
    np_data_dir = cfg.testing.embedding_dir
    prefix = "fullrank"

    # Load embeddings and expert_ids
    if cfg.model.adapters.use_adapters:
        print("EXPERTS")
        np_embedding_path = os.path.join(np_data_dir, f"doc_embeddings_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy")
        expert_ids_path = os.path.join(np_data_dir, f"expert_ids_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy")
    else:
        print("NO EXPERTS")
        np_embedding_path = os.path.join(np_data_dir, f"doc_embeddings_{cfg.model.init.save_model}_ft_{prefix}.npy")
        expert_ids_path = os.path.join(np_data_dir, f"expert_ids_{cfg.model.init.save_model}_ft_{prefix}.npy")

    all_doc_embeddings_np = np.load(np_embedding_path)
    all_expert_ids = np.load(expert_ids_path)
    ranx_qrels = Qrels.from_file(cfg.testing.qrels_path)
    bert_run = get_full_bert_rank(data, model, model_name, doc_embedding, id_to_index, all_expert_ids, ranx_qrels, cfg.model.adapters.use_adapters, 1000, output_dir=cfg.dataset.logs_dir)
    ranx_run = Run(bert_run, 'FullRun')
    models = [ranx_run]

    if cfg.model.adapters.use_adapters:
        ranx_run.save(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder-{cfg.model.init.specialized_mode}.json')
    else:
        ranx_run.save(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder-ft.json')
    
    evaluation_report = compare(ranx_qrels, models, ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'])
    print(evaluation_report)
    logging.info(f"Results for {cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder.json:\n{evaluation_report}")

    # ############################
    # # DeepView
    # ############################
    # for i in range(43):
    #     import time
    #     random.seed(time.time())  # Changes every run
    #     random_query = random.choice(data)
    #     query_id = random_query['_id']
    #     print(f"Selected query ID: {query_id}")
    #     query_data = data.get(query_id)
    #     if query_data is None:
    #         print(f"Query ID {query_id} not found in data, skipping.")

    #     # query
    #     model.eval()
    #     with torch.no_grad():
    #         query_embedding = model.query_encoder([query_data['text']]).to(cfg.model.init.device)

    #     # Get top 1000 docs
    #     topk_ids = list(bert_run[query_id].keys())[:1000]
    #     topk_indices = [id_to_index[doc_id] for doc_id in topk_ids]
    #     top_doc_embeddings = doc_embedding[topk_indices]
    #     top_doc_expert_ids = [int(all_expert_ids[i]) for i in topk_indices]
    #     relevants_dict = ranx_qrels.to_dict()
    #     relevants = set(relevants_dict.get(query_id, {}).keys())
    #     relevant_indices = [
    #         i+1
    #         for i, doc_id in enumerate(topk_ids)
    #         if doc_id in relevants
    #     ]

    #     # Generate and save DeepView plot
    #     all_embeddings = torch.cat([query_embedding, top_doc_embeddings], dim=0)
    #     with torch.no_grad():
    #         probs = dv_cls(all_embeddings.float().to(cfg.model.init.device)).softmax(dim=1).cpu().numpy()
    #     relevant_set = set(relevants)
    #     X = all_embeddings.detach().cpu().numpy()
    #     y = np.argmax(probs, axis=1)

    #     def pred_wrapper(x):
    #         with torch.no_grad():
    #             x = torch.from_numpy(x).float().to(cfg.model.init.device)
    #             pred = dv_cls(x).softmax(dim=1).cpu().numpy()
    #         return pred
        
    #     # --- Deep View Parameters ----
    #     use_case = "nlp"
    #     classes = np.arange(cfg.model.adapters.num_experts)
    #     batch_size = cfg.training.batch_size
    #     max_samples = 1001  # including query
    #     data_shape = (cfg.model.init.embedding_size,)
    #     resolution = 100
    #     N = 10
    #     lam = .6
    #     cmap = 'tab10'
    #     metric = 'cosine'
    #     disc_dist = (
    #         False
    #         if lam == 1
    #         else True
    #     )
    #     # to make sure deepview.show is blocking,
    #     # disable interactive mode
    #     interactive = False
    #     my_title = "MOE Enhanced DRM - Deepview"

    #     deepview = DeepViewIR(pred_wrapper, classes, max_samples, batch_size, data_shape,
    #                                             N, lam, resolution, cmap, interactive, my_title, metric=metric,
    #                                             disc_dist=disc_dist, relevant_docs=relevant_indices)


    #     deepview.add_samples(X, y)
    #     deepview.show()
    #     # ipdb.set_trace()
    #     fig = plt.gcf()
    #     fig.savefig(os.path.join(dv_dir, f"TREC19 - deepview_query_{query_id}_experts{cfg.model.adapters.num_experts}.png"), dpi=300)
    #     plt.close(fig)

    #     q_knn = leave_one_out_knn_dist_err(deepview.distances, deepview.y_pred)
    #     print('Lambda: %.2f - Pred. Val. Q_kNN: %.3f' % (lam, q_knn))

    #     q_knn = leave_one_out_knn_dist_err(deepview.distances, deepview.y_true)
    #     print('Lambda: %.2f - True Val. Q_kNN: %.3f' % (lam, q_knn))
    #     # ipdb.set_trace()
    #     # deepview.save_fig(os.path.join(dv_dir, f"deepview_query_{query_id}_experts{cfg.model.adapters.num_experts}.png"))
    # ############################

if __name__ == '__main__':
    main()
