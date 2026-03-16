import json
import logging
import os
import ipdb
from collections import Counter

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.models import MoEBiEncoder
from model.utils import seed_everything

from ranx import Run, Qrels, compare

import random
import numpy as np

logger = logging.getLogger(__name__)

def get_full_bert_rank(data, model, model_name, doc_embedding, softmaxed_logits, id_to_index, ranx_qrels, k=1000, device=None, output_dir=None):
    """
    Get ranking and compute knn + expert distributions
    """
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    
    # knn and expert distributions
    k_values = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    relevance_counts = {k: [] for k in k_values}
    avg_relevant_scores = {k: [] for k in k_values}
    doc_expert_counter = Counter()
    query_expert_counter = Counter()
    matching_queries = []
    mismatching_queries = []
    queries_without_relevants = []
    
    relevants_dict = ranx_qrels.to_dict()

    # Count document experts
    for idx in range(len(softmaxed_logits)):
        # ipdb.set_trace()
        doc_expert = int(softmaxed_logits[idx].argmax().item())
        doc_expert_counter[doc_expert] += 1
    
    model.eval()

    for d in tqdm.tqdm(data, total=len(data)):
        query_id = d['_id']
        
        with torch.no_grad():
            # with torch.autocast(device_type=device):
            q_encoded = model.encoder_no_moe([d['text']])

            if model.specialized_mode == 'densec3_top1':
                q_embedding = model.embedder_q_inf(q_encoded).half()
                q_embedding = torch.einsum('md,nm->nd', q_embedding.squeeze(0), softmaxed_logits.half()) + q_encoded
                q_embedding = q_embedding.half()
                
                del q_encoded
                torch.cuda.empty_cache()
                doc_embedding = doc_embedding.to(device)
                bert_scores = torch.einsum('nd,nd->n', doc_embedding, q_embedding)
                # doc_embedding = doc_embedding.to("cpu")
                
            elif model.specialized_mode == 'densec3_w':
                q_embedding = model.embedder_q(q_encoded).half()
                doc_embedding = doc_embedding.to(device)
                bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
                # doc_embedding = doc_embedding.to("cpu")

        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        # ipdb.set_trace()
        bert_run[query_id] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}
        
        # Analyze relevant documents
        relevants = set(relevants_dict.get(query_id, {}).keys())
        bert_scores_list = bert_scores.tolist()
        if not relevants:
            queries_without_relevants.append(query_id)
            for k_val in k_values:
                relevance_counts[k_val].append(0)
            continue
        
        # Count relevant docs in top-k
        for k_val in k_values:
            topk_ids = bert_ids[:k_val]
            topk_scores = bert_scores_list[:k_val]

            relevant_scores = [
                score for doc_id, score in zip(topk_ids, topk_scores)
                if doc_id in relevants
            ]

            relevance_counts[k_val].append(len(relevant_scores))

            if len(relevant_scores) > 0:
                avg_relevant_scores[k_val].append(float(np.mean(relevant_scores)))
            else:
                avg_relevant_scores[k_val].append(0.0)

        relevant_euclid_dists = []

        for doc_id in relevants: 
            doc_idx = id_to_index[doc_id]
            dist = torch.norm(
                doc_embedding[doc_idx].unsqueeze(0) - q_embedding,  # subtract for Euclidean
                p=2
            ).item()
            relevant_euclid_dists.append(dist)

        # Store average for this query
        avg_rel_euclid = float(np.mean(relevant_euclid_dists))

        # Get query expert if using adapters
        top1_doc_id = bert_ids[0]
        top1_doc_idx = id_to_index[top1_doc_id]
        query_expert = int(softmaxed_logits[top1_doc_idx].argmax().item())
        query_expert_counter[query_expert] += 1
        
        # Check query vs top-1 relevant expert matching
        top1_relevant_doc_id = None
        top1_relevant_expert = None
        
        for doc_id in bert_ids:
            if doc_id in relevants:
                top1_relevant_doc_id = doc_id
                doc_idx = id_to_index[doc_id]
                top1_relevant_expert = int(softmaxed_logits[doc_idx].argmax().item())
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
    avg_all_rel_euclid = float(np.mean(avg_rel_euclid))
    relevance_stats = {}
    for k_val in k_values:
        precision = np.array(relevance_counts[k_val]) / k_val
        avg_scores = np.array(avg_relevant_scores[k_val])

        relevance_stats[k_val] = {
            'precision': float(np.mean(precision)),
            'avg_relevant_score': float(np.mean(avg_scores)),
            'total_queries': len(precision),
            'queries_with_relevants': len(precision[precision > 0])
        }
    
    # Log KNN and expert distribution analysis
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{model_name}_knn_statistics.log'), 'w') as f:
            f.write("KNN and Expert Distribution Analysis\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Avg all-relevant Euclidean distance: {avg_all_rel_euclid:.16f}\n")
            
            for k_val in k_values:
                stats = relevance_stats[k_val]
                f.write(f"\nk={k_val}:\n")
                f.write(f"  Precision:    {stats['precision']:.4f}\n")
                f.write(f"  Avg relevant score: {stats['avg_relevant_score']:.4f}\n")
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
    print(f"Avg all-relevant Euclidean distance: {avg_all_rel_euclid:.16f}")
    for k_val in k_values:
        stats = relevance_stats[k_val]
        print(
            f"k={k_val:4d}: "
            f"precision={stats['precision']:6.4f} | "
            f"avg_rel_score={stats['avg_relevant_score']:.4f}"
        )
    
    print("\nExpert Distribution:")
    print("-" * 80)
    print(f"Document experts: {dict(doc_expert_counter)}")
    print(f"Query experts: {dict(query_expert_counter)}")

    print("\nExpert Matching Summary:")
    print("-" * 80)
    print(f"Matching queries (query expert = top relevant doc expert): {len(matching_queries)}")
    print(f"Mismatching queries: {len(mismatching_queries)}")
    print(f"Queries without relevants: {len(queries_without_relevants)}")
    if total_with_relevants > 0:
        print(f"Match rate: {match_rate:.2f}%")
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
        max_tokens=cfg.model.init.max_tokenizer_length,
        normalize=cfg.model.init.normalize,
        specialized_mode=cfg.model.init.specialized_mode,
        pooling_mode=cfg.model.init.pooling_mode,
        use_adapters = cfg.model.adapters.use_adapters,
        device=cfg.model.init.device
    )
    model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-{cfg.model.init.specialized_mode}.pt', weights_only=True))
    model_name=f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-{cfg.model.init.specialized_mode}'
    print(f'{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}-{cfg.model.init.specialized_mode}.pt')
    
    torch.set_grad_enabled(False)
    doc_embedding = torch.load(f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.pt', weights_only=True) #.to("cpu")
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
    doc_embedding_sorted = doc_embedding[sorted_indices] #.to("cpu")
    softmaxed_logits = torch.stack([softmax_logits_map.get(doc_id) for doc_id in sorted_doc_ids])
    del softmax_logits_map, doc_embedding, doc_logits, logits_map
    torch.cuda.empty_cache()
    data = Indxr(cfg.testing.query_path, key_id='_id')
    
    ranx_qrels = Qrels.from_file(cfg.testing.qrels_path)
    bert_run = get_full_bert_rank(data, model, model_name, doc_embedding_sorted, softmaxed_logits, id_to_index, ranx_qrels, 1000, cfg.model.init.device, output_dir=cfg.dataset.logs_dir)
    ranx_run = Run(bert_run, 'FullRun')
    models = [ranx_run]

    ranx_run.save(f'{cfg.dataset.runs_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder-{cfg.model.init.specialized_mode}.json')
    
    evaluation_report = compare(ranx_qrels, models, ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3'])
    print(evaluation_report)
    logging.info(f"Results for {cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_biencoder.json:\n{evaluation_report}")


if __name__ == '__main__':
    main()
