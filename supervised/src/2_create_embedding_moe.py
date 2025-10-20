import json
import logging
import os

import hydra
import torch
import tqdm
from indxr import Indxr
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig

from model.models import MoEBiEncoder
from model.utils import seed_everything

logger = logging.getLogger(__name__)


    
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_create_embedding_biencoder.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    
    seed_everything(cfg.general.seed)
    corpus = Indxr(cfg.testing.corpus_path, key_id='_id')
    corpus = sorted(corpus, key=lambda k: len(k.get("title", "") + k.get("text", "")), reverse=True)
    logits = Indxr(cfg.testing.corpus_logits, key_id='_id')
    logits_map = {doc['_id']: doc['logits'] for doc in logits}
    combined = list(map(lambda doc: {**doc, 'logits': logits_map.get(doc['_id'])} if doc['_id'] in logits_map else doc, corpus))
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.init.tokenizer)
    config = AutoConfig.from_pretrained(cfg.model.init.doc_model)
    config.num_experts = cfg.model.adapters.num_experts
    config.num_experts_to_use = cfg.model.adapters.num_experts_to_use
    config.adapter_residual = cfg.model.adapters.residual
    config.adapter_latent_size = cfg.model.adapters.latent_size
    config.adapter_non_linearity = cfg.model.adapters.non_linearity
    config.use_adapters = cfg.model.adapters.use_adapters
    config.device = cfg.model.init.device
    doc_model = AutoModel.from_pretrained(cfg.model.init.doc_model, config=config)
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

    """
    logging.info(f'Loading model from {cfg.model.init.save_model}.pt')
    if os.path.exists(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'):
        model.load_state_dict(torch.load(f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt'))
    else:
        logging.info('New model CLS requested, creating new checkpoint')
        torch.save(model.state_dict(), f'{cfg.dataset.model_dir}/{cfg.model.init.save_model}.pt')
    """
    index = 0
    texts = []
    doc_logits = []
    id_to_index = {}
    
    model.eval()
    embedding_matrix = torch.zeros(len(combined), cfg.model.init.embedding_size, device=config.device).float()
    all_doc_logits = torch.zeros(len(combined), cfg.model.adapters.num_experts, device=config.device).float()
    for doc in tqdm.tqdm(combined):
        
        id_to_index[doc['_id']] = index
        index += 1
        texts.append(doc.get('title','').lower() + ' ' + doc['text'].lower())
        doc_logits.append(torch.tensor(logits_map.get(doc['_id'])))
        if len(texts) == cfg.training.batch_size:
            with torch.no_grad():
                #with torch.autocast(device_type=cfg.model.init.device):
                all_doc_logits[index - len(texts) : index] = model.cls(torch.stack(doc_logits))
                embedding_matrix[index - len(texts) : index] = model.encoder(texts, all_doc_logits[index - len(texts) : index])
            texts = []
            doc_logits = []
    if texts:
        with torch.no_grad():
            # with torch.autocast(device_type=cfg.model.init.device):
            all_doc_logits[index - len(texts) : index] = model.cls(torch.stack(doc_logits))
            embedding_matrix[index - len(texts) : index] = model.encoder(texts, all_doc_logits[index - len(texts) : index])
            
    
    prefix = 'fullrank'
    logging.info(f'Embedded {index} documents. Saving embedding matrix in folder {cfg.testing.embedding_dir}.')
    os.makedirs(cfg.testing.embedding_dir, exist_ok=True)
    torch.save(embedding_matrix, f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.pt')
        
    logging.info('Saving id_to_index file.')
    with open(f'{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.json', 'w') as f:
        json.dump(id_to_index, f)
    
if __name__ == '__main__':
    main()