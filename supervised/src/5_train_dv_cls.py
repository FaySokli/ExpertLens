import json
import logging
import os

import hydra
from indxr import Indxr
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch import nn as nn
import tqdm
from torch import save, load
from torch.optim import AdamW
from sklearn.model_selection import train_test_split


from model.models import DeepViewClassifier

logger = logging.getLogger(__name__)

def train_dv_cls(embeddings, labels, model, optimizer, loss_fn, device, batch_size):
    n_samples = embeddings.size(0)
    total_loss, correct, total = 0.0, 0, 0

    indices = torch.randperm(n_samples)
    embeddings_shuffled = embeddings[indices]
    labels_shuffled = labels[indices]

    for start in tqdm.trange(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_embs = embeddings_shuffled[start:end].to(device)
        batch_labels = labels_shuffled[start:end].to(device)

        optimizer.zero_grad()
        logits = model(batch_embs)
        loss = loss_fn(logits, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_embs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += batch_labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total
    print(f"AVG Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    return train_loss
    
def validate_dv_cls(val_embeddings, val_labels, model, loss_fn, batch_size, device):
    n_samples = val_embeddings.size(0)
    total_loss = 0.0

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_embs = val_embeddings[start:end].to(device)
            batch_labels = val_labels[start:end].to(device)

            logits = model(batch_embs)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item() * batch_embs.size(0)

    val_loss = total_loss / n_samples
    print(f"AVG Val Loss: {val_loss:.4f}")
    return val_loss

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    os.makedirs(cfg.dataset.runs_dir, exist_ok=True)

    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_training_dv_cls.log"
    logging.basicConfig(filename=os.path.join(cfg.dataset.logs_dir, logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    dv_dir = os.path.join(cfg.dataset.output_dir, 'dv_plots')
    os.makedirs(dv_dir, exist_ok=True)

    # -------------------------------
    # Load doc embeddings and expert ids
    # -------------------------------
    doc_embedding = torch.load(f'{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.pt', weights_only=True).to(cfg.model.init.device)
    # doc_embedding = doc_embedding.half()

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
    softmaxed_logits = torch.stack([softmax_logits_map.get(doc_id) for doc_id in sorted_doc_ids]).to("cpu")
    del softmax_logits_map, doc_embedding, doc_logits, logits_map

    X_train, X_temp, y_train, y_temp = train_test_split(
        doc_embedding_sorted, softmaxed_logits, test_size=0.2, stratify=softmaxed_logits, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.05, stratify=y_temp, random_state=42
    )

    # -------------------------------
    # Model, optimizer, loss
    # -------------------------------
    model = DeepViewClassifier(cfg.model.init.embedding_size, cfg.model.adapters.num_experts, device=cfg.model.init.device)
    model = model.to(cfg.model.init.device)

    optimizer = AdamW([
        {'params': model.cls_4.parameters(), 'lr': cfg.training.lr},
        {'params': model.cls_5.parameters(), 'lr': cfg.training.lr}
    ])

    loss_fn = nn.CrossEntropyLoss()

    # -------------------------------
    # Training loop
    # -------------------------------
    best_val_loss = float('inf')
    max_epoch = cfg.training.max_epoch
    batch_size = cfg.training.batch_size

    for epoch in tqdm.tqdm(range(max_epoch), leave=True):
        # Train
        model.train()
        avg_train_loss = train_dv_cls(X_train, y_train, model, optimizer, loss_fn, device, batch_size)
        logging.info(f"TRAIN EPOCH: {epoch + 1:3d}, Average Loss: {avg_train_loss:.5e}")

        # Validate
        model.eval()
        val_loss = validate_dv_cls(X_val, y_val, model, loss_fn, batch_size, device)
        logging.info(f"VAL EPOCH: {epoch + 1:3d}, Average Val Loss: {val_loss:.5e}")

        # Save best model checkpoint
        if val_loss < best_val_loss:
            logging.info(f'Found new best cls model on epoch {epoch + 1}, new best validation loss {val_loss:.5e}')
            best_val_loss = val_loss
            checkpoint_path = os.path.join(dv_dir, "deepview_cls2.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f'Saved checkpoint: {checkpoint_path}')

    # Test
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    test_loss = validate_dv_cls(X_test, y_test, model, loss_fn, batch_size, device)

    with torch.no_grad():
        logits = model(X_test.to(device))
        preds = logits.argmax(dim=1).cpu()
        test_acc = (preds == y_test).float().mean().item()

    logging.info(f"TEST RESULTS -> Loss: {test_loss:.5e}, Accuracy: {test_acc:.4f}")
    print(f"TEST RESULTS -> Loss: {test_loss:.5e}, Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
