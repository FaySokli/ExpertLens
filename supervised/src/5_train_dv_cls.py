import json
import logging
import os

import hydra
from indxr import Indxr
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
import tqdm
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

from model.models import DeepViewClassifier

logger = logging.getLogger(__name__)

def train_dv_cls(embeddings, teacher_logits, model, optimizer, loss_fn, device, batch_size, temperature):
    n_samples = embeddings.size(0)
    total_loss = 0.0

    indices = torch.randperm(n_samples)
    embeddings = embeddings[indices]
    teacher_logits = teacher_logits[indices]

    total_correct = 0

    for start in tqdm.trange(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_embs = embeddings[start:end].to(device)
        batch_logits = teacher_logits[start:end].to(device)

        optimizer.zero_grad()

        student_logits = model(batch_embs)

        student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = torch.softmax(batch_logits / temperature, dim=1)

        loss = loss_fn(student_log_probs, teacher_probs) * (temperature ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_embs.size(0)
        total_correct += (student_logits.argmax(dim=1) == batch_logits.argmax(dim=1)).sum().item()

    avg_loss = total_loss / n_samples
    accuracy = total_correct / n_samples
    print(f"AVG Train KL Loss: {avg_loss:.6f} | Accuracy: {accuracy:.4f}")
    return avg_loss


def validate_dv_cls(
    embeddings, teacher_logits, model,
    loss_fn, batch_size, device, temperature
):
    n_samples = embeddings.size(0)
    total_loss = 0.0

    total_correct = 0

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_embs = embeddings[start:end].to(device)
            batch_logits = teacher_logits[start:end].to(device)

            student_logits = model(batch_embs)

            student_log_probs = torch.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = torch.softmax(batch_logits / temperature, dim=1)

            loss = loss_fn(student_log_probs, teacher_probs) * (temperature ** 2)
            total_loss += loss.item() * batch_embs.size(0)
            total_correct += (student_logits.argmax(dim=1) == batch_logits.argmax(dim=1)).sum().item()

    avg_loss = total_loss / n_samples
    accuracy = total_correct / n_samples
    print(f"AVG Val KL Loss: {avg_loss:.6f} | Accuracy: {accuracy:.4f}")
    return avg_loss


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, "train_dv_cls_logits.log"),
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )

    dv_dir = os.path.join(cfg.dataset.output_dir, "dv_plots")
    os.makedirs(dv_dir, exist_ok=True)

    device = cfg.model.init.device
    temperature = cfg.training.get("distill_temperature", 10.0)

    doc_embedding = torch.load(
        f"{cfg.testing.embedding_dir}/{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.pt",
        weights_only=True
    ).float()

    doc_logits = Indxr(cfg.testing.corpus_logits, key_id="_id")
    logits_map = {
        str(doc["_id"]): torch.tensor(doc["logits"], dtype=torch.float32)
        for doc in doc_logits
    }

    with open(
        f"{cfg.testing.embedding_dir}/id_to_index_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_fullrank.json",
        "r"
    ) as f:
        id_to_index = json.load(f)

    index_to_id = {v: k for k, v in id_to_index.items()}
    sorted_doc_ids = [index_to_id[i] for i in range(len(index_to_id))]
    sorted_indices = [id_to_index[_id] for _id in sorted_doc_ids]

    X = doc_embedding[sorted_indices]
    Y = torch.stack([logits_map[_id] for _id in sorted_doc_ids])

    del doc_embedding, doc_logits, logits_map

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.05, random_state=42
    )

    model = DeepViewClassifier(
        cfg.model.init.embedding_size,
        cfg.model.adapters.num_experts,
        device=device
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    best_val_loss = float("inf")

    for epoch in range(cfg.training.max_epoch):
        print(f"\nEpoch {epoch + 1}")

        model.train()
        train_loss = train_dv_cls(
            X_train, Y_train, model, optimizer,
            loss_fn, device, cfg.training.batch_size, temperature
        )

        model.eval()
        val_loss = validate_dv_cls(
            X_val, Y_val, model,
            loss_fn, cfg.training.batch_size, device, temperature
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(dv_dir, "deepview_cls.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"Saved best model → {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    test_loss = validate_dv_cls(
        X_test, Y_test, model,
        loss_fn, cfg.training.batch_size, device, temperature
    )

    print(f"\nFINAL TEST KL LOSS: {test_loss:.6f}")


if __name__ == "__main__":
    main()