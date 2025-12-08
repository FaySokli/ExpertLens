import json
import logging
import os

# Set OpenBLAS environment variables to avoid OpenMP warnings
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

# Import from overcomplete library
from overcomplete.sae.train import train_sae

# Import local model
from overcomplete.sae import TopKRAESAE, TopKSAE
from model.utils import seed_everything

logger = logging.getLogger(__name__)


def baseline_criterion(x, x_hat, pre_codes, codes, dictionary, classifier=None, beta=None):
    loss = (x - x_hat).square().mean()

    # is dead of shape (k) (nb concepts) and is 1 iif
    # not a single code has fire in the batch
    is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
    # we push the pre_codes (before relu) towards the positive orthant
    reanim_loss = (pre_codes * is_dead[None, :]).mean()

    loss -= reanim_loss * 1e-3
    return loss


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Train two SAEs on document embeddings using overcomplete's training function:
    1. First, train a plain TopKSAE (top-k sparse autoencoder)
    2. Then, train a TopKRAESAE (top-k relaxed archetypal sparse autoencoder)

    The script loads document embeddings from the specified dataset and trains
    both models sequentially using overcomplete's built-in training pipeline.
    """
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)

    logging_file = f"{cfg.model.init.doc_model.replace('/', '_')}_train_sae.log"
    logging.basicConfig(
        filename=os.path.join(cfg.dataset.logs_dir, logging_file),
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    # Load document embeddings
    np_data_dir = cfg.testing.embedding_dir
    prefix = "fullrank"

    # Load embeddings
    np_embedding_path = os.path.join(
        np_data_dir,
        f"doc_embeddings_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_{prefix}.npy"
    )

    logger.info(f"Loading embeddings from {np_embedding_path}")
    all_doc_embeddings_np = np.load(np_embedding_path)
    logger.info(
        f"Loaded {len(all_doc_embeddings_np)} document embeddings of dimension {all_doc_embeddings_np.shape[1]}")

    # Convert to torch tensor
    device = cfg.model.init.device
    embeddings_tensor = torch.from_numpy(all_doc_embeddings_np).float().to(device)

    # SAE configuration
    embedding_dim = embeddings_tensor.shape[1]
    nb_concepts = cfg.sae.nb_concepts if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'nb_concepts') else embedding_dim * 2
    top_k = cfg.sae.top_k if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'top_k') and cfg.sae.top_k is not None else None
    delta = 1.0  # Set delta = 1 as requested
    encoder_module = cfg.sae.encoder_module if hasattr(cfg, 'sae') and hasattr(cfg.sae,
                                                                               'encoder_module') and cfg.sae.encoder_module is not None else None
    num_archetypal_points = cfg.sae.num_archetypal_points if hasattr(cfg, 'sae') and hasattr(cfg.sae,
                                                                                             'num_archetypal_points') and cfg.sae.num_archetypal_points is not None else nb_concepts

    # Training configuration
    batch_size = cfg.sae.batch_size if hasattr(cfg, 'sae') and hasattr(cfg.sae,
                                                                       'batch_size') else cfg.training.batch_size
    num_epochs = cfg.sae.num_epochs if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'num_epochs') else 50
    learning_rate = cfg.sae.learning_rate if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'learning_rate') else 1e-3

    logger.info(f"SAE Configuration:")
    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Number of concepts: {nb_concepts}")
    logger.info(f"  Top-k: {top_k}")
    logger.info(f"  Delta: {delta}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Number of epochs: {num_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")

    # Create data loader
    dataset = TensorDataset(embeddings_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup loss function
    sparsity_weight = cfg.sae.sparsity_weight if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'sparsity_weight') else 0.01
    criterion = lambda x, x_hat, pre_codes, codes, dictionary: baseline_criterion(x, x_hat, pre_codes, codes,
                                                                                  dictionary)

    # ============================================================================
    # TRAIN FIRST SAE: Plain TopKSAE
    # ============================================================================
    logger.info("=" * 80)
    logger.info("TRAINING FIRST SAE: Plain TopKSAE")
    logger.info("=" * 80)

    # Create TopKSAE model
    logger.info("Creating TopKSAE model...")
    sae_model = TopKSAE(
        input_shape=embedding_dim,
        nb_concepts=nb_concepts,
        top_k=top_k,
        device=device
    )
    sae_model = sae_model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(sae_model.parameters(), lr=learning_rate)

    # Train using overcomplete's training function
    logger.info("Starting training of TopKSAE using overcomplete's train_sae function...")
    logs_topk = train_sae(
        model=sae_model,
        dataloader=data_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        nb_epochs=num_epochs,
        clip_grad=1.0,
        monitoring=2,  # Monitor and log training losses and statistics
        device=device
    )

    # Report metrics for TopKSAE
    logger.info("=" * 80)
    logger.info("TopKSAE Training Metrics Summary:")
    logger.info("=" * 80)

    if 'avg_loss' in logs_topk and len(logs_topk['avg_loss']) > 0:
        final_loss = logs_topk['avg_loss'][-1]
        initial_loss = logs_topk['avg_loss'][0]
        logger.info(f"Final Average Loss: {final_loss:.6f}")
        logger.info(f"Initial Average Loss: {initial_loss:.6f}")
        logger.info(
            f"Loss Reduction: {initial_loss - final_loss:.6f} ({((initial_loss - final_loss) / initial_loss * 100):.2f}%)")

    if 'r2' in logs_topk and len(logs_topk['r2']) > 0:
        final_r2 = logs_topk['r2'][-1]
        logger.info(f"Final R2 Score: {final_r2:.6f}")

    if 'z_sparsity' in logs_topk and len(logs_topk['z_sparsity']) > 0:
        final_sparsity = logs_topk['z_sparsity'][-1]
        logger.info(f"Final L0 Sparsity: {final_sparsity:.2f}")

    if 'dead_features' in logs_topk and len(logs_topk['dead_features']) > 0:
        final_dead_ratio = logs_topk['dead_features'][-1]
        logger.info(f"Final Dead Features Ratio: {final_dead_ratio * 100:.2f}%")

    if 'time_epoch' in logs_topk:
        total_time = sum(logs_topk['time_epoch'])
        logger.info(f"Total Training Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    logger.info("=" * 80)

    # Save the trained TopKSAE model
    model_save_path_topk = os.path.join(
        cfg.dataset.model_dir,
        f"sae_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.pt"
    )
    torch.save(sae_model.state_dict(), model_save_path_topk)
    logger.info(f"Saved TopKSAE model to {model_save_path_topk}")

    # Save training logs for TopKSAE
    logs_save_path_topk = os.path.join(
        cfg.dataset.logs_dir,
        f"sae_training_logs_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.json"
    )
    # Convert logs to JSON-serializable format
    logs_serializable_topk = {k: [float(v) if isinstance(v, (torch.Tensor, np.number)) else v for v in v_list]
                              for k, v_list in logs_topk.items()}
    with open(logs_save_path_topk, 'w') as f:
        json.dump(logs_serializable_topk, f, indent=2)
    logger.info(f"Saved TopKSAE training logs to {logs_save_path_topk}")

    logger.info("TopKSAE training completed!")

    # ============================================================================
    # TRAIN SECOND SAE: Archetypal TopKRAESAE
    # ============================================================================
    logger.info("=" * 80)
    logger.info("TRAINING SECOND SAE: Archetypal TopKRAESAE")
    logger.info("=" * 80)

    # Initialize archetypal points using KMeans on a subset of embeddings
    logger.info("Initializing archetypal points using KMeans...")
    sample_size = min(10000, len(embeddings_tensor))
    sample_indices = np.random.choice(len(embeddings_tensor), sample_size, replace=False)
    sample_embeddings = embeddings_tensor[sample_indices].cpu().numpy()

    kmeans = KMeans(n_clusters=num_archetypal_points, random_state=42, n_init=10)
    kmeans.fit(sample_embeddings)
    archetypal_points = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
    logger.info(f"Initialized {num_archetypal_points} archetypal points")

    # Create TopKRAESAE model with archetypal dictionary
    logger.info("Creating TopKRAESAE model...")
    sae_model_archetypal = TopKRAESAE(
        input_shape=embedding_dim,
        nb_concepts=nb_concepts,
        points=archetypal_points,
        top_k=top_k,
        encoder_module=encoder_module,
        delta=delta,
        device=device
    )
    sae_model_archetypal = sae_model_archetypal.to(device)

    # Setup optimizer for archetypal SAE
    optimizer_archetypal = torch.optim.Adam(sae_model_archetypal.parameters(), lr=learning_rate)

    # Train using overcomplete's training function
    logger.info("Starting training of TopKRAESAE using overcomplete's train_sae function...")
    logs_archetypal = train_sae(
        model=sae_model_archetypal,
        dataloader=data_loader,
        criterion=criterion,
        optimizer=optimizer_archetypal,
        scheduler=None,
        nb_epochs=num_epochs,
        clip_grad=1.0,
        monitoring=2,  # Monitor and log training losses and statistics
        device=device
    )

    # Report metrics for TopKRAESAE
    logger.info("=" * 80)
    logger.info("TopKRAESAE Training Metrics Summary:")
    logger.info("=" * 80)

    if 'avg_loss' in logs_archetypal and len(logs_archetypal['avg_loss']) > 0:
        final_loss = logs_archetypal['avg_loss'][-1]
        initial_loss = logs_archetypal['avg_loss'][0]
        logger.info(f"Final Average Loss: {final_loss:.6f}")
        logger.info(f"Initial Average Loss: {initial_loss:.6f}")
        logger.info(
            f"Loss Reduction: {initial_loss - final_loss:.6f} ({((initial_loss - final_loss) / initial_loss * 100):.2f}%)")

    if 'r2' in logs_archetypal and len(logs_archetypal['r2']) > 0:
        final_r2 = logs_archetypal['r2'][-1]
        logger.info(f"Final R2 Score: {final_r2:.6f}")

    if 'z_sparsity' in logs_archetypal and len(logs_archetypal['z_sparsity']) > 0:
        final_sparsity = logs_archetypal['z_sparsity'][-1]
        logger.info(f"Final L0 Sparsity: {final_sparsity:.2f}")

    if 'dead_features' in logs_archetypal and len(logs_archetypal['dead_features']) > 0:
        final_dead_ratio = logs_archetypal['dead_features'][-1]
        logger.info(f"Final Dead Features Ratio: {final_dead_ratio * 100:.2f}%")

    if 'time_epoch' in logs_archetypal:
        total_time = sum(logs_archetypal['time_epoch'])
        logger.info(f"Total Training Time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    logger.info("=" * 80)

    # Save the trained TopKRAESAE model
    model_save_path_archetypal = os.path.join(
        cfg.dataset.model_dir,
        f"sae_archetypal_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.pt"
    )
    torch.save(sae_model_archetypal.state_dict(), model_save_path_archetypal)
    logger.info(f"Saved TopKRAESAE model to {model_save_path_archetypal}")

    # Save archetypal points
    points_save_path = os.path.join(
        cfg.dataset.model_dir,
        f"sae_archetypal_points_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.pt"
    )
    torch.save(archetypal_points, points_save_path)
    logger.info(f"Saved archetypal points to {points_save_path}")

    # Save training logs for TopKRAESAE
    logs_save_path_archetypal = os.path.join(
        cfg.dataset.logs_dir,
        f"sae_archetypal_training_logs_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.json"
    )
    # Convert logs to JSON-serializable format
    logs_serializable_archetypal = {k: [float(v) if isinstance(v, (torch.Tensor, np.number)) else v for v in v_list]
                                    for k, v_list in logs_archetypal.items()}
    with open(logs_save_path_archetypal, 'w') as f:
        json.dump(logs_serializable_archetypal, f, indent=2)
    logger.info(f"Saved TopKRAESAE training logs to {logs_save_path_archetypal}")

    logger.info("TopKRAESAE training completed!")
    logger.info("=" * 80)
    logger.info("All training completed! Both SAEs have been trained and saved.")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
