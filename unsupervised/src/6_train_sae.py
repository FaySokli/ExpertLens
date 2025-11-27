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
from overcomplete.sae.losses import mse_l1

# Import local model
from model.sae_models import TopKRAESAE
from model.utils import seed_everything

logger = logging.getLogger(__name__)

def baseline_criterion(x, x_hat, pre_codes, codes, dictionary):
    """Baseline MSE reconstruction loss."""
    return (x - x_hat).square().mean()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Train an Archetypal SAE on document embeddings using overcomplete's training function.
    
    The script loads document embeddings from the specified dataset and trains
    a Top-K Relaxed Archetypal Sparse Autoencoder on them using overcomplete's
    built-in training pipeline.
    """
    os.makedirs(cfg.dataset.output_dir, exist_ok=True)
    os.makedirs(cfg.dataset.logs_dir, exist_ok=True)
    os.makedirs(cfg.dataset.model_dir, exist_ok=True)
    
    logging_file = f"{cfg.model.init.doc_model.replace('/','_')}_train_sae.log"
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
    logger.info(f"Loaded {len(all_doc_embeddings_np)} document embeddings of dimension {all_doc_embeddings_np.shape[1]}")
    
    # Convert to torch tensor
    device = cfg.model.init.device
    embeddings_tensor = torch.from_numpy(all_doc_embeddings_np).float().to(device)
    
    # SAE configuration
    embedding_dim = embeddings_tensor.shape[1]
    nb_concepts = cfg.sae.nb_concepts if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'nb_concepts') else embedding_dim * 2
    top_k = cfg.sae.top_k if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'top_k') and cfg.sae.top_k is not None else None
    delta = 1.0  # Set delta = 1 as requested
    encoder_module = cfg.sae.encoder_module if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'encoder_module') and cfg.sae.encoder_module is not None else None
    num_archetypal_points = cfg.sae.num_archetypal_points if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'num_archetypal_points') and cfg.sae.num_archetypal_points is not None else nb_concepts
    
    # Training configuration
    batch_size = cfg.sae.batch_size if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'batch_size') else cfg.training.batch_size
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
    sae_model = TopKRAESAE(
        input_shape=embedding_dim,
        nb_concepts=nb_concepts,
        points=archetypal_points,
        top_k=top_k,
        encoder_module=encoder_module,
        delta=delta,
        device=device
    )
    sae_model = sae_model.to(device)
    
    # Create data loader
    dataset = TensorDataset(embeddings_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(sae_model.parameters(), lr=learning_rate)
    
    # Use overcomplete's mse_l1 loss function
    # mse_l1 signature: (x, x_hat, pre_codes, codes, dictionary, penalty=1.0)
    sparsity_weight = cfg.sae.sparsity_weight if hasattr(cfg, 'sae') and hasattr(cfg.sae, 'sparsity_weight') else 0.01
    criterion = lambda x, x_hat, pre_codes, codes, dictionary: baseline_criterion(x, x_hat, pre_codes, codes, dictionary)
    
    # Train using overcomplete's training function
    logger.info("Starting training using overcomplete's train_sae function...")
    logs = train_sae(
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
    
    # Report metrics
    logger.info("=" * 80)
    logger.info("Training Metrics Summary:")
    logger.info("=" * 80)
    
    if 'avg_loss' in logs and len(logs['avg_loss']) > 0:
        final_loss = logs['avg_loss'][-1]
        initial_loss = logs['avg_loss'][0]
        logger.info(f"Final Average Loss: {final_loss:.6f}")
        logger.info(f"Initial Average Loss: {initial_loss:.6f}")
        logger.info(f"Loss Reduction: {initial_loss - final_loss:.6f} ({((initial_loss - final_loss) / initial_loss * 100):.2f}%)")
    
    if 'r2' in logs and len(logs['r2']) > 0:
        final_r2 = logs['r2'][-1]
        logger.info(f"Final R2 Score: {final_r2:.6f}")
    
    if 'z_sparsity' in logs and len(logs['z_sparsity']) > 0:
        final_sparsity = logs['z_sparsity'][-1]
        logger.info(f"Final L0 Sparsity: {final_sparsity:.2f}")
    
    if 'dead_features' in logs and len(logs['dead_features']) > 0:
        final_dead_ratio = logs['dead_features'][-1]
        logger.info(f"Final Dead Features Ratio: {final_dead_ratio*100:.2f}%")
    
    if 'time_epoch' in logs:
        total_time = sum(logs['time_epoch'])
        logger.info(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    logger.info("=" * 80)
    
    # Save the trained model
    model_save_path = os.path.join(
        cfg.dataset.model_dir,
        f"sae_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.pt"
    )
    torch.save(sae_model.state_dict(), model_save_path)
    logger.info(f"Saved SAE model to {model_save_path}")
    
    # Save archetypal points
    points_save_path = os.path.join(
        cfg.dataset.model_dir,
        f"sae_archetypal_points_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.pt"
    )
    torch.save(archetypal_points, points_save_path)
    logger.info(f"Saved archetypal points to {points_save_path}")
    
    # Save training logs
    logs_save_path = os.path.join(
        cfg.dataset.logs_dir,
        f"sae_training_logs_{cfg.model.init.save_model}_experts{cfg.model.adapters.num_experts}_concepts{nb_concepts}.json"
    )
    # Convert logs to JSON-serializable format
    logs_serializable = {k: [float(v) if isinstance(v, (torch.Tensor, np.number)) else v for v in v_list] 
                         for k, v_list in logs.items()}
    with open(logs_save_path, 'w') as f:
        json.dump(logs_serializable, f, indent=2)
    logger.info(f"Saved training logs to {logs_save_path}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()
