import torch
from torch import nn
from sklearn.cluster import KMeans

from overcomplete import EncoderFactory
from overcomplete.base import BaseDictionaryLearning
from overcomplete.sae.base import SAE
from overcomplete.sae import TopKSAE, RelaxedArchetypalDictionary
from einops import rearrange, reduce


class TopKRAESAE(TopKSAE):
    """
    Top-K Relaxed Archetypal Sparse Autoencoder (TopKRAESAE)
    
    This class combines TopKSAE with a RelaxedArchetypalDictionary.
    It uses top-k sparsity on the encoder output and an archetypal dictionary
    initialized from given points.
    """
    def __init__(self, input_shape, nb_concepts, points,
                 top_k=None, encoder_module=None, delta=1.0, device='cpu'):
        """
        Initialize TopKRAESAE
        
        Args:
            input_shape: Shape of input data (int for 1D, tuple for higher dims)
            nb_concepts: Number of concepts/dictionary atoms
            points: Tensor of points to initialize archetypal dictionary [N, dim]
            top_k: Number of top activations to keep (default: nb_concepts // 10)
            encoder_module: Encoder module (str name or module instance)
            delta: Delta parameter for RelaxedArchetypalDictionary
            device: Device to run on ('cpu' or 'cuda')
        """
        # Initialize TopKSAE parent class - we'll override the dictionary later
        # First initialize without dictionary, then replace it
        super().__init__(
            input_shape=input_shape,
            nb_concepts=nb_concepts,
            top_k=top_k,
            encoder_module=encoder_module,
            device=device
        )

        self.delta = delta

        # Replace dictionary with RelaxedArchetypalDictionary using given points
        in_dim = points.shape[-1]
        self.dictionary = RelaxedArchetypalDictionary(
            in_dimensions=in_dim,
            nb_concepts=nb_concepts,
            points=points,
            delta=delta,
            device=device
        )
