from torch import nn
import torch
import numpy as np

import pe as PE
import nn as NN

def get_positional_encoding(positional_encoding_type, **kwargs):
    """
    Returns a positional encoding module based on the specified encoding type.
    
    Args:
        encoding_type (str): The type of positional encoding to use. Options are 'rff', 'siren', 'sh', 'capsule'.
        input_dim (int): The input dimension for the positional encoding.
        output_dim (int): The output dimension for the positional encoding.
        **kwargs: Additional arguments for specific encoding types.
        
    Returns:
        nn.Module: The positional encoding module.
    """
    if positional_encoding_type == "rff":
        return PE.ProjectionRFF(**kwargs)
    elif positional_encoding_type == "sh":
        return PE.SphericalHarmonics(**kwargs)
    else:
        raise ValueError(f"Unsupported encoding type: {positional_encoding_type}")
    
def get_neural_network(neural_network_type, input_dim, **kwargs):
    """
    Returns a neural network module based on the specified network type.
    
    Args:
        neural_network_type (str): The type of neural network to use. Options are 'siren'.
        input_dim (int): The input dimension for the neural network.
        output_dim (int): The output dimension for the neural network.
        **kwargs: Additional arguments for specific network types.
        
    Returns:
        nn.Module: The neural network module.
    """
    if neural_network_type == "siren":
        return NN.SirenNet(
            input_dim=input_dim,
            **kwargs
        )
    elif neural_network_type == "mlp":
        return NN.MLP(
            input_dim=input_dim, 
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported network type: {neural_network_type}")
    

class LocationEncoder(nn.Module):
    def __init__(self, position_encoding_type="rff", neural_network_type="mlp", **kwargs):
        super().__init__()

        self.position_encoder = get_positional_encoding(
            position_encoding_type, **kwargs
        )

        self.neural_network = get_neural_network(
            neural_network_type, 
            input_dim=self.position_encoder.embedding_dim, 
            **kwargs
        )

        self.neural_network = nn.ModuleList([
            get_neural_network(
                neural_network_type,
                input_dim=dim,
                **kwargs
            ) for dim in self.position_encoder.embedding_dims
        ])

    def forward(self, x):
        embedding = self.position_encoder(x)
        
        if embedding.ndim == 2:
            # If the embedding is 2D, we need to add a dimension
            embedding = embedding.unsqueeze(0)

        location_features = torch.zeros(embedding.shape[1], 512).to(input.device)

        for nn, e in zip(self.neural_network, embedding):
            location_features += nn(e)

        return location_features