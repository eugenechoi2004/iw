import torch
import torch.nn as nn
import torch.nn.functional as F
import hypll.nn as hnn
from hypll.tensors import TangentTensor



class EncoderMLP(nn.Module):
    def __init__(self, num_cat, embedding_dim, hidden_dims, output_dim):
        super(EncoderMLP, self).__init__()
        self.embedding = nn.Embedding(num_cat, embedding_dim)
        input_dim = embedding_dim * 2
        model = []
        for dim in hidden_dims:
            model.append(nn.Linear(input_dim, dim))
            model.append(nn.ReLU())
            input_dim = dim
        model.append(nn.Linear(input_dim, output_dim))
        self.encoder = nn.Sequential(*model)
    
    def forward(self, first, second):
        first_embed = self.embedding(first)
        second_embed = self.embedding(second)
        concat_embed = torch.cat((first_embed, second_embed), dim = 1)
        return self.encoder(concat_embed)

class EncoderHyperbolicMLP(nn.Module):
    def __init__(self, cat_features, embedding_dims, euc_hidden_dims, hyp_hidden_dims, output_dim, manifold):
        super(EncoderHyperbolicMLP, self).__init__()
        
        self.manifold = manifold

        self.euc_mlp = EncoderMLP(cat_features, embedding_dims, euc_hidden_dims, hyp_hidden_dims[0])
        
        # Hyperbolic layers
        hyp_layers = []
        for i in range(1, len(hyp_hidden_dims)):
            hyp_layers.append(hnn.HLinear(hyp_hidden_dims[i-1], hyp_hidden_dims[i], manifold=manifold))
            hyp_layers.append(hnn.HReLU(manifold=manifold))
        
        hyp_layers.append(hnn.HLinear(hyp_hidden_dims[-1], output_dim, manifold=manifold))
        
        self.hyp_mlp = nn.Sequential(*hyp_layers)
    
    def forward(self, first, second):
        # Pass through Euclidean layers
        euc_output = self.euc_mlp(first, second)
        
        # Map to hyperbolic space
        hyp_input = self.manifold_map(euc_output, self.manifold)
        
        # Pass through Hyperbolic layers
        output = self.hyp_mlp(hyp_input)
        
        return output
    
    def manifold_map(self, x, manifold):
        """
        Maps a tensor in Euclidean space onto a Riemannian Manifold
        """
        tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
        return manifold.expmap(tangents)
