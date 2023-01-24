import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.verbose = arg.verbose
        self.num_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embeddding(self.num_vectorsm self.latent_dim)
        self.embedding.weight.data.normal_()

        self.
