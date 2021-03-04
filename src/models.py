from numpy import left_shift
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SkipGram(nn.Module):
    """
    Skip-gram model with negative sampling.
    """

    def __init__(self, voc_size, embed_dim):
        super(SkipGram, self).__init__()

        # Center word embeddings
        self.V = Parameter(torch.randn(voc_size, embed_dim))

        # Context word embeddings
        self.U = Parameter(torch.randn(voc_size, embed_dim))

    def forward(self, vi, vo, neg_samples):
        vi_embed = vi @ self.V
        vo_embed = vo @ self.U

        left = F.logsigmoid(torch.sum(vi_embed * vo_embed, dim=1, keepdim=True))

        neg_samples_embed = torch.matmul(neg_samples, self.U)
        batch_mul = torch.bmm(neg_samples_embed, vi_embed.unsqueeze(dim=2)).squeeze()
        right = torch.sum(F.logsigmoid(-1*batch_mul), dim=1, keepdim=True)

        return torch.mean(-1*(left + right))

    def get_trained_parameters(self):
       return (self.V + self.U) / 2 


class CBOW(nn.Module):
    """
    Continous bag-of-words model with negative sampling.
    """

    def __init__(self, voc_size, embed_dim):
        super(CBOW, self).__init__()

        # Center word embeddings
        self.V = Parameter(torch.randn(voc_size, embed_dim))
        # Context word embeddings
        self.U = Parameter(torch.randn(voc_size, embed_dim))

    def forward(self, vo, vi, neg_samples):
        vo_embed = vo @ self.V

        print(f"vo_embed: {vo_embed.shape}")

        # Get context embeddings for each context word and
        # average them to one context embedding vector
        vi_embed = torch.matmul(vi, self.U).mean(dim=1)
        print(f"vi: {vi.shape}")
        print(f"vi_embed: {vi_embed.shape}")

        left = F.logsigmoid(torch.sum(vo_embed * vi_embed, dim=1, keepdim=True))

        print(f"left: {left.shape}")

        neg_samples_embed = torch.matmul(neg_samples, self.U)
        print(f"neg_samples_embed: {neg_samples_embed.shape}")
        batch_mul = torch.bmm(neg_samples_embed, vo_embed.unsqueeze(dim=2)).squeeze()
        print(f"batch_mul: {batch_mul.shape}")
        right = torch.sum(F.logsigmoid(-1*batch_mul), dim=1, keepdim=True)
        print(f"right: {right.shape}")

        return torch.mean(-1*(left + right))

    def get_trained_parameters(self):
        return (self.V + self.U) / 2

