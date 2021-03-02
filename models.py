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

        left = F.logsigmoid(vi_embed @ vo_embed)

        neg_samples_embed = neg_samples @ self.U
        right = torch.sum(F.logsigmoid(-1*(neg_samples_embed @ vi_embed)))

        return -1*(left + right)

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

        # Get context embeddings for each context word and
        # average them to one context embedding vector
        vi_embed = torch.matmul(self.U.T, vi).mean(dim=1)

        left = F.logsigmoid(vi_embed @ vo_embed)

        neg_samples_embed = neg_samples @ self.U
        right = torch.sum(F.logsigmoid(-1*(neg_samples_embed @ vi_embed)))

        return -1*(left + right)

    def get_trained_parameters(self):
        return (self.V + self.U) / 2

