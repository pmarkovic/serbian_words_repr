import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SkipGram(nn.Module):
    """
    Skip-gram model with negative sampling.
    """

    def __init__(self, voc_size, embed_dim):
        super(SkipGram, self).__init__()

        # Center word embeddings
        self.V = Parameter(torch.FloatTensor(voc_size, embed_dim).uniform_(-0.5, 0.5) / embed_dim)

        # Context word embeddings
        self.U = Parameter(torch.FloatTensor(voc_size, embed_dim).uniform_(-0.5, 0.5) / embed_dim)

    def forward(self, vi, vo, neg_samples):
        # Obtain corresponding embeddings from matrices (V for center, U for context)
        vi_embed = vi @ self.V
        vo_embed = vo @ self.U

        # Dot product between center and context embeddings for every example
        # And calculate logsigmoid
        left = F.logsigmoid(torch.sum(vi_embed * vo_embed, dim=1, keepdim=True))

        # Calculating dot product between neg samples and center embedding for every example
        # And calculate logsigmoid for loss
        neg_samples_embed = torch.matmul(neg_samples, self.U)
        batch_mul = torch.bmm(neg_samples_embed, vi_embed.unsqueeze(dim=2)).squeeze()
        right = torch.sum(F.logsigmoid(-1*batch_mul), dim=1, keepdim=True)

        # Final loss and mean over all examples
        return torch.mean(-1*(left + right))

    def get_trained_parameters(self):
       return (self.V + self.U) / 2 


class CBOW(nn.Module):
    """
    Continous bag-of-words model with negative sampling.
    """

    def __init__(self, voc_size, embed_dim):
        super(CBOW, self).__init__()
        self.embed_dim = embed_dim

        # Center word embeddings
        self.V = Parameter(torch.FloatTensor(voc_size, embed_dim).uniform_(-0.5, 0.5) / embed_dim)
        # Context word embeddings
        self.U = Parameter(torch.FloatTensor(voc_size, embed_dim).uniform_(-0.5, 0.5) / embed_dim)

    def forward(self, vo, vi, neg_samples):
        # Obtain embeddings from V (matrix for center embeddings)
        vo_embed = vo @ self.V

        # Calculate context embeddings for every example in batch
        # One example in batch consists of multiple embeddings

        # Obtain embeddings from U (matrix for context embeddings)
        vi_temp = torch.matmul(vi, self.U)
        # Mask: for every example in batch store True for every embedding which is non-zero
        mask = vi.abs().sum(dim=2) != 0
        # Calculate how many non-zero embeddings per example
        mask_sum = mask.float().sum(dim=1).unsqueeze(1)
        # Calculate mean of all context embeddings per example in batch
        # Had to use mask and calculate mean this way due to different number of 
        # embeddings per examples (center words doesn't have the same number of context words)
        vi_embed = vi_temp.sum(dim=1) / mask_sum.expand(mask_sum.size(0), self.embed_dim)

        # Dot product between center and mean context embeddings for every example
        # And calculate logsigmoid for loss
        left = F.logsigmoid(torch.sum(vo_embed * vi_embed, dim=1, keepdim=True))

        # Calculating dot product between neg samples and center embedding for every example
        # And calculate logsigmoid for loss
        neg_samples_embed = torch.matmul(neg_samples, self.U)
        batch_mul = torch.bmm(neg_samples_embed, vo_embed.unsqueeze(dim=2)).squeeze()
        right = torch.sum(F.logsigmoid(-1*batch_mul), dim=1, keepdim=True)

        # Final loss and mean over all examples
        return torch.mean(-1*(left + right))

    def get_trained_parameters(self):
        return (self.V + self.U) / 2

