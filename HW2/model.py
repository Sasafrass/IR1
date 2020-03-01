import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNet(nn.Module):
    def __init__(self, vocab_size, out_dimension=200):
        super(SkipGramNet, self).__init__()
        self.vocab_size = vocab_size
        self.out_dimension = out_dimension
        self.w1 = nn.Embedding(vocab_size, out_dimension, sparse=True)
        self.w2 = nn.Embedding(vocab_size, out_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.out_dimension
        self.w1.weight.data.uniform_(-initrange, initrange)
        self.w2.weight.data.uniform_(0, 0)

    def forward(self, context_words, targets, negative_samples):

        emb_u = self.w1(context_words)
        emb_v = self.w2(targets)
        emb_neg = self.w2(negative_samples)

        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))
