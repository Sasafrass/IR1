# Import torch-y stuff
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SkipGramNet(nn.Module):
    def __init__(self, vocab_size, output_dim):
        super(SkipGramNet, self).__init__()

        self.W1 = nn.Linear(vocab_size, output_dim)
        self.W2 = nn.Linear(output_dim, vocab_size)

    def forward(self, input):
        out = self.W1(input)
        out = self.W2(out)

        return out