# Numerical imports
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class RankNet(nn.Module):
    #     Initialize network
    def __init__(self, input_size, w1_size = 512, w2_size = 256, w3_size = 128, output_size = 5):
        super(RankNet, self).__init__()

        self.w1 = nn.Linear(input_size, w1_size)
        self.w2 = nn.Linear(w1_size, w2_size)
        self.w3 = nn.Linear(w2_size, w3_size)
        self.w4 = nn.Linear(w3_size, output_size)
        self.relu = nn.LeakyReLU()

    # Forward pass
    def forward(self, x):
      return self.w4(self.relu(self.w3(self.relu(self.w2(self.relu(self.w1(x)))))))