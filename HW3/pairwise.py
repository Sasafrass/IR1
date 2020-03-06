# Numerical imports
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# import the models
from models import RankNet
from dataset import get_dataset, DataSet

def run_epoch(model, data):
    print("je moeder")


if __name__ == "__main__":

    # Get data
    dataset = get_dataset()
    data = dataset.get_data_folds()[0]
    print(data.num_features)
    data.read_data()

    # data.train is a DataFoldSplit
    print("data.train:", data.train)
    print("data.train:", data.train.num_queries())

    # Parameters for model
    input_size  = "jemoeder"
    output_size = 5

    # Define model
    model = RankNet(input_size = input_size, output_size = output_size)

    # Define what split we are using FOR THE MODEL SO TRAIN OR NOT
    split = "train" #"validation", "test"
    print(f"Split: {split}")
    split = getattr(data, split)
    print(f"\tNumber of queries {split.num_queries()}")

    # Define number of epochs and run for that amount
    num_epochs = 1
    for i in range(num_epochs):
        run_epoch(model)