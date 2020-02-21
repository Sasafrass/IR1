import os
import pickle as pkl
import json

# np and torch stuff
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

# To read the files and qrels
import read_ap
import download_ap

# import file with model
from model import SkipGramNet
from helper import build_dicts

# Actual SkipGram function
def SkipGram(model, num_epochs, window_size, docs_by_id, w2i, i2w, ce, optimizer):

    for epoch in range(num_epochs):
        run_epoch(model, window_size, docs_by_id, w2i, i2w, ce, optimizer)

def run_epoch(model, window_size, docs_by_id, w2i, i2w, ce, optimizer):
    len_vocab = len(w2i)

    for key, doc in docs_by_id.items():
        for i in range(2, len(doc)-2):
            context = one_hot(torch.tensor(w2i[doc[i]]), len_vocab)
            context = context.float()
            targets = [torch.tensor(w2i[doc[i-2]]), 
                                            torch.tensor(w2i[doc[i-1]]), 
                                            torch.tensor(w2i[doc[i+1]]), 
                                            torch.tensor(w2i[doc[i+2]])]

            out = model.forward(context)

            # Do window_size 2 times and accumulate loss
            loss = 0
            for i in range(window_size * 2):
                loss += ce(out.unsqueeze(0), targets[i].unsqueeze(0))

            # Feed loss backward and zero grad the boii
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":

    # Make sure we have the dataset
    download_ap.download_dataset()

    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # get qrels
    qrels, queries = read_ap.read_qrels()

    # Load files from json if dictionaries already exist == faster
    if os.path.isfile("w2i.json") and os.path.isfile("i2w.json"):
        # load
        w2i = json.load(open("w2i.json","r"))
        i2w = json.load(open("i2w.json","r"))
    # Otherwise build the w2i and i2w dictionaries
    else: 
        w2i, i2w = build_dicts(docs_by_id)

    # Variables to give to skip gram function, window size on each side
    # Output dimension of vector
    num_epochs  = 1
    window_size = 2
    output_dim  = 300
    vocab_size  = len(w2i)
    model = SkipGramNet(vocab_size, output_dim).float()

    # lr, loss function and optimizer
    learning_rate = 0.01
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Perform actual SkipGram training
    SkipGram(model, num_epochs, window_size, docs_by_id, w2i, i2w, ce, optimizer)