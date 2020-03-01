import os
import pickle as pkl
import json
from random import randrange
# np and torch stuff
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
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
        total_loss = 0
        for i in range(window_size, len(doc)-window_size):
            context_word = torch.tensor(w2i[doc[i]]) if doc[i] in w2i else torch.tensor(0)
            context = one_hot(context_word, len_vocab)
            context = context.float()
            targets = [torch.tensor(0) for i in range(window_size * 2)]
            for j in range(1, window_size + 1):
                targets[window_size-j] = torch.tensor(0) if doc[i-j] not in w2i else torch.tensor(w2i[doc[i-j]])
                targets[window_size+j -1] = torch.tensor(0) if doc[i+j] not in w2i else torch.tensor(w2i[doc[i+j]])
            
            negative_samples = []

            #generate 5 negative samples for every word combination
            for j in range(len(targets)):
                negative_samples.append([randrange(len_vocab) for x in range(5)])

            pos_samples = [context_word for x in range(len(targets))]            


            optimizer.zero_grad()
            loss = model.forward(
                Variable(torch.LongTensor(pos_samples)),
                Variable(torch.LongTensor(targets)),
                Variable(torch.LongTensor(negative_samples))
            )
            # Feed loss backward and zero grad the boii
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.save_embedding(i2w, filename+'.csv')
        print('loss is: ', total_loss/i)
    model.save_embedding(i2w, filename+'final.csv')

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
    window_size = 10
    output_dim  = 300
    vocab_size  = len(w2i)
    filename = '10300'
    model = SkipGramNet(vocab_size, output_dim).float()

    # lr, loss function and optimizer
    learning_rate = 0.01
    ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)

    # Perform actual SkipGram training
    SkipGram(model, num_epochs, window_size, docs_by_id, w2i, i2w, ce, optimizer)