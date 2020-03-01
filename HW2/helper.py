import os
import json
import numpy
import torch
from torch.nn.functional import one_hot

def build_dicts(docs_by_id):
    w2i = {}
    i2w = {}
    counts = {}
    vocab = set()

    # Run over all tokens
    for key, doc in docs_by_id.items():
        for token in doc:
            if token not in vocab:
                vocab.add(token)
                counts[token] = 1
            else:
                counts[token] += 1
    
    vocab = set()
    index = 1
    w2i['<UNK>'] = 0
    i2w[0] = '<UNK>'
    for word, count in counts.items():
        if(count >= 50 ):
            w2i[word] = index
            i2w[index] = word
            index += 1
            vocab.add(word)

    #for word, index in w2i.items():
        #w2i[word] = one_hot(torch.tensor(index), len(vocab))
        # print(word)
        # print(w2i[word])
        # print(len(w2i[word]))
        # print(len(vocab))

    jsonw2i = json.dumps(w2i)
    jsoni2w = json.dumps(i2w)

    f = open("w2i.json","w")
    f.write(jsonw2i)
    f.close()
    
    f = open("i2w.json","w")
    f.write(jsoni2w)
    f.close()

    return w2i, i2w
