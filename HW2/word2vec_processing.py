import pandas as pd
import json
import read_ap
import numpy as np


def doc2vector(doc, embedding_dict):
    stack = []
    for word in doc:
        if word in embedding_dict:
            stack.append(embedding_dict[word])
        else:
            stack.append(embedding_dict['<UNK>'])
    stack = np.array(stack)
    return np.mean(stack, axis=0)

def get_embedding_dict(path):
    embedding = pd.read_csv('path', sep=" ", header=None, index_col=0)
    return embedding

def get_word_embedding(path, word):
    embedding = pd.read_csv('path', sep=" ", header=None, index_col=0)
    return embedding[word]

def cosine_similarity(a,b):
    return dot(a, b)/(norm(a)*norm(b))


#embedding = pd.read_csv('word_vectors.csv', sep=" ", header=None, index_col=0)
#
#print(embedding)
#embedding_dict = embedding.T.to_dict('list')
#docs = read_ap.get_processed_docs()
#
#vect = doc2vector(docs['AP891231-0014'], embedding_dict)
#print('hello')
