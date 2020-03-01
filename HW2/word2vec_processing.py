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
    embedding = pd.read_csv(path, sep=" ", header=None, index_col=0)
    embedding_dict = embedding.T.to_dict('list')
    return embedding_dict

def get_word_embedding(path, word):
    embedding = pd.read_csv(path, sep=" ", header=None, index_col=0)
    embedding = embedding.T.to_dict('list')
    return embedding[word]

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def find_n_similar_words(word, n, path):
    word = read_ap.process_text(word)    
    word = word[0]
    result = []
    emb_dict = get_embedding_dict(path)
    for idx, value in enumerate(emb_dict):
        similarity = cosine_similarity(emb_dict[word], emb_dict[value])
        result.append((value, similarity))
    result.sort(key=lambda tup: tup[1], reverse=True)
    return result[0:n]

def find_n_relevant_docs(query, docs, n, path):
    doc_vectors = []
    query = read_ap.process_text(query)

    embedding_dict = get_embedding_dict(path)
    query_vector = doc2vector(query, embedding_dict)
    for key in docs:
        doc_vector = doc2vector(docs[key], embedding_dict)
        doc_vectors.append((key, doc_vector))
    
    similaries = []
    for vector in doc_vectors:
        similarity = cosine_similarity(vector[1], query_vector)
        similaries.append((vector[0], similarity))
    similaries.sort(key=lambda tub: tub[1], reverse=True)
    return similaries[0:n]
