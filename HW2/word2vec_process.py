import pandas as pd
import json
import read_ap
import numpy as np
import os

import pytrec_eval

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
    return embedding

def get_word_embedding(path, word):
    embedding = pd.read_csv(path, sep=" ", header=None, index_col=0)
    return embedding[word]

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))



if __name__ == "__main__":
    embedding = get_embedding_dict('tempemb5_200.csv')
    embedding_dict = embedding.T.to_dict('list')

    #get docs and queries
    docs = read_ap.get_processed_docs()
    qrels, queries = read_ap.read_qrels()

    doc_keys = list(docs.keys())
    q_keys = list(queries.keys())
    doc_vects = []
    q_vects = []

    #transform queries and documents into embedding space
    print('preprocessing queries')
    for key in q_keys:
        q_vects.append(doc2vector(queries[key], embedding_dict))
    print('preprocessing documents')
    for i, key in enumerate(doc_keys):
        doc_vects.append(doc2vector(docs[key], embedding_dict))
        if((i+1)%10000 == 0):
            print(str(i+1))

    results = {}
    
    print('generating results')
    for i, query in enumerate(q_vects):
        result = {}
        print(q_keys[i])
        for j, doc in enumerate(doc_vects):
            result[doc_keys[j]] = np.float64(cosine_similarity(doc,query))
        result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
        results[q_keys[i]] = result
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    print('evaluating results')
    metrics = evaluator.evaluate(results)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open("word2vec.json", "w") as writer:
        json.dump(metrics, writer, indent=1)
    
    print('writing results')
    f= open("word2vec_trec.txt","w+")
    for i, query in enumerate(q_keys):
        ordered_doc_names = list(results[query].keys())
        for j, doc in enumerate(ordered_doc_names):
            if(j == 1000):
                break
            f.write(query + ' Q0 ' + doc + " " + str(j+1) + " " + str(results[query][doc]) + ' STANDARD\n')

