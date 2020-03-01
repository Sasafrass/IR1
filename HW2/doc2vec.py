# File managing imports
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

# Model and helper imports
from model import SkipGramNet
from helper import build_dicts

# Imports specific to doc2vec
import gensim
from gensim.test.utils import get_tmpfile

# Pytrec eval
import pytrec_eval
from tqdm import tqdm

# Function to read a corpus gensim-wise
def read_corpus(docs_by_id, tokens_only = False):
        for key, doc in docs_by_id.items():
            if tokens_only:
                yield doc
            else:
                yield gensim.models.doc2vec.TaggedDocument(doc, [key])

# Ranking function to return all results
def ranking(query, model, num_results = 10):
    query = read_ap.process_text(query)
    vec = model.infer_vector(query)
    
    return model.docvecs.most_similar([vec], topn = num_results)

# Main function to run doc2vec
if __name__ == "__main__":

    # Make sure we have the dataset
    download_ap.download_dataset()

    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # get qrels
    qrels, queries = read_ap.read_qrels()

    # 
    overall_ser = {}

    # # Load files from json if dictionaries already exist == faster
    # if os.path.isfile("w2i.json") and os.path.isfile("i2w.json"):
    #     # load
    #     w2i = json.load(open("w2i.json","r"))
    #     i2w = json.load(open("i2w.json","r"))
    # # Otherwise build the w2i and i2w dictionaries
    # else: 
    #     w2i, i2w = build_dicts(docs_by_id)

    # No trained model, so re-train the model
    if not os.path.isfile("d2v_300_25_10.model"):
        # Retrieve training corpus and "test corpus"
        train_corpus = list(read_corpus(docs_by_id))
        test_corpus  = list(read_corpus(docs_by_id, tokens_only = True))

        # Initialize model
        model = gensim.models.doc2vec.Doc2Vec(vector_size = 50, min_count = 2, epochs = 40)
        model.build_vocab(train_corpus)

        # Train model
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        # Save model
        path = ""
        model.save(path + "d2v.model")

    # Load model from memory
    else:
        model = gensim.models.doc2vec.Doc2Vec.load("d2v_300_25_10.model")

    # Set evaluation bool, and number of displayed results to be equal to len(dataset)
    evaluate = True
    num_results = 164557

    if evaluate:
        for qid in tqdm(qrels): 
            query = queries[qid]

            # results is ordered list of tuples
            results = ranking(query, model = model, num_results = num_results)
            overall_ser[qid] = dict(results)

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
        metrics = evaluator.evaluate(overall_ser)

        # dump this to JSON
        # *Not* Optional - This is submitted in the assignment!
        with open("d2v_300_25_10.json", "w") as writer:
            json.dump(metrics, writer, indent=1)


