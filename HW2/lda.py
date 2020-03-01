# File managing imports
import os
import sys
import pickle as pkl
import json

# np and torch stuff
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot

# Imports specific to doc2vec and NLP
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.matutils import kullback_leibler
import nltk

# To read the files and qrels
import read_ap
import download_ap

# Model and helper imports
from model import SkipGramNet
from helper import build_dicts

# pytrec eval
import pytrec_eval
from tqdm import tqdm

# Function that ranks documents given a query 
def ranking_LDA(query, model, model_docs, num_topics = 10):
    scores = []
    
    # Process query to correct KL divergence form
    query = read_ap.process_text(query)
    query = dictionary.doc2bow(query)
    query = model[query]
    query = gensim.matutils.sparse2full(query, num_topics)

    # Calculate KL divergence for each document in the corpus
    for i in range(len(corpus)):
        doc = model_docs[i]
        neg_kl = float(-1 * kullback_leibler(query, doc))
        scores.append((i2str[i], neg_kl))

    # Sort on second tuple value
    scores = sorted(scores, key=lambda x: x[1], reverse = True)
    return scores

# Main function call
if __name__ == "__main__":

    # Make sure we have the dataset
    download_ap.download_dataset()

    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # get qrels
    qrels, queries = read_ap.read_qrels()

    # 
    overall_ser = {}

    # Dictionary containing index to document str name
    i2str = {}
    for i, doc in enumerate(docs_by_id):
        i2str[i] = doc

    # Load from file if we already created it..
    if os.path.isfile("bigram.json"):
        docs = json.load(open("bigram.json","r"))
    
    # .. Otherwise create docs with bigrams
    else:
        # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
        docs = [doc for key, doc in docs_by_id.items()]

        # Do actual appending of bigrams
        bigram = Phrases(docs, min_count=20)
        for i in range(len(docs)):
            for token in bigram[docs[i]]:
                if '_' in token:
                    # Token = bigram -> append to docs
                    docs[i].append(token)

        # Save bigram dictionary
        jsondocs = json.dumps(docs)
        f = open("bigram.json","w")
        f.write(jsondocs)
        f.close()

    # Create a dictionary representation of the documents and filter extremes
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    
    # turn docs into bow representation
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    # Set training parameters
    num_topics = 10
    chunksize = 2000
    passes = 20
    iterations = 1
    eval_every = None

    # Make index 2 word and word 2 index dictionaries
    loader = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    word2id = dictionary.token2id

    # Check if we have a pre-trained model and load it.. 
    if os.path.isfile("lda.model"):
        model = LdaModel.load("lda.model")

    # .. or re-train in case we don't have one
    else:
        # Train LDA model
        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )

        # Save model
        model.save("lda.model")

    # if we also want to evaluate
    evaluate = True

    if evaluate:
        # ranking_lda and shizz
        model_docs = [gensim.matutils.sparse2full(model[corpus[i]], num_topics) for i in range(len(corpus))]

        for qid in tqdm(qrels): 
            query = queries[qid]

            # results is ordered list of tuples
            results = ranking_LDA(query, model = model, 
                                         model_docs = model_docs, 
                                         num_topics = num_topics)
            overall_ser[qid] = dict(results)

            print(qid)

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
        metrics = evaluator.evaluate(overall_ser)

        # dump this to JSON
        # *Not* Optional - This is submitted in the assignment!
        with open("lda.json", "w") as writer:
            json.dump(metrics, writer, indent=1)

    