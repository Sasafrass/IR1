import os

import json
import pickle as pkl
from collections import defaultdict, Counter

import numpy as np
import pytrec_eval
from tqdm import tqdm

from gensim import models
from gensim.test.utils import get_tmpfile
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim import similarities

# To read the files and qrels
import read_ap
import download_ap


class LSIRetrieval():
    def __init__(self, docs, topic_number=500):
        # Create a dictionary representation of the documents.
        if(not os.path.isfile('./lsi/lsi_dict.dict')):
            print('creating dict')
            dictionary = Dictionary(docs)
            dictionary.save('./lsi/lsi_dict.dict')
        else:
            print('dict already exists')
            dictionary = Dictionary.load("./lsi/lsi_dict.dict")
        self.dictionary = dictionary

        # Create corpora
        if(not os.path.isfile('./lsi/lsi_corpus.mm')):
            # Filter out words that occur less than 20 documents, or more than 50% of the documents.
            print('creating bow corpus')
            dictionary.filter_extremes(no_below=20, no_above=0.5)
            corpus = [dictionary.doc2bow(doc) for doc in docs]
            MmCorpus.serialize("lsi/lsi_corpus.mm", corpus)
        else:
            print('bow corpus already exists')
            corpus = MmCorpus("./lsi/lsi_corpus.mm")
        self.corpus = corpus

        if(not os.path.isfile('./lsi/lsi_tf_corpus.mm')):
            print('creating tf corpus')
            tfidf = models.TfidfModel(corpus)
            tf_corp = tfidf[corpus]
            MmCorpus.serialize("lsi/lsi_tf_corpus.mm", tf_corp)
        else:
            print('tf corpus already exists')
            tf_corp = MmCorpus("./lsi/lsi_tf_corpus.mm")

        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        #Create the models and vectors
        if(not os.path.isfile('./lsi/lsi_bow_model.model')):
            print('creating bow model')
            bow_model = models.LsiModel(corpus=corpus, num_topics = topic_number, id2word=id2word)
            bow_model.save('lsi/lsi_bow_model.model')
        else:
            print('bow model already exists')
            bow_model = models.LsiModel.load('./lsi/lsi_bow_model.model')
        bow_vector = bow_model[corpus]
        self.bow_model = bow_model

        if(not os.path.isfile('./lsi/lsi_tf_model.model')):
            print('creating tfidf model')
            tf_model = models.LsiModel(corpus=tf_corp, num_topics = num_topics, id2word=id2word)
            tf_model.save('./lsi/lsi_tf_model.model')
        else:
            print('tfidf model already exists')
            tf_model = models.LsiModel.load('./lsi/lsi_tf_model.model')
        tf_vector = tf_model[tf_corp]
        self.tf_model = tf_model

        #Create indices
        if(not os.path.isfile('./lsi/lsi_bow_model.index')):
            print('creating bow index')
            bow_index = similarities.MatrixSimilarity(bow_vector)  # index corpus in bow LSI space
            bow_index.save('lsi/lsi_bow_model.index')
        else:
            print('bow index already exists')
            bow_index = similarities.MatrixSimilarity.load('./lsi/lsi_bow_model.index')
        self.bow_index = bow_index

        if(not os.path.isfile('./lsi/lsi_tf_model.index')):
            print('creating tf index')
            tf_index = similarities.MatrixSimilarity(tf_vector)  # index corpus in tf LSI space
            tf_index.save('lsi/lsi_tf_model.index')
        else:
            print('tf index already exists')
            tf_index = similarities.MatrixSimilarity.load('./lsi/lsi_tf_model.index')
        self.tf_index = tf_index
        print('model created!')

    def search(self, queries, docs_by_id):
        
        print('preprocessing queries')
        query_ids = list(queries.keys())
        #Transform queries into the LSI spaces
        q_bow_vec = [self.dictionary.doc2bow(q.lower().split()) for q in queries]
        bow_queries = self.bow_model[q_bow_vec]    

        q_tfidf = models.TfidfModel(self.corpus)    #create a tfidf model based on the corpus
        q_tf_vec = q_tfidf[q_bow_vec]               #transform the bow query vectors to tfidf vectors
        tf_queries = self.tf_model[q_tf_vec]        #transform the tfidf vectors to vectors in the LSI model's space
        
        #Find documents that match the queries
        bow_results = {}
        tf_results = {}
        
        print('generating bow results')
        for i, query in enumerate(bow_queries):
            results = {}
            bow_sims = self.bow_index[query]
            keys = list(docs_by_id.keys())
            for j in range(len(keys)):
                results[keys[j]] = bow_sims[j]
            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
            bow_results[query_ids[i]] = results

        print('generating tf results')
        for i, query in enumerate(tf_queries):
            results = {}
            tf_sims = self.tf_index[query]
            keys = list(docs_by_id.keys())
            for j in range(len(keys)):
                results[keys[j]] = np.float64(tf_sims[j])
            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
            tf_results[query_ids[i]] = results
        print(tf_results['101']['AP880222-0073'])
        print(type(tf_results['101']['AP880222-0073']))
        return bow_results, tf_results

if __name__ == "__main__":
    # Make sure we have the dataset
    download_ap.download_dataset()

    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # get qrels
    qrels, queries = read_ap.read_qrels()

    # Bag-of-words representation of the documents.
    docs = [doc for key, doc in docs_by_id.items()]

    lsi_search = LSIRetrieval(docs)

    bow_results, tf_results = lsi_search.search(queries, docs_by_id)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    print('evaluating bow results')
    metrics = evaluator.evaluate(bow_results)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open("lsi_bow.json", "w") as writer:
        json.dump(metrics, writer, indent=1)

    print('evaluating tf results')
    metrics = evaluator.evaluate(tf_results)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open("lsi_tf.json", "w") as writer:
        json.dump(metrics, writer, indent=1)
