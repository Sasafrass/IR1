import os

from gensim import models
from gensim.test.utils import get_tmpfile
from gensim.corpora import Dictionary
from gensim.corpora.mmcorpus import MmCorpus
from gensim import similarities

# To read the files and qrels
import read_ap
import download_ap


if __name__ == "__main__":
    # Make sure we have the dataset
    download_ap.download_dataset()

    # pre-process the text
    docs_by_id = read_ap.get_processed_docs()

    # get qrels
    qrels, queries = read_ap.read_qrels()

    # Bag-of-words representation of the documents.
    docs = [doc for key, doc in docs_by_id.items()]

    # Create a dictionary representation of the documents.
    if(not os.path.isfile('lsi\\lsi_dict.dict')):
        print('creating dict')
        dictionary = Dictionary(docs)
        dictionary.save('lsi_dict.dict')
    else:
        print('dict already exists')
        dictionary = Dictionary.load("lsi\\lsi_dict.dict")
        
    # Create corpora
    if(not os.path.isfile('lsi\\lsi_corpus.mm')):
        # Filter out words that occur less than 20 documents, or more than 50% of the documents.
        print('creating bow corpus')
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        corpus = [dictionary.doc2bow(doc) for doc in docs]
        MmCorpus.serialize("lsi\\lsi_corpus.mm", corpus)
    else:
        print('bow corpus already exists')
        corpus = MmCorpus("lsi\\lsi_corpus.mm")
        
    if(not os.path.isfile('lsi\\lsi_tf_corpus.mm')):
        print('creating tf corpus')
        tfidf = models.TfidfModel(corpus)
        tf_corp = tfidf[corpus]
        MmCorpus.serialize("lsi\\lsi_tf_corpus.mm", tf_corp)
    else:
        print('tf corpus already exists')
        tf_corp = MmCorpus("lsi\\lsi_tf_corpus.mm")

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    #Create the models and vectors
    if(not os.path.isfile('lsi\\lsi_bow_model.model')):
        print('creating bow model')
        bow_model = models.LsiModel(corpus=corpus, num_topics = 500, id2word=id2word)
        bow_model.save('lsi\\lsi_bow_model.model')
    else:
        print('bow model already exists')
        bow_model = models.LsiModel.load('lsi\\lsi_bow_model.model')
    bow_vector = bow_model[corpus]
        
    if(not os.path.isfile('lsi\\lsi_tf_model.model')):
        print('creating tfidf model')
        tf_model = models.LsiModel(corpus=tf_corp, num_topics = 500, id2word=id2word)
        tf_model.save('lsi\\lsi_tf_model.model')
    else:
        print('tfidf model already exists')
        tf_model = models.LsiModel.load('lsi\\lsi_tf_model.model')
    tf_vector = tf_model[tf_corp]

    #Create indices
    if(not os.path.isfile('lsi\\lsi_bow_model.index')):
        print('creating bow index')
        bow_index = similarities.MatrixSimilarity(bow_vector)  # index corpus in bow LSI space
        bow_index.save('lsi\\lsi_bow_model.index')
    else:
        print('bow index already exists')
        bow_index = similarities.MatrixSimilarity.load('lsi\\lsi_bow_model.index')
        
    if(not os.path.isfile('lsi\\lsi_tf_model.index')):
        print('creating tf index')
        tf_index = similarities.MatrixSimilarity(tf_vector)  # index corpus in tf LSI space
        tf_index.save('lsi\\lsi_tf_model.index')
    else:
        print('tf index already exists')
        tf_index = similarities.MatrixSimilarity.load('lsi\\lsi_tf_model.index')

    #Transform queries into the LSI spaces
    q_bow_vec = [dictionary.doc2bow(q.lower().split()) for q in queries]
    q_bow_lsi_vec = bow_model[q_bow_vec]    

    q_tfidf = models.TfidfModel(corpus)     #create a tfidf model based on the corpus
    q_tf_vec = q_tfidf[q_bow_vec]           #transform the bow query vectors to tfidf vectors
    q_tf_lsi_vec = tf_model[q_tf_vec]       #transform the tfidf vectors to vectors in the LSI model's space

    results = search(bow_index,tf_index,q_bow_lsi_vec,q_tf_lsi_vec, docs_by_id)

def search(bow_index, tf_index, bow_queries, tf_queries, docs_by_id):
    #Find documents that match the queries
    bow_results = {}
    tf_results = {}
    
    for i, query in enumerate(bow_queries):
        bow_sims = bow_index[query]
        keys = list(docs_by_id.keys())
        bow_sims = sorted(enumerate(bow_sims), key=lambda item: -item[1])
        bow_results[str(i)] = bow_sims[0][0]

    for i, query in enumerate(tf_queries):
        tf_sims = tf_index[query]
        keys = list(docs_by_id.keys())

        tf_sims = sorted(enumerate(tf_sims), key=lambda item: -item[1])
        tf_results[str(i)] = tf_sims[0][0]
    
    return bow_results, tf_results
