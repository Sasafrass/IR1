import os
import read_ap

def write_results(results):
    print('writing results')
    _, queries = read_ap.read_qrels()

    q_keys = list(queries.keys())
    f= open("doc2vec_trec.txt","w+")
    for i, query in enumerate(q_keys):
        ordered_doc_names = list(results[query].keys())
        for j, doc in enumerate(ordered_doc_names):
            if(j == 1000):
                break
            f.write(query + ' Q0 ' + doc + " " + str(j+1) + " " + str(results[query][doc]) + ' STANDARD\n')

