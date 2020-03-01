import argparse
import os
import scipy.stats
import sys
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json1', type=str, default="lsi_tf10.json", help="Path json of first results")
    parser.add_argument('--json2', type=str, default="lda.json", help="Path json of second results")
    parser.add_argument('--json3', type=str, default="doc2vec.json", help="Path json of third results")
    parser.add_argument('--json4', type=str, default="word2vec.json", help="Path json of fourth results")
    parser.add_argument('--measure', type=str, default='map', help="Measure the results were evaluated with (\'map\'/\'ndcg\')")
    args = parser.parse_args()

    with open(args.json1, 'r') as f: 
        first_results = json.load(f)
    with open(args.json2, 'r') as f: 
        second_results = json.load(f)
    with open(args.json1, 'r') as f: 
        third_results = json.load(f)
    with open(args.json2, 'r') as f: 
        fourth_results = json.load(f)

    query_ids = list(first_results.keys())

    first_scores = [
        first_results[query_id][args.measure] for query_id in query_ids]
    second_scores = [
        second_results[query_id][args.measure] for query_id in query_ids]
    third_scores = [
        third_results[query_id][args.measure] for query_id in query_ids]
    fourth_scores = [
        fourth_results[query_id][args.measure] for query_id in query_ids]
    scores = [first_scores, second_scores, third_scores, fourth_scores]
    matrix = np.array(scores)
    variances = np.var(matrix,axis=0)
    variances = list(variances)
    #variances = np.flip(np.sort(variances))
    # variances = list(variances)
    ids = [x for _,x in sorted(zip(variances,query_ids),reverse=True)]
    variances = sorted(variances,reverse=True)

    print(ids[0:5])
    print(variances[0:5])

if __name__ == "__main__":

    sys.exit(main())