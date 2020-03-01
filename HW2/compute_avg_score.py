import argparse
import os
import scipy.stats
import sys
import json
import numpy as np


def compute_avg(path,queries,measure):
    
    with open(path, 'r') as f: 
        first_results = json.load(f)

    if(queries == '25'):
        query_ids = list(first_results.keys())[76:101]
    else:
        query_ids = list(first_results.keys())

    scores = [first_results[query_id][measure] for query_id in query_ids]

    mean_score = np.mean(scores)
    print(mean_score)
    return mean_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json1', type=str, default="lsi_bow.json", help="Path to json of the results")
    parser.add_argument('--measure', type=str, default='map', help="Measure the results were evaluated with (map/ndcg)")
    parser.add_argument('--queries', type=str, default='all', help="Number of queries to consider (all/25)")
    args = parser.parse_args()
    path = args.json1
    queries = args.queries
    measure = args.measure
    compute_avg(path,queries,measure)