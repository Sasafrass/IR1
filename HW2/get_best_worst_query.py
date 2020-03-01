import argparse
import os
import scipy.stats
import sys
import json
import numpy as np

def get_best_worst_query(path,measure):
    
    with open(path, 'r') as f: 
        first_results = json.load(f)

    query_ids = list(first_results.keys())
    scores = [(first_results[query_id][measure], query_id) for query_id in query_ids]
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    print("Best: " + str(scores[0][1]) + " Score: " + str(scores[0][0]))
    print("Worst: " + str(scores[-1][1]) + " Score: " + str(scores[-1][0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default="lsi_bow10.json", help="Path to json of the results")
    parser.add_argument('--measure', type=str, default='map', help="Measure the results were evaluated with (map/ndcg)")
    args = parser.parse_args()
    path = args.json
    measure = args.measure
    get_best_worst_query(path,measure)