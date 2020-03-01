import argparse
import os
import scipy.stats
import sys
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json1', type=str, default="lsi_bow500.json", help="Path json of first results")
    parser.add_argument('--json2', type=str, default="lsi_tf500.json", help="Path json of second results")
    parser.add_argument('--measure', type=str, default='map', help="Measure the results were evaluated with (\'map\'/\'ndcg\')")
    args = parser.parse_args()

    with open(args.json1, 'r') as f: 
        first_results = json.load(f)
    with open(args.json2, 'r') as f: 
        second_results = json.load(f)

    query_ids = list(
        set(first_results.keys()) & set(second_results.keys()))

    first_scores = [
        first_results[query_id][args.measure] for query_id in query_ids]
    second_scores = [
        second_results[query_id][args.measure] for query_id in query_ids]

    print("Measure: " + str(args.measure) + " " + str(scipy.stats.ttest_rel(first_scores, second_scores)))

if __name__ == "__main__":

    sys.exit(main())