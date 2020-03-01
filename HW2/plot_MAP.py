import argparse
import os
import scipy.stats
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from compute_avg_score import compute_avg

def plot_map(args):
    means = []
    files = args.jsons.split(',')
    for file in files:
        means.append(compute_avg(file,'all','map'))
    fig, ax = plt.subplots()
    ax.plot(means)

    ax.set(xlabel='parameters', ylabel='mAP',
           title='mAP of ' + args.model_type)
    ax.grid()
    plt.xticks(np.arange(len(files)), args.param_vals.split(','))  # Set locations and labels
    # plt.show()
    fig.savefig(args.model_type + "_params.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsons', type=str, default="lsi_bow10.json,lsi_bow50.json,lsi_bow100.json,lsi_bow500.json,lsi_bow1000.json", help="Comma separated paths to jsons of the results")
    parser.add_argument('--param_vals', type=str, default="10,50,100,500,1000", help="Comma separated parameter x-axis values")
    parser.add_argument('--model_type',type=str,default='LSI', help='Type of model. Used for plot title')
    args = parser.parse_args()
    plot_map(args)