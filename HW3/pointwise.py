import numpy as np
import torch
import torch.nn as nn
import random

from dataset import get_dataset
from models import PointwiseLTR

def get_batch(dataset, batch_size):

    def get_random_possible_label(query_idx):
        id_range = dataset.train.query_range(query_idx)
        doc_idx = random.randint(id_range[0], id_range[1]-1)
        labels = dataset.train.query_labels(query_idx)
        label_idx = doc_idx - id_range[0]
        label = labels[label_idx]
        doc_features = dataset.train.doc_feat(query_idx, doc_idx)

        return doc_features, label

    num_queries = dataset.train.num_queries()
    query_idx = np.random.randint(1, num_queries+1, batch_size)

    vectorized_get_random_possible_label = np.vectorize(get_random_possible_label)

    doc_features, labels = vectorized_get_random_possible_label(query_idx)

    return doc_features, labels



def run_model(dataset, batch_size):

    doc_dim = len(dataset.train.doc_feat(1,1))
    model = PointwiseLTR(doc_dim, output_size = 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_batch, label_batch = get_batch(dataset, batch_size)









if __name__ == "__main__":

    batch_size = 128


    dataset = get_dataset()
    data = dataset.get_data_folds()[0]
    print(data.num_features)
    data.read_data()

    run_model(data, batch_size)

    query_idx = 6

    query_range = data.train.query_range(query_idx)
    print('query_range', query_range)

    query_size = data.train.query_size(query_idx)
    print('query_size', query_size, query_size.shape)

    query_sizes = data.train.query_sizes()
    print('query_sizes', query_sizes, len(query_sizes), np.max(query_sizes))

    query_labels = data.train.query_labels(query_idx)
    print('query_labels', query_labels, len(query_labels))

    query_feat = data.train.query_feat(query_idx)
    print('query_feat', query_feat)

    print('num_queries', data.train.num_queries())
    print('num_docs', data.train.num_docs())

    print('hello')