import torch
import numpy as np

# import the models
import torch
from evaluate import ndcg_at_k
import dataset
from models import RankNet
from dataset import get_dataset, DataSet
import ranking as rnk
import torch.nn.functional as F

def compute_distribution(model, dataset):

    validation_set = dataset.validation
    test_set = dataset.test

    test_label = [0, 0, 0, 0, 0]
    validation_label = [0, 0, 0, 0, 0]
    true_test_label = [0, 0, 0, 0, 0]
    true_validation_label = [0, 0, 0, 0, 0]
    # Main loop over all queries in validation set
    for qid in np.arange(validation_set.num_queries()):

        # Get features, prediction scores and labels
        qd_feats = validation_set.query_feat(qid)
        qd_labels = validation_set.query_labels(qid)
        scores = model.forward(torch.tensor(qd_feats).cuda().float())
        labels = torch.tensor(qd_labels).cuda()

        # Change scores to correct format for regression or classification
 
        softmax = F.softmax(scores)
        prediction = torch.argmax(softmax, dim=1)
        prediction = prediction.detach().cpu()

        for i in range(len(prediction)):
            validation_label[prediction[i].item()] += 1
            true_validation_label[labels[i].item()] += 1

    for qid in np.arange(test_set.num_queries()):

        # Get features, prediction scores and labels
        qd_feats = test_set.query_feat(qid)
        qd_labels = test_set.query_labels(qid)
        scores = model.forward(torch.tensor(qd_feats).cuda().float())
        labels = torch.tensor(qd_labels).cuda()

        # Change scores to correct format for regression or classification
 
        softmax = F.softmax(scores)
        prediction = torch.argmax(softmax, dim=1)
        prediction = prediction.detach().cpu()

        for i in range(len(prediction)):
            test_label[prediction[i].item()] += 1
            true_test_label[labels[i].item()] += 1



    return test_label, validation_label, true_test_label, true_validation_label

if __name__ == "__main__":

    pointwise = './stored_models/pointwise_model.pth'

    # Get data
    dataset = get_dataset()
    data = dataset.get_data_folds()[0]
    print(data.num_features)
    data.read_data()

    # Parameters for model
    input_size  = data.num_features
    output_size = data.num_rel_labels

    # Define model
    pointwise_model = RankNet(input_size = input_size, output_size = output_size).float().cuda()
    pointwise_model.load_state_dict(torch.load(pointwise))

    # data.train is a DataFoldSplit
    print("data.train:", data.test)
    dist = compute_distribution(pointwise_model, data)

    print(dist)