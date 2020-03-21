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

def evaluate_model(model, validation_set, k = 5, regression = False):
    """
    model:          Neural network model that we are using
    validation_set: Pointer to the validation set
    k:              Cutoff to be used to calculate NDCG
    regression:     Boolean variable to denote whether the model used 
                    regression or classification. This changes the way
                    we feed the scores to the NDCG function
    """
    total_ndcg = 0

    # Main loop over all queries in validation set
    for qid in np.arange(validation_set.num_queries()):

        # Get features, prediction scores and labels
        qd_feats = validation_set.query_feat(qid)
        qd_labels = validation_set.query_labels(qid)
        scores = model.forward(torch.tensor(qd_feats).cuda().float())
        labels = torch.tensor(qd_labels).cuda()

        # Change scores to correct format for regression or classification
        if regression:
            prediction = scores
        else:
            softmax = F.softmax(scores)
            prediction = torch.argmax(softmax, dim=1)

        # Get rankings based on scores
        pred_rank, _ = rnk.rank_and_invert(prediction.squeeze(1).detach().cpu())
        label_rank, _ = rnk.rank_and_invert(labels.detach().cpu())

        # Convert ranking to sorting usable for calculating NDCG
        sorted_pred = np.array([qd_labels[idx] for idx in pred_rank])
        label_rank = np.array(sorted(qd_labels, reverse=True))

        # Get rid of pesky 1-doc queries and naively prevent division by zero
        if len(sorted_pred) > 1 and np.count_nonzero(qd_labels) != 0:

            # k to len(sorted_pred) does NDCG over entire ranking
            k = len(sorted_pred)
            total_ndcg += ndcg_at_k(sorted_pred, label_rank, k)

    return total_ndcg/validation_set.num_queries()

if __name__ == "__main__":

    model_path = './stored_models/pointwise_model.pth'
    # Get data
    dataset = get_dataset()
    data = dataset.get_data_folds()[0]
    print(data.num_features)
    data.read_data()

    # Parameters for model
    input_size  = data.num_features
    output_size = data.num_rel_labels

    # Define model
    model = RankNet(input_size = input_size, output_size = output_size).float().cuda()
    model.load_state_dict(torch.load(model_path))


    # data.train is a DataFoldSplit
    print("data.train:", data.test)
    ndcg_score = evaluate_model(model, data.test)
    print('NDCG: ', ndcg_score)