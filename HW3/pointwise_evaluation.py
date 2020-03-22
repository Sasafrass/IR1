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
            prediction = prediction.squeeze(1).detach().cpu()
        else:
            softmax = F.softmax(scores)
            prediction = torch.argmax(softmax, dim=1)
            prediction = prediction.detach().cpu()

        # Get rankings based on scores
        pred_rank, _ = rnk.rank_and_invert(prediction)
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

    pointwise = './stored_models/pointwise_model.pth'
    listwise_score = './stored_models/listwise_model.pth'
    listwise_err = './stored_models/listwise_modelerr.pth'
    pairwise_speed = './stored_models/pairwise_model.pth'
    pairwise = './stored_models/pairwise_model_notspedup.pth'


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

    pairwise_speed_model = RankNet(input_size = input_size, output_size = 1).float().cuda()
    pairwise_speed_model.load_state_dict(torch.load(pairwise_speed))

    pairwise_model = RankNet(input_size = input_size, output_size = 1).float().cuda()
    pairwise_model.load_state_dict(torch.load(pairwise))

    listwise_score_model = RankNet(input_size = input_size, output_size = 1).float().cuda()
    listwise_score_model.load_state_dict(torch.load(listwise_score))

    listwise_err_model = RankNet(input_size = input_size, output_size = 1).float().cuda()
    listwise_err_model.load_state_dict(torch.load(listwise_err))

    # data.train is a DataFoldSplit
    print("data.train:", data.test)
    ndcg_score = evaluate_model(pointwise_model, data.test)

    print('pointwise')
    print('Test NDCG:', evaluate_model(pointwise_model, data.test), 'validation NDCG:', evaluate_model(pointwise_model, data.validation))

    print('pairwise slow')
    print('Test NDCG:', evaluate_model(pairwise_model, data.test), 'validation NDCG:', evaluate_model(pairwise_model, data.validation,regression=True))

    print('pairwise fast')
    print('Test NDCG:', evaluate_model(pairwise_speed_model, data.test), 'validation NDCG:', evaluate_model(pairwise_speed_model, data.validation,regression=True))

    print('listwise NDCG')
    print('Test NDCG:', evaluate_model(listwise_score_model, data.test), 'validation NDCG:', evaluate_model(listwise_score_model, data.validation,regression=True))

    print('listwise err')
    print('Test NDCG:', evaluate_model(listwise_err_model, data.test), 'validation NDCG:', evaluate_model(listwise_err_model, data.validation,regression=True))