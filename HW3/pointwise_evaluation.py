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

def evaluate_model(model, validation_set):
    total_ndcg = 0
    for qid in np.arange(validation_set.num_queries()):
        qd_feats = validation_set.query_feat(qid)
        qd_labels = validation_set.query_labels(qid)
        scores = model.forward(torch.tensor(qd_feats).float().cuda())
        labels = torch.tensor(qd_labels)
        softmax = F.softmax(scores,dim=0)
        prediction = torch.argmax(softmax, dim=1).cpu()

        pred_rank, _ = rnk.rank_and_invert(prediction)
        label_rank, _ = rnk.rank_and_invert(labels)

        sorted_pred = np.array([qd_labels[idx] for idx in pred_rank])
        label_rank = np.array(sorted(qd_labels, reverse=True))
        if len(sorted_pred) > 1 and np.count_nonzero(qd_labels) != 0:
            total_ndcg += ndcg_at_k(sorted_pred, label_rank, len(sorted_pred))

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