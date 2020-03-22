from evaluate import evaluate_query
from dataset import get_dataset, DataSet
from models import RankNet
import torch
import numpy as np

def calc_arr(model,datasplit):
    rr = 0
    scores = model.forward(torch.tensor(datasplit.feature_matrix).float().cuda()).squeeze().detach().cpu().numpy()
    for i, qid in enumerate(np.arange(datasplit.num_queries())):
        q_labels = datasplit.query_labels(qid)
        if(np.sum(q_labels) > 0):
            results = evaluate_query(datasplit,qid,scores)
            if('relevant rank' in results.keys()):
                rr += np.mean(results['relevant rank'])
    arr = rr/i
    return arr

if __name__ == "__main__":

    # Get data
    dataset = get_dataset()
    data = dataset.get_data_folds()[0]
    print(data.num_features)
    data.read_data()

    validate = True
    if(validate):
        datasplit = data.validation
    else:
        datasplit = data.test

    # Parameters for model
    input_size  = data.num_features
    output_size = 1					# Set this to 1 for regression, 5 for classification

    model_path = 'stored_models/pairwise_model.pthlr0.0002notspedupbad'
    model = RankNet(input_size = input_size, output_size = output_size).float().cuda() # Change the model class accordingly
    model.load_state_dict(torch.load(model_path))
    model.eval()

    arr = calc_arr(model,datasplit)
    print(arr)
            