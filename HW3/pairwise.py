# Numerical imports
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import the models
from models import RankNet
from dataset import get_dataset, DataSet
import ranking as rnk

# Time functions
import time
from datetime import datetime as time

# Evaluate model import
from pointwise_evaluation import evaluate_model

def run_epoch(model, optimizer, data, eval_every=3000, sped_up=False, sigma=1):
    
    # Parameters
    overall_loss = 0
    epoch_loss = 0
    
    temp_loss = 0
    # Main Pairwise RankNet function
    for i, qid in enumerate(np.arange(data.train.num_queries())):

        # Zero the gradient buffer and get doc,query combinations, labels and scores
        optimizer.zero_grad()
        qd_feats = data.train.query_feat(qid)
        qd_labels = data.train.query_labels(qid)
        scores = model.forward(torch.tensor(qd_feats).float().cuda())
        
        # Make em torchy?
        qd_feats = torch.tensor(qd_feats).cuda()
        qd_labels = torch.tensor(qd_labels).cuda()

        if(sped_up):
            loss, ranking = calc_loss_sped_up(scores,qd_labels,sigma)
        else:
            loss, ranking = calc_loss(scores,qd_labels,sigma)

        if(isinstance(ranking, int)):
            continue

        # Keep track of rolling average
        overall_loss += loss / (len(ranking) ** 2)

        temp_loss += loss

        if (i+1) % eval_every == 0:
            model.eval
            avg_ndcg = evaluate_model(model, data.validation, regression=True)
            print("Iteration: ", i+1,"NCDG: ", avg_ndcg, 'loss:',temp_loss.item()/eval_every)
            model.train
            temp_loss = 0

        # Update gradients
        loss.backward()
        optimizer.step()

        #break
    
    #print("epoch_loss: ", overall_loss / data.train.num_queries())

	# TODO: Go over data.validation

def calc_loss(scores, qd_labels, sigma):
    # TODO: Fix this spaghetti shit
    ranking_boi = np.float64(scores.squeeze().cpu().detach().numpy())
    
    # Get the ranking 
    if not isinstance(ranking_boi, np.ndarray):
        ranking_boi = np.array([ranking_boi])
    ranking, inv_ranking = rnk.rank_and_invert(ranking_boi)

    ranking = torch.tensor(ranking.copy()).cuda()

    # Get rid of pesky 1-document queries and initialize loss
    if len(scores) < 2:
        return 0, 0
    loss = 0

    # Vectorize the loss calculation
    # Calculate all score differences
    scorediff = scores - scores.T

    # Calculate all signs
    squeeze_labels = qd_labels.unsqueeze(-1)
    signs = torch.sign(squeeze_labels - qd_labels).float()

    # Loss is just vectorized formula
    # loss = (1/2) * (1-signs) * torch.sigmoid(scorediff) + torch.log(1 + torch.exp(-1 * sigma * scorediff))
    loss = (1/2) * (1-signs) * sigma * scorediff + torch.log(1 + torch.exp(-1 * sigma * scorediff))
    loss = loss.sum()

    return loss, ranking

def calc_loss_sped_up(scores, qd_labels, sigma):
    # TODO: Fix this spaghetti shit
    scores_d = scores.detach()
    ranking_boi = np.float64(scores.squeeze().cpu().detach().numpy())
    
    # Get the ranking 
    if not isinstance(ranking_boi, np.ndarray):
        ranking_boi = np.array([ranking_boi])
    ranking, inv_ranking = rnk.rank_and_invert(ranking_boi)

    ranking = torch.tensor(ranking.copy()).cuda()

    # Get rid of pesky 1-document queries and initialize loss
    if len(scores) < 2:
        return 0, 0
    loss = 0

    # Vectorize the loss calculation
    # Calculate all score differences
    scorediff = scores_d - scores_d.T

    # Calculate all signs
    squeeze_labels = qd_labels.unsqueeze(-1)
    signs = torch.sign(squeeze_labels - qd_labels).float()

    # Loss is just vectorized formula
    lambdas_ij = sigma * ((1 / 2) * (1 - scorediff) - (1 / (1 + torch.exp(sigma * scorediff))))
    lambas_i = lambdas_ij.sum(dim=1)
    loss = scores.squeeze() * lambas_i
    loss = loss.sum()

    return loss, ranking

def validate_ndcg():
    with torch.no_grad():

        total_ndcg = 0

        for i, qid in enumerate(np.arange(data.validation.num_queries())):
            qd_feats = data.validation.query_feat(qid)
            qd_labels = data.validation.query_labels(qid)
            scores = model.forward(torch.tensor(qd_feats).float().cuda())

            # Make em torchy?
            qd_feats = torch.tensor(qd_feats).cuda()
            qd_labels = torch.tensor(qd_labels).cuda()

            # TODO: Fix this spaghetti shit
            ranking_boi = np.float64(scores.squeeze().cpu().detach().numpy())
            
            # Get the ranking 
            if not isinstance(ranking_boi, np.ndarray):
                ranking_boi = np.array([ranking_boi])
            ranking, inv_ranking = rnk.rank_and_invert(ranking_boi)

            ranking = torch.tensor(ranking.copy()).cuda()
        
            # Get rid of pesky 1-document queries and initialize loss
            if len(scores) < 2:
                continue

            total_ndcg += evaluate_model(model, data.validation,regression=True)

        return total_ndcg / data.validation.num_queries()

if __name__ == "__main__":

    model_path = 'stored_models/pairwise_model.pth'
    eval_every= 1
    num_epochs = 20
    early_stopping = 0.00001
    learning_rate = 0.0002

	# Get data
    dataset = get_dataset()
    data = dataset.get_data_folds()[0]
    print(data.num_features)
    data.read_data()

    # data.train is a DataFoldSplit
    print("data.train:", data.train)
    print("data.train:", data.train.num_queries())

    # Parameters for model
    input_size  = data.num_features
    output_size = 1					# Output size is 1 because regression

    # Define model
    model = RankNet(input_size = input_size, output_size = output_size).float().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define what split we are using FOR THE MODEL SO TRAIN OR NOT
    split = "train" #"validation", "test"
    print(f"Split: {split}")
    split = getattr(data, split)
    print(f"\tNumber of queries {split.num_queries()}")

    prev_ndcg = 0

    # Define number of epochs and run for that amount
    for i in range(num_epochs):
        run_epoch(model, optimizer, data, sped_up=True)
        if (i+1) % eval_every == 0:
            avg_ndcg = evaluate_model(model, data.validation, regression=True)
            print("Epoch: ", i+1,"NCDG: ", avg_ndcg)

            if (abs(prev_ndcg - avg_ndcg) < early_stopping):
                print('early stopping')
                print(early_stopping)
                break
            prev_ndcg = avg_ndcg
    torch.save(model.state_dict(), model_path+'lr'+str(learning_rate)+'notspedupbad')
