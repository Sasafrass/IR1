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

# Evaluate model import
from pointwise_evaluation import evaluate_model

def run_epoch(model, optimizer, data, eval_every=2500, sigma=1):
    
    # Parameters
    overall_loss = 0
    epoch_loss = 0
    
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
            continue
        loss = 0

        # Vectorize the loss calculation
        # Calculate all score differences
        scorediff = scores_d - scores_d.T

        # Calculate all signs
        squeeze_labels = qd_labels.unsqueeze(-1)
        signs = torch.sign(squeeze_labels - qd_labels).float()

        # Loss is just vectorized formula
        lambdas_ij = sigma * ((1 / 2) * (1 - scorediff) - (1 / (1 + torch.exp(sigma * scorediff))))
        # TODO: Calc IRM
        # lambdas_ij = lambdas_ij * IRM
        lambas_i = lambdas_ij.sum(dim=1)
        loss = scores.squeeze() * lambas_i
        loss = loss.sum()

        # Keep track of rolling average
        overall_loss += loss / (len(ranking) ** 2)

        if (i+1) % eval_every == 0:
            avg_ndcg = evaluate_model(model, data.validation)
            print("NCDG: ", avg_ndcg)

        # Update gradients
        loss.backward()
        optimizer.step()

        #break
    
    #
    #print(ranking)

    print(ranking)
    print("epoch_loss: ", overall_loss / data.train.num_queries())

	# TODO: Go over data.validation

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

            total_ndcg += evaluate_model(model, data.validation)

        return total_ndcg / data.validation.num_queries()

if __name__ == "__main__":

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
	optimizer = optim.Adam(model.parameters())

	# Define what split we are using FOR THE MODEL SO TRAIN OR NOT
	split = "train" #"validation", "test"
	print(f"Split: {split}")
	split = getattr(data, split)
	print(f"\tNumber of queries {split.num_queries()}")

	# Define number of epochs and run for that amount
	num_epochs = 100
	for i in range(num_epochs):
		print("Epoch: ", i)
		run_epoch(model, optimizer, data, sped_up=True)