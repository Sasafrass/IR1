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

def run_epoch(model, optimizer, data):
    
    # Parameters
    overall_loss = 0
    epoch_loss = 0
    sigma = 1
    
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
        scorediff = scores - scores.T

        # Calculate all signs
        squeeze_labels = qd_labels.unsqueeze(-1)
        signs = torch.sign(squeeze_labels - qd_labels).float()

        # Loss is just vectorized formula
        # loss = (1/2) * (1-signs) * torch.sigmoid(scorediff) + torch.log(1 + torch.exp(-1 * sigma * scorediff))
        loss = (1/2) * (1-signs) * sigma * scorediff + torch.log(1 + torch.exp(-1 * sigma * scorediff))
        loss = loss.sum()

        # Keep track of rolling average
        overall_loss += loss / (len(ranking) ** 2)

        # Print interesting stuff after X iterations
        # num_iter = 1
        # if((i+1)%num_iter == 0):
        #     print(overall_loss/num_iter)
        #     print(ranking)
        #     print(qd_labels)
        #     print(scores)
        #     overall_loss = 0

        num_queries = 2500
        if (i+1) % num_queries == 0:
            with torch.no_grad():
                avg_ndcg = evaluate_model(model, data.validation)
            print("NDCG: ", avg_ndcg)

        # Update gradients
        loss.backward()
        optimizer.step()

        #break
    
    #
    #print(ranking)

    print(ranking)
    print("epoch_loss: ", overall_loss / data.train.num_queries())

	# TODO: Go over data.validation

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
		run_epoch(model, optimizer, data)