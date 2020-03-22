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

def run_epoch(model, optimizer, data, eval_every=10000, sigma=1, IRM='ndcg'):
    
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

		# Get labels and get ranking
		np_labels = qd_labels.cpu().numpy()
		np_ranking = ranking.cpu().numpy()

		# Initialize permutations list and add initial permutation
		pred_perms = []
		sorted_pred = np.array([np_labels[idx] for idx in np_ranking])
		pred_perms.append(sorted_pred)

		# Add all vectors of possible permutations to the list
		for p in range(len(sorted_pred)):
			for q in range(p+1, len(sorted_pred)):
				perm_pred = sorted_pred.copy()
				temp = sorted_pred[p] 
				perm_pred[p] = sorted_pred[q]
				perm_pred[q] = temp
				pred_perms.append(perm_pred)
		pred_perms = np.array(pred_perms)

		# Check correct IRM and calculate the scores for each permutation
		if(IRM == 'err'):
			ranking_measure = err(pred_perms)
		elif(IRM == 'ndcg'):
			label_rank = np.sort(np_labels)[::-1]
			ranking_measure = ndcg(pred_perms,label_rank)

		# Calculate all deltas for the IRM
		deltas = np.abs(ranking_measure[0] - ranking_measure[1:])
		delta_irm = np.zeros((len(ranking), len(ranking)))
		delta_irm[np.triu_indices(len(ranking), 1)] = deltas
		delta_irm = torch.from_numpy(delta_irm - delta_irm.T).float().cuda()

		# Get the lambdas and multiply with the delta irm values
		lambdas_ij = lambdas_ij * delta_irm
		lambas_i = lambdas_ij.sum(dim=1)
		loss = scores.squeeze() * lambas_i
		loss = loss.sum()

		# Keep track of rolling average
		overall_loss += loss / (len(ranking) ** 2)

		if (i+1) % eval_every == 0:
			avg_ndcg = evaluate_model(model, data.validation,regression=True)
			print("NCDG: ", avg_ndcg)

		# Update gradients
		loss.backward()
		optimizer.step()

		#break

	#
	#print(ranking)

	print(ranking)
	print("NDCG: ", evaluate_model(model, data.validation,regression=True))
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

			total_ndcg += evaluate_model(model, data.validation,regression=True)

		return total_ndcg / data.validation.num_queries()

def err(ranking):
	p = np.ones(ranking.shape[0])
	err = 0
	max_score = np.max(ranking)
	for r in range(ranking.shape[1]):
		rel_prob = (2 ** ranking[:,r] - 1) / 2 ** max_score
		err = err + p * rel_prob/(r+1)
		p *= (1 - rel_prob)
	return err

def ndcg(ranking,ideal_ranking):
	k = ranking.shape[1]
	denom = 1./np.log2(np.arange(k)+2.)
	# denom = denom.reshape(-1,1)
	nom = 2 ** ranking-1.
	dcg = np.sum(nom * denom, axis=1)
	ideal_dcg_nom = 2 ** ideal_ranking-1.
	ideal_dcg = np.sum(ideal_dcg_nom * denom)

	if(ideal_dcg == 0):
		# All labels are 0, so the order is irrelevant
		return np.ones(dcg.shape[0])

	return dcg / ideal_dcg

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
		start = time.now()
		run_epoch(model, optimizer, data,IRM='ndcg')
		print("I'm done! This run lasted: " + str(time.now() - start))
		#break