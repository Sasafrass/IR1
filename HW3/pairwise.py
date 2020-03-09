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

def run_epoch(model, optimizer, data):
	overall_loss = 0
	for i, qid in enumerate(np.arange(data.train.num_queries())):
		optimizer.zero_grad()
		qd_feats = data.train.query_feat(qid)
		qd_labels = data.train.query_labels(qid)
		scores = model.forward(torch.tensor(qd_feats).float().cuda())
		
		# TODO: Fix this spaghetti shit
		ranking_boi = np.float64(scores.squeeze().cpu().detach().numpy())
		if not isinstance(ranking_boi, np.ndarray):
			ranking_boi = np.array([ranking_boi])
			
		ranking, inv_ranking = rnk.rank_and_invert(ranking_boi)
		
		loss = 0
		
		for s_i in ranking:
			for s_j in ranking:
				#Calc loss for S_ij
				s_ij = np.sign(qd_labels[s_i]-qd_labels[s_j])
				sigmoid_ij = torch.sigmoid(scores[s_i].float() - scores[s_j].float())
				loss += (1/2) * (1-s_ij) * sigmoid_ij + torch.log(1 + torch.exp(-1*sigmoid_ij))

		overall_loss += loss / len(ranking) ** 2

		if((i+1)%1000 == 0):
			print(overall_loss/1000)
			overall_loss = 0

		loss.backward()
		optimizer.step()

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
	num_epochs = 10
	for i in range(num_epochs):
		print("Epoch: ", i)
		run_epoch(model, optimizer, data)