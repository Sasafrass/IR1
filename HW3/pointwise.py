# Numerical imports
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from pointwise_evaluation import evaluate_model
import torch.nn.functional as F
import torch.optim as optim
from evaluate import ndcg_at_k

# import the models
from models import RankNet
from dataset import get_dataset, DataSet
import ranking as rnk

def run_epoch(model, optimizer, data):
	total_loss = 0
	total_acc = 0
	for qid in np.arange(data.train.num_queries()):
		optimizer.zero_grad()
		qd_feats = data.train.query_feat(qid)
		qd_labels = data.train.query_labels(qid)
		scores = model.forward(torch.tensor(qd_feats).float().cuda())
		labels = torch.tensor(qd_labels).cuda()
		softmax = F.softmax(scores)
		prediction = torch.argmax(softmax, dim=1)
		corr = [1 if prediction[x] == labels[x] else 0 for x in range(len(prediction))]
		loss = F.cross_entropy(scores,labels)
		loss.backward()
		total_loss += loss.item()
		total_acc += sum(corr)/len(prediction)
		optimizer.step()

	print('loss: ', total_loss/data.train.num_queries(), 'Accuracy: ', total_acc/data.train.num_queries())

	# TODO: Go over data.validation

if __name__ == "__main__":

	model_path = 'stored_models/pointwise_model.pth'
	num_epochs = 10

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
	output_size = data.num_rel_labels

	# Define model
	model = RankNet(input_size = input_size, output_size = output_size).float().cuda()
	optimizer = optim.Adam(model.parameters(), lr=0.02)

	# Define what split we are using FOR THE MODEL SO TRAIN OR NOT
	split = "train" #"validation", "test"
	print(f"Split: {split}")
	split = getattr(data, split)
	print(f"\tNumber of queries {split.num_queries()}")

	# Define number of epochs and run for that amount
	for i in range(num_epochs):
		model.train
		run_epoch(model, optimizer, data)
		model.eval
		ndcg_score = evaluate_model(model, data.validation)
		print('NDCG: ', ndcg_score)


	torch.save(model.state_dict(), model_path)