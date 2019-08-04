import torch
import torch.nn as nn
from models import TextModels
import argparse
import sys
import tqdm
import numpy as np
from itertools import combinations,permutations
import time
import random
from torchvision.datasets import MNIST
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import pdb
import math
import collections
import os
import networkx as nx
from networkx.readwrite import json_graph
import random
import math
import copy
NUM_TRAINING_EXAMPLES = 10000
NUM_TEST_EXAMPLES = 10000
NUM_VALIDATION_EXAMPLES = 10000
NUM_EPOCHS_JANOSSY = 4000
NUM_EPOCHS_RNN = 4000
BASE_EMBEDDING_DIMENSION = 10
INFERENCE_PERMUTATIONS = 20
SET_NUMBER = 2

supported_tasks = {'hyperlink': {'vocab_size': 100, 'sequence_length': 5},
				   'union_sum': {'vocab_size': 10, 'sequence_length': 5},
				   'union_size': {'vocab_size': 10, 'sequence_length': 5},
				   'mul': {'vocab_size': 10, 'sequence_length': 5}}
supported_nature = ['image', 'text']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_dict = {"accuracy":[],"mae":[],"mse":[],"1_inf_accuracy":[],"1_inf_mae":[],"1_inf_mse":[]}

# Based on task selection create train and test data
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--task', help='Specify the arthimetic task', required=True)
	parser.add_argument('-m', '--model', help='Model to get accuracy scores for', default='janossy_2ary')
	parser.add_argument('-i', '--iterations',
						help='Number of iterations to run the task and model for - Confidence Interval tasks',
						default=1, type=int)
	parser.add_argument('-l', '--hidden_layers', help='Number of hidden layers in rho MLP', default=0, type=int)
	parser.add_argument('-n', '--neurons', help='Number of neurons in each hidden layer in the rho MLP', default=100,
						type=int)
	parser.add_argument('--type', help='Specify if the task involves text or images', default='text')
	parser.add_argument('-lr', '--learning_rate', help='Specify learning rate', default=0.0001, type=float)
	parser.add_argument('-b', '--batch_size', help='Specify batch size', default=128, type=int)
	parser.add_argument('-s', '--sequence_length', help='Specify the length of sequence', default=10, type=int)
	# Will enable sweeps using master script
	# parser.add_argument('-hl',help='Enable Parameter Sweep over learning rate',action='store_true')
	# parser.add_argument('-hn',help='Enable Parameter Sweep over number of neurons',action='store_true')
	# parser.add_argument('-hh',help='Enable Parameter Sweep over number of hidden layers in rho',action='store_true')
	args = parser.parse_args()
	return args

janossy_k2 = 2
def valid_argument_check(task, nature, model):
	if task not in list(supported_tasks.keys()):
		print("Specified Task %s not supported" % task)
		sys.exit()

	if nature not in supported_nature:
		print("Specified Type %s not supported" % nature)
		sys.exit()
    
	janossy_k = 1
	if model not in ['deepsets', 'lstm', 'gru', 'cnn', 'att', 'hats', 'hier']:
		if 'janossy' == model[0:7]:
			try:
				janossy_k = int(model.split('_')[1].split('ary')[0])
			except:
				print(
					"Specified format incorrect for Janossy model. Model format is janossy_kary where  k in an integer")
				sys.exit()
		else:
			print(
				"Model specified not available. Models should be selected from 'deepsets','lstm','gru','janossy_kary'")
			sys.exit()
	return janossy_k

def hash_edge(src, dst):
    return src*100000 + dst





def prepare_data(train_len, data_name='cora', task='hyperlink'):
	edges = np.loadtxt('./cora.txt').astype(int)
	graph = dict()
	nodeset = set()
	edgedict = set()
	for i in range(edges.shape[0]):
		if edges[i, 0 ] not in graph:
			graph[edges[i, 0 ]] = {edges[i, 1 ]}
		else:
			graph[edges[i, 0 ]].add(edges[i, 1 ])
		if edges[i, 1 ] not in graph:
			graph[edges[i, 1 ]] = {edges[i, 0 ]}
		else:
			graph[edges[i, 1 ]].add(edges[i, 0 ])
		nodeset.add(edges[i, 0 ])
		nodeset.add(edges[i, 1 ])
		edgedict.add(hash_edge(edges[i,0], edges[i,1]))
		edgedict.add(hash_edge(edges[i,1], edges[i,0]))
	nodeset = list(nodeset)
	edges_len = edges.shape[0]
	max_neigh_len = 10
	label = np.zeros((train_len,1))
	nodeset_len = len(nodeset)
	features = np.zeros((train_len, 4, 20))
	i = 0
	while i < train_len:
		randid_A = random.randint(0, nodeset_len-1)
		randid_B = random.randint(0, nodeset_len-1)
		if randid_A == randid_B:
			continue
		first = nodeset[randid_A]
		second = nodeset[randid_B]
		first_neigh = graph[first]
		second_neigh = graph[second]
		intersect = first_neigh.intersection(second_neigh)
		if len(intersect) > 0:
			adam = 0.0
			for val in intersect:
				adam += 1/math.log(len(graph[val]))
			label[i] = adam
			count = 0
			for val in first_neigh:
				features[i, 0, count] = val
				features[i, 1, count] =  (len(graph[val]))
				count += 1
				
				if count >= max_neigh_len:
					break
			count = 0
			for val in second_neigh:
				features[i, 2, count] = val
				features[i, 3, count] = (len(graph[val]))
				count += 1
				if count >= max_neigh_len:
					break
			i += 1

	return features, label
	

def janossy_text_input_construction(X, janossy_k=1, janossy_k2 = 2 ):
	X_janossy = []
	janossy_k = 1
	janossy_k2 = 2
	for index in range(len(X)):
		tmp_janossy = []
		for set_index in range(X.shape[1]):
			temp = list(X[index, set_index])
			temp = [int(x) for x in temp]
			temp.sort()
			#pdb.set_trace()
			temp = list(permutations(temp, janossy_k))
			temp = [list(x) for x in temp]
			tmp_janossy.append(temp)
		temp_list = list(tmp_janossy)
		k_f = math.factorial(janossy_k2)
		
		temp2 = temp_list
		temp_res = []
		#pdb.set_trace()
		i = 0
		while i < len(temp2):
			temp_res2 = []
			#for j in range(len(temp2[i])):
			for p1 in range(len(temp2[i])):
				for p2 in range(len(temp2[i+1])):
					#pdb.set_trace()
					temp_3 = [temp2[i][p1], temp2[i+1][p2]]
					temp_res2.append(temp_3)
			i += 2
			temp_res.append(temp_res2)
		X_janossy.append(temp_res)
	return np.squeeze(np.array(X_janossy))

def text_dataset_construction(train_or_test, janossy_k, task, janossy_k2 = 2):
	""" Data Generation """
	janossy_k = 1
	janossy_k2 = 1
	if train_or_test == 1:
		num_examples = NUM_TRAINING_EXAMPLES
	elif train_or_test == 0:
		num_examples = NUM_TEST_EXAMPLES
	elif train_or_test == 2:
		num_examples = NUM_VALIDATION_EXAMPLES

	X, output_X = prepare_data(num_examples)
	if janossy_k == 1 and janossy_k2 == 1:
		return X, output_X
	else:
		# Create Janossy Input
		X_janossy = janossy_text_input_construction(X, janossy_k,janossy_k2)
		X_janossy = embed_X(X_janossy)
		return X_janossy, output_X


def image_dataset_construction(train_or_test, janossy_k):
	"""Image Data Generation"""
	pass


def determine_vocab_size(task):
	return supported_tasks[task]['vocab_size']


def permute(x):
	return np.random.permutation(x)


def func(x):
	unique, counts = np.unique(ar=x, return_counts=True)
	return unique[np.argmax(counts)]


def unison_shuffled(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]
	

def pi_permute(X):
	#pdb.set_trace()
	i = 0
	while i < X.shape[1]:
		perm = np.random.permutation(X.shape[-1])
		X[:,i] = X[:,i,perm]
		X[:,i+1] = X[:,i+1,perm]
		i += 2
	return X


def train_text(vocab_size, input_dim, task, model, num_layers, num_neurons, janossy_k,janossy_k2, learning_rate,batch_size,iteration):
	# Construct vocab size base on model
	janossy_model = TextModels(vocab_size, input_dim, model, num_layers, num_neurons, janossy_k,janossy_k2, device)
	janossy_model.to(device)
	X, output_X = text_dataset_construction(1, janossy_k, task)
	#pdb.set_trace()
	V, output_V = text_dataset_construction(2, janossy_k, task)
	Y, output_Y = text_dataset_construction(0, janossy_k, task)
	# model.train
	if model in ['lstm', 'gru','cnn']:
		num_epochs = NUM_EPOCHS_RNN
	else:
		num_epochs = NUM_EPOCHS_JANOSSY
	checkpoint_file_name = str(model) + "_" + str(task) +"_" + str(num_layers) + "_" + "iteration" + str(iteration) + "_" + "learning_rate_" + str(learning_rate) + "_batch_size_" + str(batch_size) + "_checkpoint.pth.tar"
	# Use Adam Optimizer on all parameters with requires grad as true
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, janossy_model.parameters()), lr=learning_rate)
	# Train over multiple epochs
	start_time = time.time()
	num_batches = int(NUM_TRAINING_EXAMPLES / batch_size)
	best_val_loss = 100000.0
	best_model = copy.deepcopy(janossy_model)
	for epoch in range(num_epochs):
		pi_permute(X)
		pi_permute(V)
		for batch in range(num_batches):
			batch_seq = torch.FloatTensor(X[batch_size * batch:batch_size * batch + batch_size]).to(device)
			optimizer.zero_grad()
			loss = janossy_model.loss(batch_seq, torch.FloatTensor(output_X[np.array(range(batch_size * batch, batch_size * batch + batch_size))]).to(device))
			loss.backward()
			optimizer.step()
		if epoch % 2 == 0:
			with torch.no_grad():
				num_v_batches = int(NUM_VALIDATION_EXAMPLES / batch_size)
				val_total_loss = 0.0
				for vbatch in range(num_v_batches):
					valid_batch_seq = torch.FloatTensor(V[batch_size * vbatch:batch_size * vbatch + batch_size]).to(device)
					valid_batch_output = torch.FloatTensor(output_V[batch_size * vbatch:batch_size * vbatch + batch_size]).to(device)
					valid_loss = janossy_model.loss(valid_batch_seq, valid_batch_output)
					val_total_loss += valid_loss
				val_total_loss = val_total_loss / num_v_batches
				if val_total_loss <= best_val_loss:
					best_val_loss = val_total_loss
					#Save Weights
					best_model = copy.deepcopy(janossy_model)
		print(epoch, loss.data.item(),val_total_loss)
	end_time = time.time()
	total_training_time = end_time - start_time
	print("Total Training Time: ", total_training_time)

	# model.eval
	janossy_model = copy.deepcopy(best_model)
	inference_output = np.zeros((NUM_TEST_EXAMPLES, 1))
	with torch.no_grad():
	
		num_t_batches = int(NUM_TEST_EXAMPLES / batch_size)
		test_output = None
		for vbatch in range(num_t_batches):
			test_batch_seq = torch.FloatTensor(Y[batch_size * vbatch:batch_size * vbatch + batch_size]).to(device)
			_test_output = (janossy_model.forward(test_batch_seq).data.cpu().numpy())
			if(test_output is None):
				test_output = _test_output
			else:
				test_output = np.concatenate((test_output,_test_output),0)
		inference_output = test_output 

	correct = 0
	acc =  1.0 * correct / len(inference_output)
	mae = mean_absolute_error(output_Y[0:len(inference_output)],test_output)
	mse = mean_squared_error(output_Y[0:len(inference_output)],test_output)
	print("Accuracy :", acc)
	print("Mean Absolute Error: ",mae)
	print("Mean Squared Error: ",mse)
	output_dict["accuracy"].append(acc)
	output_dict["mae"].append(mae)
	output_dict["mse"].append(mse)




def main():
	# if iteration more than 1 present with stddev as well over the runs
	args = parse_args()
	batch_size = args.batch_size
	task = str(args.task).lower()
	nature = str(args.type).lower()
	model = str(args.model).lower()
	num_iterations = int(args.iterations)
	num_neurons = args.neurons
	sequence_len = args.sequence_length
	num_layers = args.hidden_layers
	learning_rate = args.learning_rate
	janossy_k = valid_argument_check(task, nature, model)
	vocabulary_size = determine_vocab_size(task)
	output_file_name = str(model) + "_" + str(task) + "_" + str(num_layers) + "_" + str(args.learning_rate) + "_" + str(batch_size) + "10000.txt"
	for iteration in range(num_iterations) :
		train_text(vocabulary_size, BASE_EMBEDDING_DIMENSION, task, model, num_layers, num_neurons, janossy_k, janossy_k2, learning_rate,batch_size,iteration)
		with open(output_file_name,'w') as file :
			file.write(json.dumps(output_dict))

if __name__ == '__main__':
	main()
