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
import copy

NUM_EPOCHS_JANOSSY = 2000
NUM_EPOCHS_RNN = 2000
BASE_EMBEDDING_DIMENSION = 10
INFERENCE_PERMUTATIONS = 20
SET_NUMBER = 2

supported_tasks = {'multiple': {'vocab_size': 100, 'sequence_length': 5},
				   'binary': {'vocab_size': 100, 'sequence_length': 5},					
				   'union_sum': {'vocab_size': 10, 'sequence_length': 5},
				   'union_size': {'vocab_size': 10, 'sequence_length': 5},
				   'mul': {'vocab_size': 10, 'sequence_length': 5}}
supported_nature = ['image', 'text']

# Skip GPU 0 because most people like GPU 0.
select_gpu_id = random.randint(1, torch.cuda.device_count()-1)
device = torch.device("cuda:"+str(select_gpu_id) if torch.cuda.is_available() else "cpu")
output_dict = {"accuracy":[],"mae":[],"mse":[],"1_inf_accuracy":[],"1_inf_mae":[],"1_inf_mse":[]}

# Based on task selection create train and test data
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--task', help='Specify the arthimetic task', required=True)
	parser.add_argument('-m', '--model', help='Model to get accuracy scores for', default='janossy_2ary')
	parser.add_argument('-p', '--pool_method', help='Model to get accuracy scores for', default='kary')
	parser.add_argument('-s', '--sequence_length', help='Specify the length of sequence', default=10, type=int)
	parser.add_argument('-i', '--iterations',
						help='Number of iterations to run the task and model for - Confidence Interval tasks',
						default=1, type=int)
	parser.add_argument('-l', '--hidden_layers', help='Number of hidden layers in rho MLP', default=0, type=int)
	parser.add_argument('-k', '--k_ary', help='Size of k-ary', default=0, type=int)
	parser.add_argument('-d', '--data_len', help='Number of data used in experiment', default=0, type=int)
	parser.add_argument('-n', '--neurons', help='Number of neurons in each hidden layer in the rho MLP', default=100,
						type=int)
	parser.add_argument('--type', help='Specify if the task involves text or images', default='text')
	parser.add_argument('-lr', '--learning_rate', help='Specify learning rate', default=0.001, type=float)
	parser.add_argument('-b', '--batch_size', help='Specify batch size', default=128, type=int)
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
	if model not in ['deepsets', 'lstm', 'gru', 'cnn', 'att', 'hats', 'hier', 'mil']:
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


def prepare_data(train_len, subgraph_size, data_name='cora', task='hyperlink'):
	embed_in = np.loadtxt('emb/'+data_name+'.emb',skiprows=0)

	embed = dict()
	for i in range(embed_in.shape[0]):
	    #pdb.set_trace()
	    embed[embed_in[i,0]] = embed_in[i, 1:]

	edges = np.loadtxt('data/'+data_name+'.txt').astype(int)

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


	for k,v in graph.items():
	    #pdb.set_trace()
	    graph[k] = list(v)

	nodeset = list(nodeset)
	node_size = len(nodeset)
	source_set = set()
	train_len = 10000
	src_set = np.zeros((train_len, subgraph_size))
	i = 0
	while i < train_len:
	    #print(i)
	    k = nodeset[random.randint(0,node_size-1)]
	    #pdb.set_trace()
	    cur_len = len(graph[k])
	    if (cur_len < subgraph_size * 2):
	        continue
	    src_set[i,0] = k
	    cur_set = set()
	    j = 0
	    while j < subgraph_size - 1:
	        m = random.randint(0, cur_len-1)
	        if m not in cur_set:
	            cur_set.add(m)
	        else:
	            continue
	        src_set[i,j + 1] = graph[k][m]
	        j += 1
	    i += 1


	dst_set = np.zeros((train_len, subgraph_size))
	i = 0
	while i < train_len:
	    #print(i)
	    k = nodeset[random.randint(0, node_size-1)]
	    cur_len = len(graph[k])
	    if (cur_len < subgraph_size*2):
	        continue
	    dst_set[i,0] = k
	    cur_set = set()
	    j = 0
	    while j < subgraph_size - 1:
	        m = random.randint(0, cur_len-1)
	        if m not in cur_set:
	            cur_set.add(m)
	        else:
	            continue
	        dst_set[i,j + 1] = graph[k][m]
	        j += 1
	    i += 1

	labels = np.zeros(train_len)

	for i in range(train_len):
	    sum = 0;
	    for p in range(subgraph_size):
	        for q in range(subgraph_size):
	            if hash_edge(src_set[i, p], dst_set[i, q]) in edgedict:
	                sum += 1
	    labels[i] = sum

	print(collections.Counter(labels))

	thres_label = 1
	labels[labels >= thres_label] = thres_label
	label_dict = dict(collections.Counter(labels))
	weights = np.ones(int(np.max(labels)) + 1)
	#pdb.set_trace()
	for k,v in label_dict.items():
	    weights[int(k)] = v
	weights = np.sum(weights)/weights
	weights = weights/ np.sum(weights)
	output_label_size = np.max(labels) + 1

	src_features = np.zeros((train_len, subgraph_size, embed_in.shape[1] - 1))
	for i in range(train_len):
	    for j in range(subgraph_size):
	        src_features[i,j] = embed[src_set[i,j]]

	dst_features = np.zeros((train_len, subgraph_size, embed_in.shape[1] - 1))
	for i in range(train_len):
	    for j in range(subgraph_size):
	        dst_features[i,j] = embed[dst_set[i,j]]
	perm = np.arange(src_features.shape[0])
	np.random.shuffle(perm)
	src_set = src_set[perm]
	dst_set = dst_set[perm]
	labels = labels[perm]
	com = np.array([src_set,dst_set])
	com = np.transpose(com, (1,0,2))
	#pdb.set_trace()
	return com, labels
	

def janossy_text_input_construction(X, janossy_k, janossy_k2 = 2 ):
	X_janossy = []
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
		
		temp2 = list(combinations(temp_list, janossy_k2))
		temp2 = [list(x) for x in temp2]
		temp_res = []
		#pdb.set_trace()
		for i in range(len(temp2)):
			temp_res2 = []
			#for j in range(len(temp2[i])):
			for p1 in range(len(temp2[i][0])):
				for p2 in range(len(temp2[i][1])):
					temp_3 = [temp2[i][0][p1][0], temp2[i][1][p2][0]]
					temp_res2.append(temp_3)
			temp_res.append(temp_res2)
		X_janossy.append(temp_res)
	return np.squeeze(np.array(X_janossy))
def embed_X(X, data_name = 'cora'):
	embed_in = np.loadtxt('emb/'+data_name+'.emb',skiprows=0)

	embed = dict()
	for i in range(embed_in.shape[0]):
	    #pdb.set_trace()
	    embed[embed_in[i,0]] = embed_in[i, 1:]
	embedded = np.zeros((X.shape[0],X.shape[1],X.shape[2], embed_in.shape[1] - 1))
	for i in range(X.shape[0]):
	    for j in range(X.shape[1]):
	    	for k in range(X.shape[2]):
	        	embedded[i,j,k] = embed[X[i,j,k]]
	return embedded

def text_dataset_construction(train_or_test, janossy_k, task, janossy_k2, sequence_len, all_data_size=0):
	""" Data Generation """
	janossy_k = 1
	janossy_k2 = 1
	args = parse_args()
	task = str(args.task).lower()
	X = np.load('../data_'+str(task)+str(sequence_len)+'.npy')
	output_X = np.load('../label_'+str(task)+str(sequence_len)+'.npy')
	output_X = np.reshape(output_X,(output_X.shape[0],1))
	total_len = X.shape[0]
	if (all_data_size > 0):
		total_len = all_data_size
	train_len = int(total_len*0.4)
	valid_len = int(total_len*0.2)
	NUM_TRAINING_EXAMPLES = train_len
	NUM_VALIDATION_EXAMPLES = valid_len
	NUM_TEST_EXAMPLES = total_len - train_len - valid_len
	#pdb.set_trace()
	if train_or_test == 1:
		X = X[0:train_len]
		output_X = output_X[0:train_len]
		num_examples = NUM_TRAINING_EXAMPLES
	elif train_or_test == 2:
		X = X[train_len:train_len+valid_len]
		output_X = output_X[train_len:train_len+valid_len]
		num_examples = NUM_VALIDATION_EXAMPLES
	elif train_or_test == 0:
		X = X[train_len+valid_len:]
		output_X = output_X[train_len+valid_len:]
		num_examples = NUM_TEST_EXAMPLES

	set_numbers = X.shape[1]
	train_length = X.shape[0]
	if janossy_k == 1 and janossy_k2 == 1:
		return X, output_X
	else:
		X_janossy = janossy_text_input_construction(X, janossy_k,janossy_k2)
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

def my_pool(X, size, method='mean'):
	
	if size <= 0:
		return X
		
	if method == 'kary':
		return X[:,:,0:size,:]
	
	res = X.view((X.shape[0],X.shape[1],int(X.shape[2]/size), size, X.shape[3]))
	if method == 'mean':
		res = torch.mean(res, 3)
	else:
		res, _ = torch.max(res, 3)
	return res

def pi_permute(X):
	for i in range(X.shape[1]):
		r=torch.randperm(X.shape[-2])
		X [:,i] = X[:,i,r,:]
	r=torch.randperm(X.shape[1])
	X = X[:,r]
	return X

def kary(X,size):
	if size <= 0:
		return X
	X = X[:,:,0:size,:]
	return X

def train_text(vocab_size, input_dim, task, model, num_layers, num_neurons, janossy_k,janossy_k2, learning_rate,batch_size,iteration, sequence_len, data_len, k_ary, pool_method):
	# Construct vocab size base on model
	janossy_model = TextModels(vocab_size, input_dim, model, num_layers, num_neurons, janossy_k,janossy_k2, device)
	janossy_model.to(device)
	data_X, output_X = text_dataset_construction(1, janossy_k, task, 1,sequence_len, data_len)
	data_X = torch.FloatTensor(data_X).to(device)
	#pdb.set_trace()
	output_X = torch.LongTensor(output_X).to(device)
	data_V, output_V = text_dataset_construction(2, janossy_k, task, 1,sequence_len, data_len)
	data_V = torch.FloatTensor(data_V).to(device)
	output_V = torch.LongTensor(output_V).to(device)
	Y, output_Y = text_dataset_construction(0, janossy_k, task, 1,sequence_len, data_len)
	Y = torch.FloatTensor(Y).to(device)
	# model.train
	if model in ['lstm', 'gru','cnn']:
		num_epochs = NUM_EPOCHS_RNN
	else:
		num_epochs = NUM_EPOCHS_JANOSSY
	checkpoint_file_name = str(model) + "_" + str(task) +"_" + str(num_layers) + "_" + "iteration" + str(iteration) + "_" + "learning_rate_" + str(learning_rate) + "_batch_size_" + str(batch_size)+'_sequence_len_' +str(sequence_len) +'_data_len_' +str(data_len) + "_checkpoint.pth.tar"
	# Use Adam Optimizer on all parameters with requires grad as true
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, janossy_model.parameters()), lr=learning_rate)
	# Train over multiple epochs
	start_time = time.time()
	num_batches = int(data_X.shape[0] / batch_size)
	NUM_TRAINING_EXAMPLES = data_X.shape[0]
	NUM_TEST_EXAMPLES = Y.shape[0]
	NUM_VALIDATION_EXAMPLES = data_V.shape[0]
	best_val_accuracy = 0.0
	POOL_SIZE = 200
	best_model = copy.deepcopy(janossy_model)
	for epoch in range(num_epochs):
		# Do seed and random shuffle of the input
		#X, output_X = unison_shuffled(X, output_X)
		#Performing pi-SGD for RNN's
		if model in ['lstm','gru','hier','hats']:
			data_X = pi_permute(data_X)
			data_V = pi_permute(data_V)
		X = my_pool(data_X, k_ary, pool_method)
		V = my_pool(data_V, k_ary, pool_method)
		
		for batch in range(num_batches):
			batch_seq = X[batch_size * batch:batch_size * batch + batch_size]
			optimizer.zero_grad()
			#pdb.set_trace()
			loss = janossy_model.loss(batch_seq, output_X[np.array(range(batch_size * batch, batch_size * batch + batch_size))])
			loss.backward()
			optimizer.step()
		if epoch % 10 == 0:
			with torch.no_grad():
			
				num_v_batches = int(NUM_VALIDATION_EXAMPLES / batch_size)
				val_output = None
				for vbatch in range(num_v_batches):
					valid_batch_seq = V[batch_size * vbatch:batch_size * vbatch + batch_size]
					_val_output = janossy_model.forward(valid_batch_seq)
					if(val_output is None):
						val_output = _val_output
					else:
						val_output = torch.cat((val_output,_val_output),0)
				val_output = torch.argmax(val_output, 1)
				val_correct = torch.sum(torch.eq(val_output, output_V[0:val_output.shape[0]].squeeze())).item()
				val_accuracy = val_correct * 1.0/val_output.shape[0]
				if val_accuracy >= best_val_accuracy:
					best_val_accuracy = val_accuracy
					#Save Weights
					best_model = copy.deepcopy(janossy_model)
		if epoch % 10 == 0:
			print(model, epoch, sequence_len, data_len ,loss.data.item(),val_accuracy)
	end_time = time.time()
	total_training_time = end_time - start_time
	print("Total Training Time: ", total_training_time)
	
	janossy_model = copy.deepcopy(best_model)
	inference_output = np.zeros((NUM_TEST_EXAMPLES, 1))
	Y = my_pool(Y, k_ary, pool_method)
	with torch.no_grad():
	
		num_t_batches = int(NUM_TEST_EXAMPLES / batch_size)
		test_output = None
		for vbatch in range(num_t_batches):
			test_batch_seq = Y[batch_size * vbatch:batch_size * vbatch + batch_size]
			_test_output = np.argmax(janossy_model.forward(test_batch_seq).data.cpu().numpy(), 1)
			if(test_output is None):
				test_output = _test_output
			else:
				test_output = np.concatenate((test_output,_test_output),0)
		inference_output = test_output 
	correct = 0
	for j in range(len(inference_output)):
		if output_Y[j, 0] == inference_output[j]:
			correct += 1
	acc =  1.0 * correct / len(inference_output)
	mae = mean_absolute_error(output_Y[0:len(inference_output)],inference_output)
	mse = mean_squared_error(output_Y[0:len(inference_output)],inference_output)
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
	num_layers = args.hidden_layers
	sequence_len = args.sequence_length
	data_len = args.data_len
	k_ary = args.k_ary
	pool_method = args.pool_method
	learning_rate = args.learning_rate
	janossy_k = valid_argument_check(task, nature, model)
	vocabulary_size = determine_vocab_size(task)
	output_file_name = str(model) + "_" + str(task) + "_" + str(num_layers) + "_" + str(args.learning_rate) + "_" + str(batch_size) + '_' + str(sequence_len) + "_" + str(data_len) + "_" + str(k_ary) +'_'+pool_method+ ".txt"
	for iteration in range(num_iterations) :
		train_text(vocabulary_size, BASE_EMBEDDING_DIMENSION, task, model, num_layers, num_neurons, janossy_k, janossy_k2, learning_rate,batch_size,iteration, sequence_len, data_len, k_ary, pool_method)
		with open(output_file_name,'w') as file :
			file.write(json.dumps(output_dict))

if __name__ == '__main__':
	main()
