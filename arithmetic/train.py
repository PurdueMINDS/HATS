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
import copy

NUM_TRAINING_EXAMPLES = 10000
NUM_TEST_EXAMPLES = 10000
NUM_VALIDATION_EXAMPLES = 10000
NUM_EPOCHS_JANOSSY = 1000
NUM_EPOCHS_RNN = 1000
BASE_EMBEDDING_DIMENSION = 10
INFERENCE_PERMUTATIONS = 20
SET_NUMBER = 4
xrange = range
supported_tasks = {'range': {'vocab_size': 10, 'sequence_length': 5},
				   'sum': {'vocab_size': 10, 'sequence_length': 5},
				   'unique_sum': {'vocab_size': 10, 'sequence_length': 5},
				   'unique_count': {'vocab_size': 10, 'sequence_length': 10},
				   'variance': {'vocab_size': 100, 'sequence_length': 10},
				   'stddev': {'vocab_size': 100, 'sequence_length': 10},
				   'range_sum': {'vocab_size': 15, 'sequence_length': 10},
				   'inter': {'vocab_size': 5, 'sequence_length': 10},
				   'inter_binary': {'vocab_size': 10, 'sequence_length': 10},
				   'inter_sum': {'vocab_size': 10, 'sequence_length': 10},
				   'union_sum': {'vocab_size': 10, 'sequence_length': 10},
				   'union_size': {'vocab_size': 10, 'sequence_length': 10},
				   'mul': {'vocab_size': 10, 'sequence_length': 5}}
supported_nature = ['image', 'text']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_dict = {"accuracy":[],"mae":[],"mse":[],"1_inf_accuracy":[],"1_inf_mae":[],"1_inf_mse":[]}

# Based on task selection create train and test data
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--task', help='Specify the arthimetic task', required=True)
	parser.add_argument('-m', '--model', help='Model to get accuracy scores for', default='janossy_2ary')
	parser.add_argument('-s', '--sequence_length', help='Specify the length of sequence', default=10, type=int)
	parser.add_argument('-i', '--iterations',
						help='Number of iterations to run the task and model for - Confidence Interval tasks',
						default=1, type=int)
	parser.add_argument('-l', '--hidden_layers', help='Number of hidden layers in rho MLP', default=0, type=int)
	parser.add_argument('-n', '--neurons', help='Number of neurons in each hidden layer in the rho MLP', default=100,
						type=int)
	parser.add_argument('--type', help='Specify if the task involves text or images', default='text')
	parser.add_argument('-lr', '--learning_rate', help='Specify learning rate', default=0.01, type=float)
	parser.add_argument('-b', '--batch_size', help='Specify batch size', default=128, type=int)
	# Will enable sweeps using master script
	# parser.add_argument('-hl',help='Enable Parameter Sweep over learning rate',action='store_true')
	# parser.add_argument('-hn',help='Enable Parameter Sweep over number of neurons',action='store_true')
	# parser.add_argument('-hh',help='Enable Parameter Sweep over number of hidden layers in rho',action='store_true')
	args = parser.parse_args()
	return args

janossy_k2 = 1
def valid_argument_check(task, nature, model):
	if task not in list(supported_tasks.keys()):
		print("Specified Task %s not supported" % task)
		sys.exit()

	if nature not in supported_nature:
		print("Specified Type %s not supported" % nature)
		sys.exit()
    
	janossy_k = 1
	print(model)
	if model not in ['lstm', 'gru', 'cnn', 'att', 'hats', 'hier', 'deepset']:
		if 'janossy' == model[0:7]:
			try:
				janossy_k = int(model.split('_')[1].split('ary')[0])
			except:
				print(
					"Specified format incorrect for Janossy model. Model format is janossy_kary where  k in an integer")
				sys.exit()
		else:
			print(
				"Model specified not available. Models should be selected from 'deepset','lstm','gru','janossy_kary'")
			sys.exit()
	return janossy_k


def construct_task_specific_output(task, input_sequence):
	if task == 'range':
		return np.max(input_sequence) - np.min(input_sequence)
	if task == 'sum':
		return np.sum(input_sequence)
	if task == 'unique_sum':
		return np.sum(np.unique(input_sequence))
	if task == 'unique_count':
		return np.size(np.unique(input_sequence))
	if task == 'variance':
		return np.var(input_sequence)
	if task == 'stddev':
		return np.std(input_sequence)
	if task == 'mul':
		return np.prod(input_sequence)

def construct_task_specific_output_2D(task, input_sequence):
	if task == 'range_sum':
		input_sequence_sum = np.sum(input_sequence, 1)
		return np.max(input_sequence_sum) - np.min(input_sequence_sum)
	if task == 'inter':
		result = input_sequence[0]
		for i in range(1, input_sequence.shape[0]):
			result = np.intersect1d(input_sequence[i], result)
		return len(result)*1.0
	if task == 'inter_binary':
		result = input_sequence[0]
		for i in range(1, input_sequence.shape[0]):
			result = np.intersect1d(input_sequence[i], result)
		return len(result) > 0
	if task == 'inter_sum':
		result = input_sequence[0]
		for i in range(1, input_sequence.shape[0]):
			result = np.intersect1d(input_sequence[i], result)
		#pdb.set_trace()
		return np.sum(result)
	if task == 'union_size':
		result = input_sequence[0]
		for i in range(1, input_sequence.shape[0]):
			result = np.union1d(input_sequence[i], result)
		return len(result)
	if task == 'union_sum':
		result = input_sequence[0]
		for i in range(1, input_sequence.shape[0]):
			result = np.union1d(input_sequence[i], result)
		return np.sum(result)
	if task == 'range':
		#pdb.set_trace()
		input_sequence = input_sequence.reshape(input_sequence.shape[0] * input_sequence.shape[1])
		return np.max(input_sequence) - np.min(input_sequence)
	if task == 'sum':
		input_sequence = input_sequence.reshape(input_sequence.shape[0] * input_sequence.shape[1])
		return np.sum(input_sequence)
	if task == 'unique_sum':
		input_sequence = input_sequence.reshape(input_sequence.shape[0] * input_sequence.shape[1])
		return np.sum(np.unique(input_sequence))
	if task == 'unique_count':
		input_sequence = input_sequence.reshape(input_sequence.shape[0] * input_sequence.shape[1])
		return np.size(np.unique(input_sequence))
	if task == 'variance':
		input_sequence = input_sequence.reshape(input_sequence.shape[0] * input_sequence.shape[1])
		return np.var(input_sequence)
	if task == 'stddev':
		input_sequence = input_sequence.reshape(input_sequence.shape[0] * input_sequence.shape[1])
		return np.std(input_sequence)
	if task == 'mul':
		input_sequence = input_sequence.reshape(input_sequence.shape[0] * input_sequence.shape[1])
		return np.prod(input_sequence)
	


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

def janossy_text_input_construction_1D(X, janossy_k):
	X_janossy = []
	for index in range(len(X)):
		temp = list(X[index])
		temp = [int(x) for x in temp]
		temp.sort()
		temp = list(combinations(temp, janossy_k))
		temp = [list(x) for x in temp]
		X_janossy.append(temp)
	return np.array(X_janossy)

def text_dataset_construction(train_or_test, janossy_k, task, janossy_k2, sequence_length):
	""" Data Generation """
	if train_or_test == 1:
		num_examples = NUM_TRAINING_EXAMPLES
	elif train_or_test == 0:
		num_examples = NUM_TEST_EXAMPLES
	elif train_or_test == 2:
		num_examples = NUM_VALIDATION_EXAMPLES
	#janossy_k = 1
	#janossy_k2 = 1
	set_numbers = SET_NUMBER
	train_length = sequence_length
	vocab_size = sequence_length#int(supported_tasks[task]['vocab_size'])
	X = np.zeros((num_examples, set_numbers, train_length))
	output_X = np.zeros((num_examples, 1))
	for i in tqdm.tqdm(range(num_examples), desc='Generating Training / Validation / Test Examples: '):
		for k in range(set_numbers):
			X[i, k] = np.random.choice(range(vocab_size), train_length, replace=True)
		output_X[i, 0] = construct_task_specific_output_2D(task, X[i])
	if janossy_k == 1 and janossy_k2 == 1:
		return X, output_X
	else:
		# Create Janossy Input
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


def train_text(vocab_size, input_dim, task, model, num_layers, num_neurons, janossy_k,janossy_k2, learning_rate,batch_size,iteration, sequence_len):
	# Construct vocab size base on model
	janossy_model = TextModels(vocab_size, input_dim, model, num_layers, num_neurons, janossy_k,janossy_k2, device)
	janossy_model.to(device)
	X, output_X = text_dataset_construction(1, 1, task, 1, sequence_len)
	#pdb.set_trace()
	V, output_V = text_dataset_construction(2, 1, task, 1, sequence_len)
	Y, output_Y = text_dataset_construction(0, 1, task, 1, sequence_len)
	# model.train
	if model in ['lstm', 'gru','cnn']:
		num_epochs = NUM_EPOCHS_RNN
	else:
		num_epochs = NUM_EPOCHS_JANOSSY
	checkpoint_file_name = str(model) + "_" + str(task) +"_" + str(num_layers) + "_" + "iteration" + str(iteration) + "_" + "learning_rate_" + str(learning_rate) + "_batch_size_" + str(batch_size)+ "_seq_length_"+ str(sequence_len)+ "_vocab_"+ str(sequence_len)+ "_checkpoint.pth.tar"
	# Use Adam Optimizer on all parameters with requires grad as true
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, janossy_model.parameters()), lr=learning_rate)
	# Train over multiple epochs
	start_time = time.time()
	num_batches = int(NUM_TRAINING_EXAMPLES / batch_size)
	best_val_accuracy = 0.0
	best_model = copy.deepcopy(janossy_model)
	for epoch in range(num_epochs):
		for batch in range(num_batches):
		
			batch_seq = X[batch_size * batch:batch_size * batch + batch_size]
			# Performing pi-SGD for RNN
			if model in ['lstm','gru','hats','hier']:
				batch_seq = np.transpose(batch_seq,(1,0,2))
				np.random.shuffle(batch_seq)
				batch_seq = np.transpose(batch_seq,(1,0,2))
				batch_seq = np.apply_along_axis(permute, 2, batch_seq)
			batch_seq = torch.LongTensor(batch_seq).to(device)
			optimizer.zero_grad()
			loss = janossy_model.loss(batch_seq, torch.FloatTensor(output_X[np.array(range(batch_size * batch, batch_size * batch + batch_size))]).to(device))
			loss.backward()
			optimizer.step()
		if epoch % 2 == 0:
			with torch.no_grad():
			
				num_v_batches = int(NUM_VALIDATION_EXAMPLES / batch_size)
				val_output = None
				for vbatch in range(num_v_batches):
					valid_batch_seq = torch.LongTensor(V[batch_size * vbatch:batch_size * vbatch + batch_size]).to(device)
					_val_output = np.round(janossy_model.forward(valid_batch_seq).data.cpu().numpy())
					if(val_output is None):
						val_output = _val_output
					else:
						val_output = np.concatenate((val_output,_val_output),0)
				val_correct = 0
				for j in range(len(val_output)):
					if output_V[j,0] == val_output[j,0]:
						val_correct+=1
				val_accuracy = (1.0*val_correct)/(len(val_output))
				if val_accuracy >= best_val_accuracy:
					best_val_accuracy = val_accuracy
					#Save Weights
					best_model = copy.deepcopy(janossy_model)
					torch.save(janossy_model.state_dict(),checkpoint_file_name)	
		print(epoch, loss.data.item(),val_accuracy)
	end_time = time.time()
	total_training_time = end_time - start_time
	print("Total Training Time: ", total_training_time)

	# model.eval
	janossy_model = copy.deepcopy(best_model)
	inference_output = np.zeros((int(NUM_TEST_EXAMPLES / batch_size) * batch_size , 1))
	with torch.no_grad():
	
		num_t_batches = int(NUM_TEST_EXAMPLES / batch_size)
		for inference_step in range(INFERENCE_PERMUTATIONS - 1):
			test_output = None
			for vbatch in range(num_t_batches):
				test_batch_seq = torch.LongTensor(Y[batch_size * vbatch:batch_size * vbatch + batch_size]).to(device)
				_test_output = np.round(janossy_model.forward(test_batch_seq).data.cpu().numpy())
				if(test_output is None):
					test_output = _test_output
				else:
					test_output = np.concatenate((test_output,_test_output),0)
			inference_output = np.column_stack((inference_output, np.round(test_output)))
		inference_output = np.apply_along_axis(func, 1, inference_output)
		inference_output = np.reshape(inference_output,(-1,1))
	correct = 0
	for j in range(len(inference_output)):
		if output_Y[j, 0] == inference_output[j, 0]:
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
	sequence_len = args.sequence_length
	num_neurons = args.neurons
	num_layers = args.hidden_layers
	learning_rate = args.learning_rate
	janossy_k = valid_argument_check(task, nature, model)
	vocabulary_size = determine_vocab_size(task)
	output_file_name = str(model) + "_" + str(task) + "_" + str(num_layers) + "_" + str(args.learning_rate) + "_" + str(batch_size) + "_"+str(args.sequence_length)+"_"+str(args.sequence_length) + ".txt"
	for iteration in range(num_iterations) :
		train_text(vocabulary_size, BASE_EMBEDDING_DIMENSION, task, model, num_layers, num_neurons, janossy_k, janossy_k2, learning_rate,batch_size,iteration, sequence_len)
		with open(output_file_name,'w') as file :
			file.write(json.dumps(output_dict))

if __name__ == '__main__':
	main()
