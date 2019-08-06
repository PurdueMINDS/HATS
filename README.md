# HATS
### Authors: Changping Meng

## Overview:
This is the code for HATS: A Hierarchical Sequence-Attention Framework for Inductive Set-of-Sets Embeddings

We focus on Set-of-Sets problems and evaluate our model in arithmatic set operation task, Adamic/Adar Index, subgraph hyperlink prediction and set of pointclouds classification.

The first Set-of-Sets task is to perform arithmetic on sequences of integers: intersection sum, intersection binary, unition sum. 

The second Set-of-Sets task learns to compute Adamic-Adar Index.

The third Set-of-Sets task predicts whether there is a hyperlink between two set of nodes. The feature of each node is the embedding vectors from [GraphSAGE].

The fourth Set-of-Sets task predicts among a set of pointclouds  a) whether there is a pointcloud with different label. b) the number of unique labels.  
Please see the supplementary section for a brief description and summary of the code. 

## Requirements
* PyTorch 0.4.0 or later - which can be downloaded [here](https://www.pytorch.org)
* Python 3.7

## How to Run
For the sequence based tasks, please use the following format:
* python train.py -m "model name" -t "task" -l "number of hidden layers in rho" -lr "learning rate" -b "batch size" -i "running times" 
* Permitted models are {deepsets,lstm,gru,cnn,hats,hier}

For the graph based tasks, we have provided an example below:
* python train.py -m hier -t inter_sum -i 2 -lr 0.001

We recommend training these models on a GPU.

## Data
* For the arithmetic tasks, the data is generated on the fly as described in our paper. You can adjust the number of training, test and validation examples used.
* For the Adamic/Adar and hyperlink prediction tasks, the following datasets were used:
  - [PPI](https://snap.stanford.edu/graphsage/ppi.zip)
  - Cora, [available at the GraphSAGE PyTorch repo](https://github.com/williamleif/graphsage-simple/)
  - Pubmed, [available at the GraphSAGE PyTorch repo](https://github.com/williamleif/graphsage-simple/)
* For pointcloud prediction tasks, we used ModelNet40 dataset (http://modelnet.cs.princeton.edu/)

## Questions
Please feel free to reach out to Changping Meng (meng40  at  purdue.edu) if you have any questions.

## Citation
If you use this code, please consider citing:
```
@inproceedings{meng2019hats,
  title={HATS: A Hierarchical Sequence-Attention Framework for Inductive Set-of-Sets Embeddings},
  author={Meng, Changping and Yang, Jiasen and Ribeiro, Bruno and Neville, Jennifer},
  year={2019},
  organization={KDD}
}
```
