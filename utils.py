import os
import random

import networkx as nx
import numpy as np

from gensim.models import Word2Vec

def add_weight(subgraph):
    for n1, n2 in subgraph.edges():
        subgraph[n1][n2]['weight'] = 1
    return subgraph
    
def load_data(data_path, read_feature=False):
    G = nx.read_edgelist(os.path.join(data_path, 'edgelist.txt'), nodetype=int)
    add_weight(G)
    if read_feature and os.path.isfile(os.path.join(data_path, 'features.txt')):
        with open(os.path.join(data_path, 'features.txt')) as fp:
            for line in fp:
                vec = line.split()
                G.nodes[int(vec[0])]['feature'] = np.array(
                    [float(x) for x in vec[1:]])
    return G

def print_cut(G: nx.Graph, labels, n_label=5):
    assign_matrix = np.zeros((len(G), n_label))
    assign_matrix[np.arange(len(G)), labels] = 1

    print(np.matmul(assign_matrix.T, nx.adjacency_matrix(G).dot(assign_matrix)))

def init_partition(G, n_partitions,seed=123):
    random.seed(seed)
    n_node_per_part = len(G)//n_partitions
    labels = []
    for i in range(n_partitions-1):
        labels += [i]*n_node_per_part
    labels += [n_partitions-1]*(len(G)-(n_partitions-1)*n_node_per_part)
    random.shuffle(labels)
    assert len(labels) == len(G)
    assignment_dict = {j:labels[i] for i,j in enumerate(G.nodes())}
    return assignment_dict