import random 
from collections import Counter, defaultdict
import sys, os
import argparse

import networkx as nx
import numpy as np 

from pyspark import SparkContext

from deepwalk import DeepWalk,embed_by_deepwalk
from utils import load_data, print_cut, init_partition
from evaluate import eval_classify

def parse_args():
   parser = argparse.ArgumentParser(description="Parallel embedding")
   parser.add_argument('--data_path', default="./cora")
   parser.add_argument('--num_subgraphs', default=3,
                     type=int, help="Number of subgraphs")
   parser.add_argument('--dim_size', default=128,
                     type=int, help="Dimension size")
   parser.add_argument('--num_anchors', default=500, type=int,
                     help="Number of anchor nodes, required if num_anchors_percent is not specified.")
   parser.add_argument('--num_anchors_percent', default=0.01, type=float,
                     help="Number of anchor by percent, if specified, num_anchors_percent will be used rather than num_anchors.")
   parser.add_argument('--seed', default=42, type=int)

   # Evaluation by classification
   parser.add_argument('--train_percent', default=0.5,
                     type=float, help="Train percent")
    
   # Embed by deepwalk
   parser.add_argument('--walk_length', default=10,
                     type=int, help="Walk length")
   parser.add_argument('--num_walks', default=20,
                     type=int, help="Number of walks")
   parser.add_argument('--num_workers', default=4, type=int,
                     help="Number of workers for gensim")
   parser.add_argument('--window_size', default=5,
                     type=int, help="Window size")
   return parser.parse_args()

args = parse_args()

sc = SparkContext("local", "Accumulator app") 

G = load_data(args.data_path)
G_boardcast = sc.broadcast(G)
assignment_dict = init_partition(G, args.num_subgraphs)

def move_signal(x): 
   # Lấy label của các node hàng xóm
   neighbor_labels = [assignment_dict_bd.value[i] for i in G_boardcast.value.neighbors(x)] if x in G_boardcast.value else []
   # Chọn label xuất hiện nhiều nhất
   new_label = Counter(neighbor_labels).most_common(1)[0][0]
   # return node, partition cũ, partition mới muốn chuyển đến
   return (x, assignment_dict_bd.value[x], new_label)

def actual_move(x): 
   # Nếu parition cũ = parition mới muốn chuyển đến
   if x[1] == x[2]:
      return x[1]
   else:
      prob_to_move =  move_dict_bd.value[(x[1],x[2])]
      if prob_to_move == 1:
         return x[2]
      elif prob_to_move == 0:
         return x[1]
      elif random.uniform(0,1) < prob_to_move:
         return x[2]
      else:
         return x[1]

for step in range(20):
   assignment_dict_bd = sc.broadcast(assignment_dict)

   my_list_rdd = sc.parallelize([i for i in G.nodes()]).map(lambda x: move_signal(x))
   # all_transitions là list các (node, partition cũ, parititon mới muốn chuyển đến)
   all_transitions = my_list_rdd.collect()
   my_list_rdd = sc.parallelize(all_transitions).filter(lambda x: x[1]!=x[2]).map(lambda x: ((x[1],x[2]), [x[0]]) ).reduceByKey(lambda a, b: a + b)
   transitions = my_list_rdd.collect()
   
   # transitions_dict là dict có key là (parition cũ, partition mới muốn chuyển đến), value là list các node muốn chuyển
   transitions_dict = dict(transitions)
   transitions_len_dict = {k:len(v) for k,v in transitions_dict.items()}
   # xác suất cho 1 node chuyển từ parition cũ i -> parition mới j = mij/min(mị, mji)
   move_dict = {k : min( transitions_len_dict[k], transitions_len_dict.get( ( k[1],k[0] ), 0) )/transitions_len_dict[k] for k in transitions_len_dict }
   move_dict_bd = sc.broadcast(move_dict)
   my_list_rdd = sc.parallelize(all_transitions).map(lambda x: actual_move(x))

   labels = my_list_rdd.collect()
   assignment_dict = {j:labels[i] for i,j in enumerate(G.nodes())}
   print("=================================")
   print(step)
   print_cut(G, labels, args.num_subgraphs)


clusters = defaultdict(list)
for node, label in assignment_dict.items():
   clusters[label].append(node)

assignment_dict_bd = sc.broadcast(assignment_dict)

def count_cut(x):
   count = 0
   for node in G_boardcast.value.neighbors(x):
      count += int(assignment_dict_bd.value[x] == assignment_dict_bd.value[node])
   return x, count 

my_list_rdd = sc.parallelize([i for i in G.nodes()]).map(lambda x: count_cut(x))
count_cut_dict = dict(my_list_rdd.collect())
sorted_anchors = sorted(count_cut_dict.items(),
                        key=lambda x : x[1], reverse=True)

num_anchors = args.num_anchors if args.num_anchors_percent is None else int(args.num_anchors_percent*len(G)) 
anchors = sorted([i[0] for i in sorted_anchors[:num_anchors]])
anchors_bd = sc.broadcast(list(map(str, anchors)))

subgraphs = []
for cluster in clusters.values():
   subgraph = G.subgraph(cluster+anchors)
   subgraphs.append(subgraph)
   print(len(subgraph))

num_paths_bd = sc.broadcast(args.num_walks)
path_length_bd = sc.broadcast(args.walk_length)
dim_size_bd = sc.broadcast(args.dim_size)
window_size_bd = sc.broadcast(args.window_size)
num_workers_bd = sc.broadcast(args.num_workers)

def train_embedding(subgraph):
   # return DeepWalk(subgraph, 
   #                num_paths_bd.value, 
   #                path_length_bd.value,
   #                dim_size_bd.value,
   #                window_size_bd.value,
   #                num_workers_bd.value).model
   return embed_by_deepwalk(subgraph, 
                  dim_size_bd.value,
                  num_paths_bd.value, 
                  path_length_bd.value,
                  num_workers_bd.value)

my_list_rdd = sc.parallelize(subgraphs).map(lambda x: train_embedding(x))
models = my_list_rdd.collect()
anchors_emb = models[0].wv[list(map(str, anchors))]
anchors_emb = anchors_emb/np.sqrt(np.sum(anchors_emb**2,axis=1,keepdims=True))

model_anchor_bd = sc.broadcast(anchors_emb)

def transform(x):
   anchors_emb = x.wv[anchors_bd.value]
   anchors_emb = anchors_emb/np.sqrt(np.sum(anchors_emb**2,axis=1,keepdims=True))

   trans_matrix, _, _,_ = np.linalg.lstsq(anchors_emb, model_anchor_bd.value, rcond=-1)
   nodes = list(x.wv.vocab.keys())
   embeddings = x.wv[nodes]
   embeddings = embeddings/np.sqrt(np.sum(embeddings**2,axis=1,keepdims=True))

   new_embeddings = np.matmul(embeddings, trans_matrix)
   return dict(zip(nodes, new_embeddings))

my_list_rdd = sc.parallelize(models).map(lambda x: transform(x))
embedding_dicts = my_list_rdd.collect()
sc.stop()

merged_embedding_dict = embedding_dicts[-1]
for embedding_dict in reversed(embedding_dicts[:-1]):
   merged_embedding_dict.update(embedding_dict)


scores = eval_classify(merged_embedding_dict, args.data_path, args.train_percent, args.seed)
print(scores)