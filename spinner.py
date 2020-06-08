import networkx as nx
from pyspark import SparkContext
import random 
from collections import Counter, defaultdict

from deepwalk import DeepWalk
from utils import load_data, print_cut
import sys 

c = 1.01
sc = SparkContext("local", "Accumulator app") 
G = load_data(sys.argv[1])

n_label = 5
C = c*len(G.edges())/n_label
C_bd = sc.broadcast(C)
n_node_per_label = len(G)//n_label
labels = []
for i in range(n_label-1):
   labels += [i]*n_node_per_label
labels += [n_label-1]*(len(G)-(n_label-1)*n_node_per_label)
random.shuffle(labels)
assignment_dict = {j:labels[i] for i,j in enumerate(G.nodes())}
assignment_dict_bd = sc.broadcast(assignment_dict)

G_boardcast = sc.broadcast(G)
my_list_rdd = sc.parallelize([i for i in G.nodes()]).map(lambda x: (assignment_dict_bd.value[x],G_boardcast.value.degree(x))).reduceByKey(lambda a, b: a + b)
part_size = my_list_rdd.collect()
part_size = dict(part_size)
part_size = {k:v/C for k,v in part_size.items()}
part_size_bd = sc.broadcast(part_size)

def move_signal(x): 
   # Lấy label của các node hàng xóm
   neighbor_labels = [assignment_dict_bd.value[i] for i in G_boardcast.value.neighbors(x)] if x in G_boardcast.value else []
   # Chọn label xuất hiện nhiều nhất
   neigh_label_freq = dict(Counter(neighbor_labels))
   neigh_label_freq = {k:v/G_boardcast.value.degree(x)- part_size_bd.value[k] for k,v in neigh_label_freq.items()}
   best_label, best_score = max(neigh_label_freq.items(), key=lambda x:x[1])
   if best_score > neigh_label_freq.get(assignment_dict_bd.value[x],0):
      new_label = best_label
   else:
      new_label = assignment_dict_bd.value[x]
   # return node, partition cũ, partition mới muốn chuyển đến
   return (x, assignment_dict_bd.value[x], new_label)

def actual_move(x): 
   # Nếu parition cũ = parition mới muốn chuyển đến
   # if x[1] == x[2]:
   #    return x[1]
   # else:
   #    prob_to_move =  move_dict_bd.value[(x[1],x[2])]
   #    if prob_to_move == 1:
   #       return x[2]
   #    elif prob_to_move == 0:
   #       return x[1]
   #    elif random.uniform(0,1) < prob_to_move:
   #       return x[2]
   #    else:
   #       return x[1]
   return x[2]


def train_embedding(subgraph):
   return DeepWalk(subgraph)

for _ in range(1000):
   assignment_dict_bd = sc.broadcast(assignment_dict)
   my_list_rdd = sc.parallelize([i for i in G.nodes()]).map(lambda x: (assignment_dict_bd.value[x],G_boardcast.value.degree(x))).reduceByKey(lambda a, b: a + b)
   part_size = my_list_rdd.collect()
   part_size = dict(part_size)
   part_size = {k:v/C for k,v in part_size.items()}
   part_size_bd = sc.broadcast(part_size)

   my_list_rdd = sc.parallelize([i for i in G.nodes()]).map(lambda x: move_signal(x))
   # all_transitions là list các (node, partition cũ, parititon mới muốn chuyển đến)
   all_transitions = my_list_rdd.collect()
   my_list_rdd = sc.parallelize(all_transitions).map(lambda x: actual_move(x))

   labels = my_list_rdd.collect()
   assignment_dict = {j:labels[i] for i,j in enumerate(G.nodes())}
   print("=================================")
   print_cut(G, labels, n_label)


# clusters = defaultdict(list)
# for node, label in enumerate(labels):
#    clusters[label].append(node)

# subgraphs = []
# for cluster in clusters.values():
#    subgraph = G.subgraph(cluster)
#    new_subgraph = nx.relabel_nodes(subgraph, {i:i for i in subgraph.nodes()})
#    subgraphs.append(new_subgraph)

# my_list_rdd = sc.parallelize(subgraphs).map(lambda x: train_embedding(x))
# models = my_list_rdd.collect()
# import pdb;pdb.set_trace()
# for model in models:
#    print(model)