import os
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/poincare-embeddings/python-eggs"

import random
import numpy as np
import networkx as nx
import pandas as pd
import argparse
from tqdm import tqdm
from heat.utils import load_data

def write_edgelist_to_file(edgelist, file):
	with open(file, "w+") as f:
		for u, v in edgelist:
			f.write("{}\t{}\n".format(u, v))

def split_edges(edges, graph,pivot_time):

	nodes_in_training=set()
	test_edges=[]
	train_edges=[]
	val_edges=[]
	val_non_edges = []
	for edge in edges:
		if(edge[2]['time']>pivot_time):
			test_edges.append((edge[0],edge[1]))
		else:
			train_edges.append((edge[0],edge[1]))
			nodes_in_training.add(edge[0])
			nodes_in_training.add(edge[1])
	print("before ",len(test_edges))
	test_edges=[edge for edge in test_edges if (edge[0] in nodes_in_training and edge[1] in nodes_in_training)]
	print("after ",len(test_edges))
	random.seed(0)
	if(len(graph.nodes())<20000):
		non_edges = list(nx.non_edges(graph))
		test_non_edges =random.sample(non_edges,len(test_edges))
	else:
		print("Using another method")
		test_non_edges=[]
		nodes = graph.nodes()
		N=len(test_edges)
		with tqdm(total=N, desc='False Edges', unit='false_edge') as pbar:
			while len(test_non_edges)<N:
				random_edge = sorted(np.random.choice(nodes, 2, replace=False))
				if random_edge[1] not in graph[random_edge[0]] and random_edge not in test_non_edges:
					test_non_edges.append(random_edge)
					pbar.update(1)
	print("check split edges:")
	print(len(val_edges))
	print(len(test_edges))
	print(len(test_non_edges))
	print(len(train_edges))



	return train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges)
def multigraph2graph(multi_graph_nx):
	'''
	convert a multi_graph into a graph, where a multi edge becomes a singe weighted edge
	Args:
		multi_graph_nx: networkx - the given multi_graph

	Returns:
		networkx graph
	'''
	if type(multi_graph_nx) == nx.Graph or type(multi_graph_nx) == nx.DiGraph:
		print("No worries, No change")
		return multi_graph_nx
	graph_nx = nx.DiGraph() if multi_graph_nx.is_directed() else nx.Graph()

	if len(multi_graph_nx.nodes()) == 0:
		return graph_nx

	# add edges + attributes
	for u, v, data in multi_graph_nx.edges(data=True):
		data['weight'] = data['weight'] if 'weight' in data else 1.0

		if graph_nx.has_edge(u, v):
			graph_nx[u][v]['weight'] += data['weight']
		else:
			graph_nx.add_edge(u, v, **data)

	# add node attributes
	for node, attr in multi_graph_nx.nodes(data=True):
		if node not in graph_nx:
			continue
		graph_nx.nodes[node].update(attr)

	return graph_nx
def get_graph_T(graph_nx, min_time=-np.inf, max_time=np.inf, return_df=False):
	'''
	Given a graph with a time attribute for each edge, return the subgraph with only edges between an interval.
	Args:
		graph_nx: networkx - the given graph
		min_time: int - the minimum time step that is wanted. Default value -np.inf
		max_time: int - the maximum time step that is wanted. Default value np.inf
		return_df: bool - if True, return a DataFrame of the edges and attributes,
						  else, a networkx object

	Returns:
		sub_graph_nx: networkx - subgraph with only edges between min_time and max_time
	'''
	relevant_edges = []
	attr_keys = []

	if len(graph_nx.nodes()) == 0:
		return graph_nx

	for u, v, attr in graph_nx.edges(data=True):
		if min_time < attr['time'] and attr['time'] <= max_time:
			relevant_edges.append((u, v, *attr.values()))

			if attr_keys != [] and attr_keys != attr.keys():
				raise Exception('attribute keys in \'get_graph_T\' are different')
			attr_keys = attr.keys()

	graph_df = pd.DataFrame(relevant_edges, columns=['from', 'to', *attr_keys])

	if return_df:
		node2label = nx.get_node_attributes(graph_nx, 'label')
		if len(node2label) > 0:
			graph_df['from_class'] = graph_df['from'].map(lambda node: node2label[node])
			graph_df['to_class'] = graph_df['to'].map(lambda node: node2label[node])
		return graph_df
	else:
		sub_graph_nx = nx.from_pandas_edgelist(graph_df, 'from', 'to', list(attr_keys), create_using=type(graph_nx)())

		# add node attributes
		for node, attr in graph_nx.nodes(data=True):
			if node not in sub_graph_nx:
				continue
			sub_graph_nx.nodes[node].update(attr)

		return sub_graph_nx


def get_graph_times(graph_nx):
	'''
	Return all times in the graph edges attributes
	Args:
		graph_nx: networkx - the given graph

	Returns:
		list - ordered list of all times in the graph
	'''
	return np.sort(np.unique(list(nx.get_edge_attributes(graph_nx, 'time').values())))

def get_pivot_time(graph_nx, wanted_ratio=0.2, min_ratio=0.1):
	'''
	Given a graph with 'time' attribute for each edge, calculate the pivot time that gives
	a wanted ratio to the train and test edges
	Args:
		graph_nx: networkx - Graph
		wanted_ratio: float - number between 0 and 1 representing |test|/(|train|+|test|)
		min_ratio: float - number between 0 and 1 representing the minimum value of the expected ratio

	Returns:
		pivot_time: int - the time step that creates such deviation
	'''
	times = get_graph_times(graph_nx)
	if wanted_ratio == 0:
		return times[-1]

	time2dist_from_ratio = {}
	for time in times[int(len(times) / 3):]:
		train_graph_nx = multigraph2graph(get_graph_T(graph_nx, max_time=time))
		num_edges_train = len(train_graph_nx.edges())

		test_graph_nx = get_graph_T(graph_nx, min_time=time)
		print(time," before :",len(test_graph_nx.edges()))
		test_graph_nx.remove_nodes_from([node for node in test_graph_nx if node not in train_graph_nx])
		test_graph_nx = multigraph2graph(test_graph_nx)
		num_edges_test = len(test_graph_nx.edges())
		print(time," after :",len(test_graph_nx.edges()))
		
		current_ratio = num_edges_test / (num_edges_train + num_edges_test)
		print(time,"   ",current_ratio)
		if current_ratio <= min_ratio:
			continue

		time2dist_from_ratio[time] = np.abs(wanted_ratio - current_ratio)

	pivot_time = min(time2dist_from_ratio, key=time2dist_from_ratio.get)

	print(f'pivot time {pivot_time}, is close to the wanted ratio by {round(time2dist_from_ratio[pivot_time], 3)}')

	return pivot_time

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Script to remove edges for link prediction experiments")

	parser.add_argument("--edgelist", dest="edgelist", type=str, default=None,
		help="edgelist to load.")
	parser.add_argument("--features", dest="features", type=str, default=None,
		help="features to load.")
	parser.add_argument("--labels", dest="labels", type=str, default=None,
		help="path to labels")
	parser.add_argument("--output", dest="output", type=str, default=None,
		help="path to save training and removed edges")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')

	parser.add_argument("--seed", type=int, default=0)

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	seed= args.seed
	training_edgelist_dir = os.path.join(args.output, "training_edges")
	removed_edges_dir = os.path.join(args.output,  "removed_edges")

	if not os.path.exists(training_edgelist_dir):
		os.makedirs(training_edgelist_dir, exist_ok=True)
	if not os.path.exists(removed_edges_dir):
		os.makedirs(removed_edges_dir, exist_ok=True)

	training_edgelist_fn = os.path.join(training_edgelist_dir, "edgelist.tsv")
	val_edgelist_fn = os.path.join(removed_edges_dir, "val_edges.tsv")
	val_non_edgelist_fn = os.path.join(removed_edges_dir, "val_non_edges.tsv")
	test_edgelist_fn = os.path.join(removed_edges_dir, "test_edges.tsv")
	test_non_edgelist_fn = os.path.join(removed_edges_dir, "test_non_edges.tsv")
	
	graph, _, _ = load_data(args)
	print("loaded dataset")
	pivot_time=get_pivot_time(graph, wanted_ratio=0.2, min_ratio=0.1)
	print(pivot_time)
	edges = list(graph.edges.data())
	#non_edges = list(nx.non_edges(graph))

	_, (val_edges, val_non_edges), (test_edges, test_non_edges) = split_edges(edges, graph, pivot_time)

	for edge in test_edges:
		assert edge in graph.edges() or edge[::-1] in graph.edges()

	graph.remove_edges_from(val_edges + test_edges)
	#graph.add_edges_from(((u, u, {"weight": 0}) for u in graph.nodes())) # ensure that every node appears at least once by adding self loops
	####
	#daoluan=test_edges+test_non_edges
	#test_edges=random.sample(daoluan,len(test_edges))
	#test_non_edges=random.sample(daoluan,len(test_edges))
	####
	print ("removed edges")

	nx.write_edgelist(graph, training_edgelist_fn, delimiter="\t", data=["time"])
	write_edgelist_to_file(val_edges, val_edgelist_fn)
	write_edgelist_to_file(val_non_edges, val_non_edgelist_fn)
	write_edgelist_to_file(test_edges, test_edgelist_fn)
	write_edgelist_to_file(test_non_edges, test_non_edgelist_fn)

	print ("done")

if __name__ == "__main__":
	main()
