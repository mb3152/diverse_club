#!/home/despoB/mb3152/anaconda2/bin/python

# rich club stuff
from richclub import preserve_strength, RC

# graph theory stuff you might have
import brain_graphs
import networkx as nx
import community as louvain
import igraph
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering

#standard stuff you probably have
import pandas as pd
import os
import sys
import time
from collections import Counter
import numpy as np
from quantities import millimeter
import subprocess
import pickle
import random
import scipy
from scipy.io import loadmat
import scipy.io as sio
from scipy.stats.stats import pearsonr
from sklearn.metrics.cluster import normalized_mutual_info_score
from itertools import combinations, permutations
import glob
import math
from multiprocessing import Pool

#plotting
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import patches
plt.rcParams['pdf.fonttype'] = 42
path = '/home/despoB/mb3152/anaconda2/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/Helvetica.ttf'
mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = 'Helvetica'

#some globals that we want to have everywhere
global homedir #set this to wherever you save the repo to, e.g., mine is in /home/despoB/mb3152/diverse_club/
homedir = '/home/despoB/mb3152/'
global tasks #HCP tasks
tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
global costs #for functional connectivity matrices
costs = np.arange(5,21) *0.01
global algorithms

algorithms = np.array(['walktrap','infomap','edge_betweenness','label_propogation','louvain'])

other = ['spin_glass','walktrap_n','spectral','louvain_res']

def mm_2_inches(mm):
	mm = mm * millimeter
	mm.units = 'inches'
	return mm.item()

def load_object(path_to_object):
	f = open('%s' %(path_to_object),'r')
	return pickle.load(f)

def save_object(o,path_to_object):
	f = open(path_to_object,'w+')
	pickle.dump(o,f)
	f.close()

def make_airport_graph():
	vs = []
	sources = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[3].values
	dests = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[5].values
	graph = Graph()
	for s in sources:
		if s in dests:
			continue
		try:
			vs.append(int(s))
		except:
			continue
	for s in dests:
		try:
			vs.append(int(s))
		except:
			continue
	graph.add_vertices(np.unique(vs).astype(str))
	sources = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[3].values
	dests = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[5].values
	for s,d in zip(sources,dests):
		if s == d:
			continue
		try:
			int(s)
			int(d)
		except:
			continue
		if int(s) not in vs:
			continue
		if int(d) not in vs:
			continue
		s = str(s)
		d = str(d)
		eid = graph.get_eid(s,d,error=False)
		if eid == -1:
			graph.add_edge(s,d,weight=1)
		else:
			graph.es[eid]['weight'] = graph.es[eid]["weight"] + 1
	graph.delete_vertices(np.argwhere((np.array(graph.degree())==0)==True).reshape(-1))
	airports = pd.read_csv('/home/despoB/mb3152/dynamic_mod/airports.dat',header=None)
	longitudes = []
	latitudes = []
	for v in range(graph.vcount()):
		latitudes.append(airports[6][airports[0]==int(graph.vs['name'][v])].values[0])
		longitudes.append(airports[7][airports[0]==int(graph.vs['name'][v])].values[0])
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	degree = graph.strength(weights='weight')
	graph.vs['community'] = np.array(vc.community.membership)
	graph.vs['longitude'] = np.array(longitudes)
	graph.vs['latitude'] = np.array(latitudes)
	graph.vs['pc'] = np.array(vc.pc)
	graph.write_gml('/home/despoB/mb3152/diverse_club/graphs/airport_viz.gml')

def airport_analyses():
	graph = igraph.read('/home/despoB/mb3152/diverse_club/graphs/airport.gml')
	pc_int = []
	degree_int = []
	for i in range(2,1000):
		num_int = 0
		for i in range(1,i):
		    if 'Intl' in airports[1][airports[0]==int(graph.vs['name'][np.argsort(pc)[-i]])].values[0]:
		    	num_int = num_int + 1
		print num_int
		pc_int.append(num_int)
		num_int = 0
		for i in range(1,i):
		    if 'Intl' in airports[1][airports[0]==int(graph.vs['name'][np.argsort(graph.strength(weights='weight'))[-i]])].values[0]:
		    	num_int = num_int + 1
		print num_int
		degree_int.append(num_int)

def nan_pearsonr(x,y):
	x = np.array(x)
	y = np.array(y)
	isnan = np.sum([x,y],axis=0)
	isnan = np.isnan(isnan) == False
	return pearsonr(x[isnan],y[isnan])

def plot_corr_matrix(matrix,membership):	
	"""
	matrix: square, whatever you like
	membership: the community (or whatever you like of each node in the matrix)
	colors: the colors of each node in the matrix (same order as membership)
	out_file: save the file here, will supress plotting, do None if you want to plot it.
	line: draw those little lines to divide up communities
	rectangle: draw colored rectangles around each community
	draw legend: draw legend...
	colorbar: colorbar...
	"""
	sns.set(style='dark',context="paper",font='Helvetica',font_scale=1.2)
	std = np.nanstd(matrix)
	mean = np.nanmean(matrix)
	np.fill_diagonal(matrix,0.0)
	fig = sns.heatmap(matrix,yticklabels=[''],xticklabels=[''],cmap=sns.diverging_palette(260,10,sep=10, n=20,as_cmap=True),rasterized=True,**{'vmin':mean - (std*2),'vmax':mean + (std*2)})
	# Use matplotlib directly to emphasize known networks
	for i,network in zip(np.arange(len(membership)),membership):
		if network != membership[i - 1]:
			fig.figure.axes[0].add_patch(patches.Rectangle((i+len(membership[membership==network]),len(membership)-i),len(membership[membership==network]),len(membership[membership==network]),facecolor="none",edgecolor='black',linewidth="2",angle=180))
	sns.plt.show()

def attack(variables):
	np.random.seed(variables[0])
	vc = variables[1]
	graph = vc.community.graph
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	deg = np.array(vc.community.graph.strength(weights='weight'))
	nidx = graph.vcount()/5
	connector_nodes = np.argsort(pc)[-nidx:]
	degree_nodes = np.argsort(deg)[-nidx:]
	healthy_sp = np.sum(np.array(graph.shortest_paths()))
	degree_edges = []
	for d in combinations(degree_nodes,2):
		if graph[d[0],d[1]] > 0:
			degree_edges.append([d])
	connector_edges = []
	for d in combinations(connector_nodes,2):
		if graph[d[0],d[1]] > 0:
			connector_edges.append([d])
	connector_edges = np.array(connector_edges)
	degree_edges = np.array(degree_edges)
	idx = 0
	de = len(connector_edges)
	dmin = int(de*.5)
	dmax = int(de*.9)
	attack_degree_sps = []
	attack_pc_sps = []
	attack_degree_mods = []
	attack_pc_mods = []
	while True:
		num_kill = np.random.choice(range(dmin,dmax),1)[0]
		np.random.shuffle(degree_edges)
		np.random.shuffle(connector_edges)
		d_edges = degree_edges[:num_kill]
		c_edges = connector_edges[:num_kill]
		d_graph = graph.copy()
		c_graph = graph.copy()
		for cnodes,dnodes in zip(c_edges.reshape(c_edges.shape[0],2),d_edges.reshape(d_edges.shape[0],2)):
			cnode1,cnode2,dnode1,dnode2  = cnodes[0],cnodes[1],dnodes[0],dnodes[1]
			d_graph.delete_edges([(dnode1,dnode2)])
			c_graph.delete_edges([(cnode1,cnode2)])
			if d_graph.is_connected() == False or c_graph.is_connected() == False:
				d_graph.add_edges([(dnode1,dnode2)])
				c_graph.add_edges([(cnode1,cnode2)])
		deg_sp = np.array(d_graph.shortest_paths()).astype(float)
		c_sp = np.array(c_graph.shortest_paths()).astype(float)
		attack_degree_sps.append(np.nansum(deg_sp))
		attack_pc_sps.append(np.nansum(c_sp))
		idx = idx + 1
		if idx == 10000:
			break
	return [attack_pc_sps,attack_degree_sps,healthy_sp]

def check_network(network):
	loops = np.array(network.community.graph.is_loop())
	multiples = np.array(network.community.graph.is_multiple())
	assert np.isnan(network.pc).any() == False
	assert len(loops[loops==True]) == 0.0
	assert len(multiples[multiples==True]) == 0.0
	assert np.min(network.community.graph.degree()) > 0
	assert np.isnan(network.community.graph.degree()).any() == False

def pc_distribution_thresh_v_res(matrix,savestr):
	res_rs = np.flip(np.linspace(.25,.75,15),0)
	thresh_rs = np.flip(np.linspace(0.05,0.2,15),0)
	res_pcs = np.zeros((len(rs),matrix.shape[0]))
	res_community_sizes = np.zeros(len(res_rs))
	thresh_pcs = np.zeros((len(rs),matrix.shape[0]))
	thresh_community_sizes = np.zeros(len(res_rs))
	for idx,r in enumerate(thresh_rs):
		vc = partition_network(matrix,'infomap',r)
		print len(vc.community.sizes())
		thresh_community_sizes[idx] = len(vc.community.sizes())
		thresh_pcs[idx] = vc.pc
	for idx,r in enumerate(res_rs):
		vc = partition_network(matrix,'louvain_res',r)
		print len(vc.community.sizes())
		res_community_sizes[idx] = len(vc.community.sizes())
		res_pcs[idx] = vc.pc

	rmatrix = np.zeros((len(res_pcs),len(thresh_pcs)))
	for i in range(len(res_pcs)):
		for j in range(len(thresh_pcs)):
			rmatrix[i,j] = scipy.stats.spearmanr(res_pcs[i],thresh_pcs[j])[0]
	heatmap = sns.heatmap(rmatrix)
	heatmap.set_yticklabels(np.flip(thresh_community_sizes,0).astype(int),rotation=360)
	heatmap.set_xticklabels(res_community_sizes.astype(int))
	sns.plt.savefig(savestr)
	sns.plt.show()

def pc_distribution(matrix,bins,savestrs,community_alg='louvain_res',rs=np.flip(np.linspace(.25,.75,15),0)):
	# rs = np.flip(np.linspace(.25,.75,15),0)
	# graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=1,mst=True)
	# rs = np.flip(np.linspace(0.05,0.2,15),0)
	community_alg = 'louvain'
	pcs = np.zeros((len(rs),graph.vcount()))
	community_sizes = np.zeros(len(rs))
	for idx,r in enumerate(rs):
		vc = partition_network(matrix,community_alg,r)
		print len(vc.community.sizes())
		community_sizes[idx] = len(vc.community.sizes())
		pcs[idx] = vc.pc
	
	sns.set_style('dark')
	sns.set(rc={'axes.facecolor':'.5','axes.grid': False})
	colors = sns.light_palette("red",pcs.shape[0])
	gs = gridspec.GridSpec(10, 10)
	ax1 = plt.subplot(gs[0,:])
	ax2 = plt.subplot(gs[1:,:])
	
	n = len(colors)
	ax1.imshow(np.arange(n).reshape(1, n),cmap=mpl.colors.ListedColormap(list(colors)),interpolation="nearest", aspect="auto")
	ax1.set_xticks(np.arange(n) - .5)
	ax1.set_yticks([-.5, .5])
	ax1.set_xticklabels(community_sizes.astype(int))
	ax1.set_yticklabels([])
	ax1.set_title('community size')

	for i in range(pcs.shape[0]):
		sns.kdeplot(pcs[i],color=colors[i],ax=ax2)
	sns.plt.yticks([])
	sns.plt.xticks(np.arange(0,11)*0.1,np.arange(0,11)*0.1)
	sns.plt.tight_layout()
	# sns.plt.savefig(savestr[0])
	sns.plt.show()

	hist_matrix = np.zeros((pcs.shape[0],bins))
	for i in range(pcs.shape[0]):
		hist_matrix[i] = np.histogram(pcs[i],np.arange(0,21)*0.05)[0]
	sns.heatmap(hist_matrix)
	sns.plt.xticks(np.arange(0,21),np.arange(0,21)*0.05)
	sns.plt.yticks(np.arange(len(community_sizes)),np.flip(community_sizes.astype(int),0))
	sns.plt.tight_layout()
	# sns.plt.savefig(savestr[1])
	sns.plt.show()


	rmatrix = np.zeros((len(pcs),len(pcs)))
	for i,j in permutations(range(len(pcs)),2):
		rmatrix[i,j] = scipy.stats.spearmanr(pcs[i],pcs[j])[0]
	np.fill_diagonal(rmatrix,1)
	heatmap = sns.heatmap(rmatrix)
	heatmap.set_yticklabels(np.flip(community_sizes,0).astype(int),rotation=360)
	heatmap.set_xticklabels(community_sizes.astype(int),rotation=90)
	# sns.plt.savefig(savestr[2])
	sns.plt.show()

def test_threshold(matrix,savestr):
	rs = np.arange(5,21)*.001
	pcs = np.zeros((len(rs),matrix.shape[0]))
	community_sizes = np.zeros(len(rs))
	for idx,r in enumerate(rs):
		graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=r,mst=True)
		nxg = igraph_2_networkx(graph)
		louvain_vc = louvain.best_partition(nxg)
		vc = brain_graphs.brain_graph(VertexClustering(graph,louvain_vc.values()))
		print len(vc.community.sizes())
		community_sizes[idx] = len(vc.community.sizes())
		pcs[idx] = vc.pc
	rmatrix = np.zeros((len(pcs),len(pcs)))
	for i,j in permutations(range(len(pcs)),2):
		rmatrix[i,j] = scipy.stats.spearmanr(pcs[i],pcs[j])[0]
	np.fill_diagonal(rmatrix,1)
	heatmap = sns.heatmap(rmatrix,cmap='RdBu_r',vmin=-1,vmax=1)
	heatmap.set_yticklabels(np.flip(community_sizes,0).astype(int),rotation=360)
	heatmap.set_xticklabels(community_sizes.astype(int),rotation=90)
	sns.plt.savefig(savestr)

def club_intersect(vc,rankcut):
	pc = vc.pc
	assert np.isnan(pc).any() == False
	mem = np.array(vc.community.membership)
	deg = vc.community.graph.strength(weights='weight')
	assert np.isnan(deg).any() == False
	deg = np.argsort(deg)[rankcut:]
	pc = np.argsort(pc)[rankcut:]
	try:
		overlap = float(len(np.intersect1d(pc,deg)))/len(deg)
	except:
		overlap = 0
	return [overlap,len(np.unique(mem[pc]))/float(len(np.unique(mem))),len(np.unique(mem[deg]))/float(len(np.unique(mem)))]

def igraph_2_networkx(graph):
	nxg = nx.Graph()
	for edge,weight in zip(graph.get_edgelist(),graph.es['weight']):
		nxg.add_edge(edge[0],edge[1],{'weight':weight})
	return nxg

def clubness(network,name,community_alg,niters=100,randomize_topology=True,permute_strength=False):
	graph = network.community.graph
	pc = np.array(network.pc)
	assert np.min(graph.strength(weights='weight')) > 0
	assert np.isnan(pc).any() == False
	pc_emperical_phis = RC(graph, scores=pc).phis()
	try: pc_average_randomized_phis = np.load('/%s/diverse_club/random_clubness/%s_%s_%s_%s_%s_diverse.npy'%(homedir,name,community_alg,randomize_topology,permute_strength,niters))
	except:
		pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=pc).phis() for i in range(niters)],axis=0)
		np.save('/%s/diverse_club/random_clubness/%s_%s_%s_%s_%s_diverse.npy'%(homedir,name,community_alg,randomize_topology,permute_strength,niters),pc_average_randomized_phis)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	try: degree_average_randomized_phis = np.load('/%s/diverse_club/random_clubness/%s_%s_%s_%s_%s_rich.npy'%(homedir,name,community_alg,randomize_topology,permute_strength,niters))
	except:
		degree_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=graph.strength(weights='weight')).phis() for i in range(niters)],axis=0)
		np.save('/%s/diverse_club/random_clubness/%s_%s_%s_%s_%s_rich.npy'%(homedir,name,community_alg,randomize_topology,permute_strength,niters),degree_average_randomized_phis)
	degree_normalized_phis = degree_emperical_phis/degree_average_randomized_phis
	return np.array(pc_normalized_phis),np.array(degree_normalized_phis)

class Network:
	def __init__(self,networks,rankcut,names,subsets,community_alg):
		"""
		networks: the networks you want to analyze
		rankcut: cutoff for the clubs, in percent form.
		names: the names for the different networks, 
		seperated by _ for different versions of same network
		subsets: if different version of same network, names for the columns
		e.g., names = ['WM_0.05','WM_0.1'], subsets = ['task','cost']
		community_alg: algorithm used to partition the networks
		"""
		vcounts = []
		for network in networks:
			check_network(network)
			vcounts.append(network.community.graph.vcount())
		self.vcounts = np.array(vcounts)
		self.networks = np.array(networks)
		self.names = np.array(names)
		self.subsets = np.array(subsets)
		rankcuts = np.zeros((len(self.networks)))
		for idx,network in enumerate(self.networks):
			rankcuts[idx] = int(network.community.graph.vcount()*rankcut)
		self.ranks = rankcuts.astype(int)
		self.community_alg = community_alg
	def calculate_clubness(self,niters=100,randomize_topology=True,permute_strength=False):
		for idx,network in enumerate(self.networks):
			diverse_clubness = np.zeros((self.vcounts[idx]))
			rich_clubness = np.zeros((self.vcounts[idx]))
			diverse_clubness,rich_clubness = clubness(network,self.names[idx],self.community_alg,niters=niters,randomize_topology=randomize_topology,permute_strength=permute_strength)
			temp_df = pd.DataFrame()
			temp_df["rank"] = np.arange((self.vcounts[idx]))
			temp_df['clubness'] = diverse_clubness
			temp_df['club'] = 'diverse'
			for cidx,c in enumerate(self.subsets):
				temp_df[c] = self.names[idx].split('_')[cidx]
			if idx == 0: df = temp_df.copy()
			else: df = df.append(temp_df)
			temp_df = pd.DataFrame()
			temp_df["rank"] = np.arange((self.vcounts[idx]))
			temp_df['clubness'] = rich_clubness
			temp_df['club'] = 'rich'
			for cidx,c in enumerate(self.subsets):
				temp_df[c] = self.names[idx].split('_')[cidx]
			df = df.append(temp_df)
		self.clubness = df.copy()
		self.clubness.clubness[self.clubness.clubness==np.inf] = np.nan
	def calculate_intersect(self):
		for idx,network in enumerate(self.networks):
			intersect_results = club_intersect(network,self.ranks[idx])
			temp_df = pd.DataFrame(index=np.arange(1))
			temp_df["percent overlap"] = intersect_results[0]
			temp_df['percent community, diverse club'] = intersect_results[1]
			temp_df['percent community, rich club'] = intersect_results[2]
			temp_df['condition'] = self.names[idx].split("_")[0]
			if idx == 0:
				df = temp_df.copy()
				continue
			df = df.append(temp_df)
		df["percent overlap"] = df["percent overlap"].astype(float)
		df['percent community, diverse club'] = df['percent community, diverse club'].astype(float)
		df['percent community, rich club'] = df['percent community, rich club'].astype(float)
		self.intersect = df.copy()
	def calculate_betweenness(self):
		for idx,network in enumerate(self.networks):
			degree = np.array(network.community.graph.strength(weights='weight'))
			pc = network.pc
			assert np.isnan(pc).any() == False
			assert np.isnan(degree).any() == False
			b = np.array(network.community.graph.betweenness())
			temp_df = pd.DataFrame()
			temp_df["betweenness"] = b[np.argsort(pc)[self.ranks[idx]:]]
			temp_df['club'] = 'diverse'
			temp_df['condition'] = self.names[idx].split("_")[0]
			if idx == 0: df = temp_df.copy()
			else: df = df.append(temp_df)
			temp_df = pd.DataFrame()
			temp_df["betweenness"] = b[np.argsort(degree)[self.ranks[idx]:]]
			temp_df['club'] = 'rich'
			temp_df['condition'] = self.names[idx].split("_")[0]
			df = df.append(temp_df)
		df.betweenness = df.betweenness.astype(float)
		self.betweenness = df.copy()
	def calculate_edge_betweenness(self):
		for idx,network in enumerate(self.networks):
			degree = np.array(network.community.graph.strength(weights='weight'))
			pc = network.pc
			assert np.isnan(pc).any() == False
			assert np.isnan(degree).any() == False

			pc_matrix = np.zeros((self.vcounts[idx],self.vcounts[idx]))
			degree_matrix = np.zeros((self.vcounts[idx],self.vcounts[idx]))
			
			between = network.community.graph.edge_betweenness()

			edge_matrix =  np.zeros((self.vcounts[idx],self.vcounts[idx]))

			diverse_club = np.arange(self.vcounts[idx])[np.argsort(pc)[self.ranks[idx]:]]
			rich_club = np.arange(self.vcounts[idx])[np.argsort(degree)[self.ranks[idx]:]]

			for eidx,edge in enumerate(network.community.graph.get_edgelist()):
				edge_matrix[edge[0],edge[1]] = between[eidx]
				edge_matrix[edge[1],edge[0]] = between[eidx]
				if edge[0] in diverse_club:
					if edge[1] in diverse_club:
						pc_matrix[edge[0],edge[1]] = 1
						pc_matrix[edge[1],edge[0]] = 1
				if edge[0] in rich_club:
					if edge[1] in rich_club:
						degree_matrix[edge[0],edge[1]] = 1
						degree_matrix[edge[1],edge[0]] = 1
			temp_df = pd.DataFrame()
			temp_df["edge_betweenness"] = edge_matrix[pc_matrix>0]
			temp_df['club'] = 'diverse'
			temp_df['condition'] = self.names[idx].split("_")[0]
			if idx == 0: df = temp_df.copy()
			else: df = df.append(temp_df)
			temp_df = pd.DataFrame()
			temp_df["edge_betweenness"] = edge_matrix[degree_matrix>0]
			temp_df['club'] = 'rich'
			temp_df['condition'] = self.names[idx].split("_")[0]
			df = df.append(temp_df)
		df.edge_betweenness = df.edge_betweenness.astype(float)
		self.edge_betweenness = df.copy()
	def attack(self,attack_name):
		variables = []
		attack_conditions = []
		for i,network,name in zip(range(len(self.networks)),self.networks,self.names):
			if attack_name == None:
				variables.append([i,network])
				attack_conditions.append(self.names[i].split("_")[0])
				continue
			elif name.split('_')[1] == attack_name:
				variables.append([i,network])
				attack_conditions.append(self.names[i].split("_")[0])
				print name
		if len(variables) < 21: pool = Pool(len(variables))
		else: pool = Pool(20)
		results = pool.map(attack,variables)
		
		for i,r in enumerate(results):
			attack_diverse_sps = np.array(r[0]).reshape(-1)
			attack_rich_sps = np.array(r[1]).reshape(-1)
			healthy_sps = np.array(r[2]).reshape(-1)
			temp_df = pd.DataFrame()
			temp_df['sum of shortest paths'] = healthy_sps
			temp_df['attack'] = 'none'
			temp_df['condition'] = attack_conditions[i]
			if i == 0: df = temp_df.copy()
			else: df = df.append(temp_df)
			temp_df = pd.DataFrame()
			temp_df['sum of shortest paths'] = attack_diverse_sps
			temp_df['attack'] = 'diverse'
			temp_df['condition'] = attack_conditions[i]
			df = df.append(temp_df)
			temp_df = pd.DataFrame()
			temp_df['sum of shortest paths'] = attack_rich_sps
			temp_df['attack'] = 'rich'
			temp_df['condition'] = attack_conditions[i]
			df = df.append(temp_df)
		self.attacked = df.copy()

def plot_attacks(n):
	n.attacked = n.attacked.sort_values('condition')
	conditions = np.unique(n.attacked.condition.values)
	conditions.sort()
	df = n.attacked.copy()

	for condition in np.unique(df.condition):
		df['sum of shortest paths'][(df['attack'] == 'diverse') & (df['condition']==condition)] = df['sum of shortest paths'][(df['attack'] == 'diverse') & (df['condition']==condition)] / np.nanmean(df['sum of shortest paths'][(df['attack'] == 'none') & (df['condition']==condition)])
		df['sum of shortest paths'][(df['attack'] == 'rich') & (df['condition']==condition)] = df['sum of shortest paths'][(df['attack'] == 'rich') & (df['condition']==condition)] / np.nanmean(df['sum of shortest paths'][(df['attack'] == 'none') & (df['condition']==condition)])
	df = df[df.attack != 'none']
	sns.set(context="paper",font='Helvetica',style='white')
	df.rename(columns={'sum of shortest paths': 'sum of shortest paths increase'}, inplace=True)
	sns.boxplot(data=df,x='condition',hue='attack',y="sum of shortest paths increase",showfliers=False)
	for i,c in enumerate(conditions):
		stat = tstatfunc(df['sum of shortest paths increase'][(df['attack'] == 'diverse') & (df['condition']==c)],df['sum of shortest paths increase'][(df['attack'] == 'rich') & (df['condition']==c)],len(np.unique(df.condition.values)))
		maxvaly = np.mean(df['sum of shortest paths increase'][df.condition==c]) +(np.std(df['sum of shortest paths increase'][df.condition==c]) * .5)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig(savestr,dpi=3600)
	sns.plt.show()
	sns.plt.close()


def plot_betweenness(n,savestr):
	n.betweenness = n.betweenness.sort_values('condition')
	conditions = np.unique(n.betweenness.condition.values)
	conditions.sort()
	
	sns.set(context="paper",font='Helvetica',style='white')
	sns.boxplot(data=n.betweenness,x='condition',hue='club',y="betweenness",palette=[sns.color_palette()[1],sns.color_palette()[0]],showfliers=False)
	for i,c in enumerate(conditions):
		stat = tstatfunc(n.betweenness['betweenness'][(n.betweenness['club'] == 'diverse') & (n.betweenness['condition']==c)],n.betweenness['betweenness'][(n.betweenness['club'] == 'rich') & (n.betweenness['condition']==c)],len(np.unique(n.betweenness.condition.values)))
		maxvaly = np.mean(n.betweenness['betweenness'][n.betweenness.condition==c]) + (np.std(n.betweenness['betweenness'][n.betweenness.condition==c]) * .5)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig(savestr,dpi=3600)
	sns.plt.show()
	sns.plt.close()

def plot_edge_betweenness(n,savestr):
	n.edge_betweenness = n.edge_betweenness.sort_values('condition')
	conditions = np.unique(n.edge_betweenness.condition.values)
	conditions.sort()
	
	sns.set(context="paper",font='Helvetica',style='white')
	sns.boxplot(data=n.edge_betweenness,x='condition',hue='club',y="edge_betweenness",palette=[sns.color_palette()[1],sns.color_palette()[0]],showfliers=False)
	for i,c in enumerate(conditions):
		stat = tstatfunc(n.edge_betweenness['edge_betweenness'][(n.edge_betweenness['club'] == 'diverse') & (n.edge_betweenness['condition']==c)],n.edge_betweenness['edge_betweenness'][(n.edge_betweenness['club'] == 'rich') & (n.edge_betweenness['condition']==c)],len(np.unique(n.edge_betweenness.condition.values)))
		maxvaly = np.mean(n.edge_betweenness['edge_betweenness'][n.edge_betweenness.condition==c]) + (np.std(n.edge_betweenness['edge_betweenness'][n.edge_betweenness.condition==c]) * .5)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig(savestr,dpi=3600)
	sns.plt.show()
	sns.plt.close()

def plot_clubness(n,savestr):
	sns.set(context="paper",font='Helvetica',style='white')
	nconditions = len(np.unique(n.clubness[n.subsets[0]]))
	fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(247)))
	subplots = subplots.reshape(-1)
	for idx,diffnetwork in enumerate(np.unique(n.clubness[n.subsets[0]])):
		ax = sns.tsplot(data=n.clubness[(n.clubness[n.subsets[0]]==diffnetwork)],condition='club',unit=n.subsets[1],time='rank',value='clubness',ax=subplots[idx])
		ax.set_title(diffnetwork.lower(),fontdict={'fontsize':'large'})
		n_nodes = int(ax.lines[0].get_data()[0][-1] + 1)
		xcutoff = int(n_nodes*.95)
		ax.set_xlim(0,xcutoff)
		ax.set_ylim(ax.get_ylim()[0],np.nanmax(ax.lines[0].get_data()[1][:xcutoff]))
	if len(subplots) != nconditions:
		fig.delaxes(subplots[-1])

	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig(savestr)
	sns.plt.close()

def partition_network(matrix,community_alg,cost):
	if community_alg[:7] == 'louvain':
		if community_alg == 'louvain_res':
			graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=1,mst=True)
		if community_alg == 'louvain':
			graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=cost,mst=True)
		nxg = igraph_2_networkx(graph)
	elif community_alg == 'walktrap_n':
		graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=1,mst=True)
	else:
		graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=cost,mst=True)
	if community_alg == 'walktrap_n':
		vc = brain_graphs.brain_graph(graph.community_walktrap(weights='weight').as_clustering(cost))
	if community_alg == 'infomap':
		vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	if community_alg == 'walktrap':
		vc = brain_graphs.brain_graph(graph.community_walktrap(weights='weight').as_clustering())
	if community_alg == 'edge_betweenness':
		vc = brain_graphs.brain_graph(graph.community_edge_betweenness(weights='weight').as_clustering())
	if community_alg == 'spectral':
		vc = brain_graphs.brain_graph(graph.community_leading_eigenvector(weights='weight'))
	if community_alg == 'label_propogation':
		vc = brain_graphs.brain_graph(graph.community_label_propagation(weights='weight'))
	if community_alg == 'spin_glass':
		vc = brain_graphs.brain_graph(graph.community_spin_glass(weights='weight'))
	if community_alg == 'louvain':		
		louvain_vc = louvain.best_partition(nxg)
		vc = brain_graphs.brain_graph(VertexClustering(graph,louvain_vc.values()))
	if community_alg == 'louvain_res':		
		louvain_vc = louvain.best_partition(nxg,resolution=cost)
		vc = brain_graphs.brain_graph(VertexClustering(graph,louvain_vc.values()))
	return vc

def human_network_membership(n):
	network_names = np.array(pd.read_csv('%s/diverse_club/Consensus264.csv'%(homedir),header=None)[36].values)
	network_df = pd.DataFrame(columns=["club", "network",'number of club nodes'])
	for idx,network in enumerate(n.networks):
		rich_rank = np.argsort(network.community.graph.strength(weights='weight'))[n.ranks[idx]:]
		diverse_rank = np.argsort(network.pc)[n.ranks[idx]:]
		for network_name in network_names:
			diverse_networks = network_names[diverse_rank]==network_name
			rich_networks = network_names[rich_rank]==network_name
			network_df = network_df.append({'task':n.names[idx].split('_')[0],'club':'diverse club','network':network_name,'number of club nodes':len(diverse_networks[diverse_networks==True])},ignore_index=True)
			network_df = network_df.append({'task':n.names[idx].split('_')[0],'club':'rich club', 'network':network_name,'number of club nodes':len(rich_networks[rich_networks==True])},ignore_index=True)
	sns.barplot(data=network_df,x='network',y='number of club nodes',hue='club',palette=sns.color_palette(['#7EE062','#050081']))
	sns.plt.xticks(rotation=90)
	sns.plt.ylabel('mean number of club nodes in network')
	sns.plt.title('rich and diverse club network membership across tasks and costs')
	sns.plt.tight_layout()
	sns.plt.savefig('/%s/diverse_club/figures/human_both_club_membership_%s.pdf'%(homedir,n.community_alg),dpi=3600)
	sns.plt.close()

def plot_intersect(n,savestr):
	sns.barplot(data=n.intersect,x='percent overlap',y='condition')
	sns.plt.xlim((0,1))
	sns.plt.savefig('/%s/diverse_club/figures/%s_percent_overlap_human_range_%s.pdf'%(homedir,savestr,n.community_alg))
	sns.plt.close()
	sns.barplot(data=n.intersect,x='percent community, diverse club',y='condition')
	sns.plt.xlim((0,1))
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	sns.plt.savefig('/%s/diverse_club/figures/%s_percent_community_diverse_%s.pdf'%(homedir,savestr,n.community_alg))
	sns.plt.close()
	sns.barplot(data=n.intersect,x='percent community, rich club',y='condition')
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	sns.plt.xlim((0,1))
	sns.plt.savefig('/%s/diverse_club/figures/%s_percent_community_rich_%s.pdf'%(homedir,savestr,n.community_alg))
	sns.plt.close()

def make_networks(network,rankcut,community_alg):
	if network == 'human':
		if community_alg == 'louvain_res':
			subsets = ['task','resolution']
			subsetiters = np.linspace(.25,.75,15)
		elif community_alg == 'walktrap_n':
			subsets = ['task','n_clusters']
			subsetiters = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
		else:
			subsets = ['task','cost']
			subsetiters = costs
		try: 
			n = load_object('/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
		except:
			networks = []
			names = []
			for task in tasks:
				matrix = np.load('/%s/diverse_club/graphs/%s.npy'%(homedir,task))
				for idx,cost in enumerate(subsetiters):
					networks.append(partition_network(matrix,community_alg,cost))
					names.append('%s_%s'%(task,cost))
			n = Network(networks,rankcut,names,subsets,community_alg)
			save_object(n,'/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
	return n

def run_networks(network,nrandomiters=50,attack=False,rankcut=.8,community_alg='infomap',randomize_topology=True,permute_strength=False):
	network='human'
	nrandomiters=100
	rankcut=.80
	community_alg='infomap'
	randomize_topology=True
	permute_strength=False

	if network == 'human':
		n = make_networks(network,rankcut,community_alg)
		
		human_network_membership(n)
		# for t in tasks:
		# 	print t, scipy.stats.ttest_ind(n.edge_betweenness.edge_betweenness[(n.edge_betweenness.club=='diverse')&(n.edge_betweenness.condition==t)],n.edge_betweenness.edge_betweenness[(n.edge_betweenness.club=='rich')&(n.edge_betweenness.condition==t)])
		# 	print t, scipy.stats.ttest_ind(n.betweenness.betweenness[(n.betweenness.club=='diverse')&(n.betweenness.condition==t)],n.betweenness.betweenness[(n.betweenness.club=='rich')&(n.betweenness.condition==t)])
	n.calculate_clubness(niters=nrandomiters,randomize_topology=randomize_topology,permute_strength=permute_strength)
	plot_clubness(n,'%s/diverse_club/figures/human_clubness_community_alg_%s_rt_%s_ps_%s.pdf'% \
		(homedir,community_alg,randomize_topology,permute_strength))
	n.calculate_intersect()
	plot_intersect(n,network)
	n.attack()
	n.calculate_betweenness()
	n.calculate_edge_betweenness()	

def corrfunc(x, y, **kws):
	r, _ = pearsonr(x, y)
	ax = plt.gca()
	ax.annotate("r={:.3f}".format(r) + ",p={:.3f}".format(_),xy=(.1, .9), xycoords=ax.transAxes)

def tstatfunc(x, y,bc=False):
	t, p = scipy.stats.ttest_ind(x,y)
	if bc != False:
		bfc = np.around((p * bc),5)
		if bfc <= 0.05:
			return "t=%s \n p=%s \n bf=%s" %(np.around(t,3),np.around(p,5),bfc)
		else:
			return "t=%s" %(np.around(t,3))
	return "t=%s,p=%s" %(np.around(t,3),np.around(p,5))

def c_elegans_rich_club(plt_mat=False):
	
	df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Worm'])
	plt_mat = False
	draw_graph = False
	import matlab
	import matlab.engine
	eng = matlab.engine.start_matlab()
	eng.addpath('/home/despoB/mb3152/brain_graphs/bct/')
	worms = ['Worm1','Worm2','Worm3','Worm4']
	for worm in worms:
		matrix = np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:]
		print np.isnan(matrix).any()
		avg_pc_normalized_phis = []
		avg_degree_normalized_phis = []
		print matrix.shape
		for cost in np.arange(5,21)*0.01:
			temp_matrix = matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix.copy(),cost=cost,mst=True)
			gmatrix = np.array(graph.get_adjacency(attribute='weight').data)
			
			vc = graph.community_infomap(edge_weights='weight',trials=500)
			pc = brain_graphs.brain_graph(vc).pc
			membership = np.array(vc.membership) + 1
			m_pc = np.array(eng.participation_coef(matlab.double(gmatrix.tolist()),matlab.double(membership.tolist()))).reshape(-1)
			assert np.isclose(pearsonr(m_pc,pc)[0],1.0)
			assert np.isclose(np.max(abs(m_pc-pc)),0.0)
			print pearsonr(m_pc,pc), np.max(abs(m_pc-pc))
			loops = np.array(graph.is_loop())
			assert len(loops[loops==True]) == 0.0
			loops = np.array(graph.is_multiple())
			assert len(loops[loops==True]) == 0.0

			if cost == .1:
				if plt_mat == True:
					np.fill_diagonal(matrix,0.0)
					vc = graph.community_infomap(edge_weights='weight',trials=500)
					plot_corr_matrix(matrix,vc.membership,return_array=False,out_file='/home/despoB/mb3152/dynamic_mod/figures/%s_corr_mat.pdf'%(worm),label=False)
			if graph.is_connected() == False:
				continue
			if cost == .1:
				if draw_graph == True:
					vc = graph.community_infomap(edge_weights='weight',trials=500)
					pc = brain_graphs.brain_graph(vc).pc
					graph.vs['pc'] = pc
					graph.vs['Community'] = vc.membership
					rc_pc = np.array(pc)> np.percentile(pc,80)
					graph.vs['rc_rc'] = rc_pc
					degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(np.array(graph.strength(weights='weight')),80)
					graph.vs['degree_rc'] = degree_rc
					graph.write_gml('ce_gephi_%s.gml'%(worm))

			inters = rich_club_intersect(graph,int(graph.vcount()*.8))
			temp_df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Worm'],index=np.arange(1))
			temp_df["Percent Overlap"] = inters[0]
			temp_df['Percent Community, PC'] = inters[1]
			temp_df['Percent Community, Degree'] = inters[2]
			temp_df['Worm'] = worm
			df = df.append(temp_df)

			degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
			average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True,permute_strength=False),scores=graph.strength(weights='weight')).phis() for i in range(500)],axis=0)
			degree_normalized_phis = degree_emperical_phis/average_randomized_phis
			avg_degree_normalized_phis.append(degree_normalized_phis)
			vc = graph.community_infomap(edge_weights='weight',trials=500)
			pc = brain_graphs.brain_graph(vc).pc
			assert np.min(graph.degree()) > 0
			assert np.isnan(pc).any() == False

			pc[np.isnan(pc)] = 0.0


			pc_emperical_phis = RC(graph, scores=pc).phis()
			pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True,permute_strength=False),scores=pc).phis() for i in range(500)],axis=0)
			pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
			avg_pc_normalized_phis.append(pc_normalized_phis)
		sns.set_style("white")
		sns.set_style("ticks")
		with sns.plotting_context("paper",font_scale=1):	
			sns.tsplot(np.array(avg_degree_normalized_phis)[:,:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=95)
			sns.tsplot(np.array(avg_pc_normalized_phis)[:,:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=95)
			plt.ylabel('Normalized Club Coefficient')
			plt.xlabel('Rank')
			sns.despine()
			plt.legend()
			plt.tight_layout()
			plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/topology_rich_club_%s.pdf'%(worm),dpi=3600)
			plt.close()
	sns.set_style("white")
	sns.set_style("ticks")
	df["Percent Overlap"] = df["Percent Overlap"].astype(float)
	df['Percent Community, PC'] = df['Percent Community, PC'].astype(float)
	df['Percent Community, Degree'] = df['Percent Community, Degree'].astype(float)
	sns.barplot(data=df,x='Percent Overlap',y='Worm')
	sns.plt.xlim((0,1))
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_overlap_fce_0_1.pdf')
	sns.plt.show()
	newintdf = pd.DataFrame()
	for worm in worms:
		newintdf = newintdf.append(pd.DataFrame({'Network':worm,'PercentCommunity':df['Percent Community, PC'][df.Worm==worm],'Club':'Diverse'}),ignore_index=True)
		newintdf = newintdf.append(pd.DataFrame({'Network':worm,'PercentCommunity':df['Percent Community, Degree'][df.Worm==worm],'Club':'Rich'}),ignore_index=True)

	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.barplot(data=newintdf,x='PercentCommunity',y='Network',hue='Club',palette=colors)
	# sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_both_cef.pdf')

	sns.barplot(data=df,x='Percent Community, PC',y='Worm')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_pc_fce.pdf')
	sns.plt.show()
	sns.barplot(data=df,x='Percent Community, Degree',y='Worm')
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_degree_fce.pdf')	
	sns.plt.show()
	
	# degree_normalized_phis = np.nanmean(avg_degree_normalized_phis,axis=0)
	# pc_normalized_phis = np.nanmean(avg_pc_normalized_phis,axis=0)

def power_rich_club(return_graph=False):
	graph = Graph.Read_GML('/home/despoB/mb3152/dynamic_mod/power.gml')
	graph.es["weight"] = np.ones(graph.ecount())
	if return_graph == True:
		return graph
	# inters = rich_club_intersect(graph,(graph.vcount())-graph.vcount()/5)
	# variables = []
	# for i in np.arange(20):
	# 	temp_matrix = matrix.copy()
	# 	graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=1.,mst=True)
	# 	variables.append(graph)		
	# pool = Pool(20)
	# results = pool.map(attack,variables)
	# attack_degree_sps = np.array([])
	# attack_pc_sps = np.array([])
	# healthy_sp = np.array([])
	# for r in results:
	# 	attack_pc_sps = np.append(r[0],attack_pc_sps)
	# 	attack_degree_sps = np.append(r[1],attack_degree_sps)
	# 	healthy_sp = np.append(r[2],healthy_sp)
	# attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
	# attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
	# print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)
	# btw = between_community_centrality(graph)
	# scipy.stats.ttest_ind(btw[0],btw[1])

	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(50)],axis=0)
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	vc = graph.community_infomap(edge_weights='weight',trials=25)
	pc = brain_graphs.brain_graph(vc).pc
	assert np.min(graph.degree()) > 0
	assert np.isnan(pc).any() == False
	pc[np.isnan(pc)] = 0.0
	# graph.vs['pc'] = pc
	# graph.vs['Community'] = vc.membership
	# pc_rc = np.array(pc)>np.percentile(pc,80)
	# graph.vs['pc_rc'] = pc_rc
	# degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
	# graph.vs['degree_rc'] = degree_rc
	# graph.write_gml('power_gephi.gml')
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(50)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(np.array(degree_normalized_phis)[:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=99)
		plt.ylabel('Normalized Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend()
		plt.tight_layout()
		plt.show()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_power.pdf',dpi=3600)
		plt.show()

def c_elegans_str_rich_club():
	graph = Graph.Read_GML('celegansneural.gml')
	graph.es["weight"] = np.ones(graph.ecount())
	print graph.vcount(), graph.ecount()
	graph.to_undirected()
	graph.es["weight"] = np.ones(graph.ecount())
	matrix = np.array(graph.get_adjacency(attribute='weight').data)
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(100)],axis=0)
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	vc = graph.community_infomap(edge_weights='weight',trials=1000)
	pc = brain_graphs.brain_graph(vc).pc
	assert np.min(graph.degree()) > 0
	assert np.isnan(pc).any() == False
	pc[np.isnan(pc)] = 0.0
	# graph.vs['pc'] = pc
	# graph.vs['Community'] = vc.membership
	# pc_rc = np.array(pc)>np.percentile(pc,80)
	# graph.vs['pc_rc'] = pc_rc
	# degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
	# graph.vs['degree_rc'] = degree_rc
	# graph.write_gml('struc_ce_gephi.gml')
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(100)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(np.array(degree_normalized_phis)[:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=99)
		sns.tsplot(np.array(pc_normalized_phis)[:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=99)
		plt.ylabel('Normalized Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/rich_club_structural.pdf',dpi=3600)
		plt.show()

def macaque_rich_club():
	matrix = loadmat('%s.mat'%('macaque'))['CIJ']
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	print graph.vcount(), graph.ecount()
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	degree = graph.strength(weights='weight')
	average_randomized_phis = np.mean([RC(preserve_strength(graph,randomize_topology=True,permute_strength=False),scores=degree).phis() for i in range(5000)],axis=0)
	degree_normalized_phis = degree_emperical_phis/average_randomized_phis
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	pc = vc.pc
	assert np.min(graph.degree()) > 0
	assert np.isnan(pc).any() == False
	pc[np.isnan(pc)] = 0.0
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.mean([RC(preserve_strength(graph,randomize_topology=True,permute_strength=False),scores=pc).phis() for i in range(5000)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	sns.set_style("white")
	sns.set_style("ticks")
	# if animal == 'macaque':
	# 	graph.vs['pc'] = pc
	# 	graph.vs['Community'] = vc.community.membership
	# 	pc_rc = np.array(pc)>np.percentile(pc,80)
	# 	graph.vs['pc_rc'] = pc_rc
	# 	degree_rc = np.array(graph.strength(weights='weight')) > np.percentile(graph.strength(weights='weight'),80)
	# 	graph.vs['degree_rc'] = degree_rc
	# 	graph.write_gml('macaque_gephi.gml')
	with sns.plotting_context("paper",font_scale=1):	
		sns.tsplot(degree_normalized_phis[:-int(graph.vcount()/20.)],color='b',condition='Rich Club',ci=90)
		sns.tsplot(pc_normalized_phis[:-int(graph.vcount()/20.)],color='r',condition='Diverse Club',ci=90)
		plt.ylabel('Normalized Club Coefficient')
		plt.xlabel('Rank')
		sns.despine()
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/topology_rich_club_macaque.pdf',dpi=3600)
		plt.show()
		# plt.close()

def make_airport_graph():
	vs = []
	sources = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[3].values
	dests = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[5].values
	graph = Graph()
	for s in sources:
		if s in dests:
			continue
		try:
			vs.append(int(s))
		except:
			continue
	for s in dests:
		try:
			vs.append(int(s))
		except:
			continue
	graph.add_vertices(np.unique(vs).astype(str))
	sources = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[3].values
	dests = pd.read_csv('/home/despoB/mb3152/dynamic_mod/routes.dat',header=None)[5].values
	for s,d in zip(sources,dests):
		if s == d:
			continue
		try:
			int(s)
			int(d)
		except:
			continue
		if int(s) not in vs:
			continue
		if int(d) not in vs:
			continue
		s = str(s)
		d = str(d)
		eid = graph.get_eid(s,d,error=False)
		if eid == -1:
			graph.add_edge(s,d,weight=1)
		else:
			graph.es[eid]['weight'] = graph.es[eid]["weight"] + 1
	graph.delete_vertices(np.argwhere((np.array(graph.degree())==0)==True).reshape(-1))
	airports = pd.read_csv('/home/despoB/mb3152/dynamic_mod/airports.dat',header=None)
	longitudes = []
	latitudes = []
	for v in range(graph.vcount()):
		latitudes.append(airports[6][airports[0]==int(graph.vs['name'][v])].values[0])
		longitudes.append(airports[7][airports[0]==int(graph.vs['name'][v])].values[0])
	vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	degree = graph.strength(weights='weight')
	graph.vs['community'] = np.array(vc.community.membership)
	graph.vs['longitude'] = np.array(longitudes)
	graph.vs['latitude'] = np.array(latitudes)
	graph.vs['pc'] = np.array(vc.pc)
	graph.write_gml('/home/despoB/mb3152/diverse_club/graphs/airport_viz.gml')

def airport_analyses():
	graph = igraph.read('/home/despoB/mb3152/diverse_club/graphs/airport.gml')
	pc_int = []
	degree_int = []
	for i in range(2,1000):
		num_int = 0
		for i in range(1,i):
		    if 'Intl' in airports[1][airports[0]==int(graph.vs['name'][np.argsort(pc)[-i]])].values[0]:
		    	num_int = num_int + 1
		print num_int
		pc_int.append(num_int)
		num_int = 0
		for i in range(1,i):
		    if 'Intl' in airports[1][airports[0]==int(graph.vs['name'][np.argsort(graph.strength(weights='weight'))[-i]])].values[0]:
		    	num_int = num_int + 1
		print num_int
		degree_int.append(num_int)

def submit_2_sge(network='human'):
	for algorithm in algorithms:
		command = 'qsub -V -l mem_free=8G -j y -o /%s/diverse_club/sge/ -e /%s/diverse_club/sge/ -N %s diverse_club.py run %s %s ' \
		%(homedir,homedir,algorithm,network,algorithm)
		os.system(command)

if len(sys.argv) > 1:
	if sys.argv[1] == 'run':
		run_networks(sys.argv[2],nrandomiters=100,attack=False,rankcut=.8,community_alg=sys.argv[3],randomize_topology=True,permute_strength=False)







