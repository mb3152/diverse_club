#!/home/despoB/mb3152/anaconda2/bin/python

# rich club stuff
from richclub import preserve_strength, RC

# graph theory stuff you might have
import brain_graphs
import networkx as nx
import community as louvain
import igraph
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering, arpack_options
arpack_options.maxiter=300000

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
import matplotlib.gridspec as gridspec
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
global cores
cores = 5
algorithms = np.array(['infomap','walktrap','spectral','edge_betweenness','label_propogation','louvain','spin_glass','walktrap_n','louvain_res'])
algorithms = np.array(['infomap','walktrap','edge_betweenness','label_propogation','louvain','spin_glass','walktrap_n','louvain_res'])
def corrfunc(x, y, **kws):
	r, _ = pearsonr(x, y)
	ax = plt.gca()
	ax.annotate("r={:.3f}".format(r) + ",p={:.3f}".format(_),xy=(.1, .9), xycoords=ax.transAxes)

def tstatfunc(x, y,bc=False):
	t, p = scipy.stats.ttest_ind(x,y)
	if bc != False:
		bfc = np.around((p * bc),5)
		p = np.around(p,5)
		if bfc <= 0.05:
			if bfc == 0.0:
				bfc = ' < 1e-5'
			if p == 0.0:
				p = ' < 1e-5'
			return "t=%s \n p=%s \n bf=%s" %(np.around(t,3),p,bfc)
		else:
			return "t=%s" %(np.around(t,3))
	else:
		p = np.around(p,5)
		if p == 0.0:
			p = ' < 1e-5'
		return "t=%s,p=%s" %(np.around(t,3),p)

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
	sources = pd.read_csv('/%s/diverse_club/graphs/routes.dat'%(homedir),header=None)[3].values
	dests = pd.read_csv('/%s/diverse_club/graphs/routes.dat'%(homedir),header=None)[5].values
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
	sources = pd.read_csv('/%s/diverse_club/graphs/routes.dat'%(homedir),header=None)[3].values
	dests = pd.read_csv('/%s/diverse_club/graphs//routes.dat'%(homedir),header=None)[5].values
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
	v_to_d = []
	for i in range(graph.vcount()):
		if len(graph.subcomponent(i)) < 3000:
			v_to_d.append(i)
	graph.delete_vertices(v_to_d)
	graph.write_gml('/%s/diverse_club/graphs/airport.gml'%(homedir))
	airports = pd.read_csv('/%s/diverse_club/graphs/airports.dat'%(homedir),header=None)
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
	graph.write_gml('/%s/diverse_club/graphs/airport_viz.gml'%(homedir))

def nan_pearsonr(x,y):
	x = np.array(x)
	y = np.array(y)
	isnan = np.sum([x,y],axis=0)
	isnan = np.isnan(isnan) == False
	return pearsonr(x[isnan],y[isnan])

def attack(variables):
	np.random.seed(variables[0])
	vc = variables[1]
	n_attacks = variables[2]
	rankcut = variables[3]
	graph = vc.community.graph
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	deg = np.array(vc.community.graph.strength(weights='weight'))
	connector_nodes = np.argsort(pc)[rankcut:]
	degree_nodes = np.argsort(deg)[rankcut:]
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
		if idx == n_attacks:
			break
	return [attack_pc_sps,attack_degree_sps,healthy_sp]

def check_network(network):
	loops = np.array(network.community.graph.is_loop())
	multiples = np.array(network.community.graph.is_multiple())
	assert network.community.graph.is_connected() == True
	assert np.isnan(network.pc).any() == False
	assert len(loops[loops==True]) == 0.0
	assert len(multiples[multiples==True]) == 0.0
	assert np.min(network.community.graph.degree()) > 0
	assert np.isnan(network.community.graph.degree()).any() == False

def plot_pc_similarity(network):
	def plot_pc_correlations(matrix,labels,nlabels,title,savestr):
		heatmap = sns.heatmap(pd.DataFrame(matrix,columns=labels),square=True,yticklabels=nlabels,xticklabels=nlabels)
		heatmap.set_xticklabels(heatmap.axes.get_xticklabels(),rotation=90,fontsize=6)
		heatmap.set_yticklabels(np.flip(heatmap.axes.get_xticklabels(),0),rotation=360,fontsize=6)
		mean=np.nanmean(matrix[matrix!=1])
		heatmap.set_title(title + ', mean=%s'%(np.around(mean,3)))
		sns.plt.tight_layout()
		sns.plt.savefig(savestr,dpi=600)
		sns.plt.close()
	# for each species, show similarity of pc across algorithms
	if network == 'human':
		network_objects = []
		matrices = []
		for community_alg in algorithms:
			network_objects.append(load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,community_alg)))
		for task in tasks:
			labels = []
			alglabels = []
			plot_network_objects = []
			for i in range(len(network_objects)):
				if algorithms[i] == 'louvain_res':
					subsetiters = np.linspace(.25,.75,15)
				elif algorithms[i] == 'walktrap_n':
					subsetiters = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
				else:
					subsetiters = costs
				label_idx = 0
				for j in range(len(network_objects[i].names)):
					if task in network_objects[i].names[j]:
						plot_network_objects.append(network_objects[i].networks[j])
						labels.append(algorithms[i] + ',' + str(np.around(subsetiters[label_idx],4)))
						alglabels.append(algorithms[i])
						label_idx += 1
			task_pc_matrix = np.zeros((len(plot_network_objects),len(plot_network_objects)))
			for i in range(len(plot_network_objects)):
				for j in range(len(plot_network_objects)):
					task_pc_matrix[i,j] = scipy.stats.spearmanr(plot_network_objects[i].pc,plot_network_objects[j].pc)[0]
			matrices.append(task_pc_matrix)
			savestr = '/%s/diverse_club/figures/pc_similarity_algorithms_human_%s.pdf'%(homedir,task)
			plot_pc_correlations(task_pc_matrix,alglabels,16,'human (%s)'%(task.lower()),savestr)
		savestr = '/%s/diverse_club/figures/pc_similarity_algorithms_human_mean.pdf'%(homedir)
		plot_pc_correlations(np.nanmean(matrices,axis=0),labels,4,'human, mean across tasks',savestr)
	if network == 'f_c_elegans':
		network_objects = []
		matrices = []
		for community_alg in algorithms:
			network_objects.append(load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,community_alg)))
		for worm in ['Worm1','Worm2','Worm3','Worm4']:
			labels = []
			alglabels = []
			plot_network_objects = []
			for i in range(len(network_objects)):
				if algorithms[i] == 'louvain_res':
					subsetiters = np.linspace(.25,.75,15)
				elif algorithms[i] == 'walktrap_n':
					subsetiters = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
				else:
					subsetiters = costs
				label_idx = 0
				for j in range(len(network_objects[i].names)):
					if worm in network_objects[i].names[j]:
						plot_network_objects.append(network_objects[i].networks[j])
						labels.append(algorithms[i] + ',' + str(np.around(subsetiters[label_idx],4)))
						alglabels.append(algorithms[i])
						label_idx += 1
			task_pc_matrix = np.zeros((len(plot_network_objects),len(plot_network_objects)))
			for i in range(len(plot_network_objects)):
				for j in range(len(plot_network_objects)):
					task_pc_matrix[i,j] = scipy.stats.spearmanr(plot_network_objects[i].pc,plot_network_objects[j].pc)[0]
			matrices.append(task_pc_matrix)
			savestr = '/%s/diverse_club/figures/pc_similarity_algorithms_f_c_elegans_%s.pdf'%(homedir,worm)
			plot_pc_correlations(task_pc_matrix,alglabels,16,'functional c elegans, %s'%(worm.lower()),savestr)
		savestr = '/%s/diverse_club/figures/pc_similarity_algorithms_f_c_elegans_mean.pdf'%(homedir)
		plot_pc_correlations(np.nanmean(matrices,axis=0),labels,4,'functional c elegans, mean across worms',savestr)
	if network == 'structural':
		network_objects = []
		matrices = []
		for community_alg in algorithms:
			network_objects.append(load_object('/%s/diverse_club/graphs/graph_objects/structural_networks_0.8_%s.obj'%(homedir,community_alg)))
		for structural_network in ['macaque','c_elegans','US power grid','flight traffic']:
			labels = []
			alglabels = []
			plot_network_objects = []
			for i in range(len(network_objects)):
				if algorithms[i] == 'louvain_res' or algorithms[i] == 'walktrap_n':
					meanpc = []
					for j in range(len(network_objects[i].names)):
						if structural_network in network_objects[i].names[j]:
							meanpc.append(network_objects[i].networks[j].pc)
					plot_network_objects.append(meanpc)
					continue				
				else:
					for j in range(len(network_objects[i].names)):
						if structural_network in network_objects[i].names[j]:
							plot_network_objects.append(network_objects[i].networks[j].pc)
			pc_matrix = np.zeros((len(plot_network_objects),len(plot_network_objects)))
			for i in range(len(plot_network_objects)):
				for j in range(len(plot_network_objects)):
					pc_matrix[i,j] = scipy.stats.spearmanr(plot_network_objects[i],plot_network_objects[j])[0]
			savestr = '/%s/diverse_club/figures/pc_similarity_algorithms_sturctural_%s.pdf'%(homedir,structural_network)
			title = structural_network
			heatmap = sns.heatmap(pc_matrix,square=True,**{'vmin':0})
			heatmap.set_xticklabels(algorithms[:4],rotation=90,fontsize=6)
			heatmap.set_yticklabels(np.flip(algorithms[:4],0),rotation=360,fontsize=6)
			mean=np.nanmean(pc_matrix[pc_matrix!=1])
			heatmap.set_title(title + ', mean=%s'%(np.around(mean,3)))
			sns.plt.tight_layout()
			sns.plt.savefig(savestr,dpi=600)
			sns.plt.show()
			sns.plt.close()

def plot_pc_distribution(network,community_alg,savestrs,bins=10,filter_name=None):
	n_object = load_object('/%s/diverse_club/results/%s_%s_%s_%s_%s.obj'%(homedir,network,0.8,community_alg,True,False))
	
	nconditions = len(filter_names)
	fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
	subplots = subplots.reshape(-1)
	sns.set_style('dark')
	sns.set(rc={'axes.facecolor':'.5','axes.grid': False})
	for sidx,filter_name in enumerate(filter_names):
		pcs = []
		community_sizes = []
		densities = []
		for idx,network in enumerate(n_object.networks):

			if filter_name not in n_object.names[idx]: continue
			pcs.append(network.pc)
			community_sizes.append(len(network.community.sizes()))
			densities.append(np.around(network.community.graph.density(),2))
		
		pcs = np.array(pcs)
		community_sizes = np.array(community_sizes)
		densities = np.array(densities)

		pcs = pcs[np.argsort(community_sizes)]

		colors = sns.light_palette("red",pcs.shape[0],reverse=True)
		n = len(colors)
		sns.plt.sca(subplots[sidx])
		for i in range(pcs.shape[0]):
			f = sns.kdeplot(pcs[i],color=colors[i],**{'alpha':.5})
			f.set_yticklabels('')
			# a = ['','']
			# a = np.append(a,np.arange(1,11)*0.1)
			# f.set_xticklabels(a)
		subplots[sidx].set_title(filter_name.lower())
	if len(subplots) != nconditions:
		fig.delaxes(subplots[-1])
	sns.plt.tight_layout()
	sns.plt.show()
	# sns.plt.savefig(savestr,dpi=600)
	sns.plt.close()
	# 
	# n = len(colors)
	# ax1.imshow(np.arange(n).reshape(1, n),cmap=mpl.colors.ListedColormap(list(colors)),interpolation="nearest", aspect="auto")
	# ax1.set_xticks(np.arange(n) - .5)
	# ax1.set_yticks([-.5, .5])
	# ax1.set_xticklabels(densities)

	# for label in ax1.xaxis.get_ticklabels()[1::5]:
	# 	label.set_visible(True)
	# ax1.set_yticklabels([])
	# ax1.set_title('density')

	for i in range(pcs.shape[0]):
		sns.kdeplot(pcs[i],color=colors[i],ax=ax1,**{'alpha':.25})
	sns.plt.yticks([])
	sns.plt.xticks(np.arange(0,11)*0.1,np.arange(0,11)*0.1)
	sns.plt.tight_layout()
	# sns.plt.savefig(savestr)
	sns.plt.show()

	# hist_matrix = np.zeros((pcs.shape[0],bins))
	# for i in range(pcs.shape[0]):
	# 	hist_matrix[i] = np.histogram(pcs[i],np.arange(0,21)*0.05)[0]
	# sns.heatmap(hist_matrix,ax=ax3)
	# sns.plt.xticks(np.arange(0,21),np.arange(0,21)*0.05)
	# sns.plt.yticks(np.arange(len(community_sizes)),np.flip(community_sizes.astype(int),0))
	# # sns.plt.tight_layout()
	# # sns.plt.savefig(savestr[1])
	# sns.plt.show()


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

def clubness(variables):
	network = variables[0]
	name = variables[1]
	community_alg = variables[2]
	niters = variables[3]
	randomize_topology = variables[4]
	permute_strength = variables[5]
	graph = network.community.graph
	pc = np.array(network.pc)
	assert np.min(graph.strength(weights='weight')) > 0
	assert np.isnan(pc).any() == False
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=pc).phis() for i in range(niters)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	degree_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=graph.strength(weights='weight')).phis() for i in range(niters)],axis=0)
	degree_normalized_phis = degree_emperical_phis/degree_average_randomized_phis
	return np.array(pc_normalized_phis),np.array(degree_normalized_phis)

class Network:
	def __init__(self,networks,rankcut,names,subsets,community_alg):
		"""
		networks: the networks you want to analyze
		rankcut: cutoff for the clubs, in percentile form.
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
		variables = []
		pool = Pool(cores)
		for idx, network in enumerate(self.networks):
			variables.append([network,self.names[idx],self.community_alg,niters,randomize_topology,permute_strength])
		results = pool.map(clubness,variables)
		for idx, result in enumerate(results):	
			# diverse_clubness,rich_clubness = clubness(network,self.names[idx],self.community_alg,niters=niters,randomize_topology=randomize_topology,permute_strength=permute_strength)
			diverse_clubness,rich_clubness = result[0],result[1]
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
	def attack(self,attack_name,n_attacks):
		variables = []
		attack_conditions = []
		for i,network,name in zip(range(len(self.networks)),self.networks,self.names):
			if attack_name == None:
				variables.append([i,network,n_attacks,self.ranks[i]])
				attack_conditions.append(self.names[i].split("_")[0])
				continue
			elif name.split('_')[1] == attack_name:
				variables.append([i,network,n_attacks,self.ranks[i]])
				attack_conditions.append(self.names[i].split("_")[0])
		pool = Pool(cores)
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

def swap(matrix,membership):
	membership = np.array(membership)
	swap_indices = []
	new_membership = np.zeros(len(membership))
	for i in np.unique(membership):
		for j in np.where(membership == i)[0]:
			swap_indices.append(j)
	return swap_indices

def plot_matrices(n,filter_name,savestr):
	to_plot = []
	plot_names = []
	for i,network,name in zip(range(len(n.networks)),n.networks,n.names):
		if filter_name == None:
			to_plot.append(network)
			plot_names.append(name.split('_')[0].lower())
			continue
		elif name.split('_')[1] == filter_name:
			to_plot.append(network)
			plot_names.append(name.split('_')[0].lower())
			continue
	nconditions = len(to_plot)
	fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
	subplots = subplots.reshape(-1)
	
	for idx,tp,name in zip(np.arange(len(to_plot)),to_plot,plot_names):
		matrix = tp.matrix
		sns.set(style='dark',context="paper",font='Helvetica',font_scale=1.2)
		np.fill_diagonal(matrix,0.0)
		membership = np.array(tp.community.membership)
		swap_indices = swap(matrix,membership)
		heatfig = sns.heatmap(matrix[swap_indices,:][:,swap_indices],yticklabels=[''],xticklabels=[''],square=True,rasterized=True,ax=subplots[idx],**{'vmax':1.})
		# Use matplotlib directly to emphasize known networks
		heatfig.set_title(name,fontdict={'fontsize':'large'})
		heatfig.annotate('Q=%s'%(np.around(tp.community.modularity,2)), xy=(1,1),**{'fontsize':'small'})
		for i,network in zip(np.arange(len(membership)),membership[swap_indices]):
			if network != membership[swap_indices][i - 1]:
				heatfig.figure.axes[idx].add_patch(patches.Rectangle((i+len(membership[membership==network]),len(membership)-i),len(membership[membership==network]),len(membership[membership==network]),facecolor="none",edgecolor='black',linewidth="1",angle=180))
	if len(subplots) != nconditions:
		fig.delaxes(subplots[-1])
	sns.plt.tight_layout()
	sns.plt.savefig(savestr,dpi=600)
	sns.plt.close()
	# sns.plt.show()

def plot_attacks(n,savestr):
	n.attacked.condition = n.attacked.condition.str.lower()
	n.attacked = n.attacked.sort_values('condition')
	conditions = np.unique(n.attacked.condition.values)
	conditions.sort()
	df = n.attacked.copy()

	for condition in np.unique(df.condition):
		df['sum of shortest paths'][(df['attack'] == 'diverse') & (df['condition']==condition)] = df['sum of shortest paths'][(df['attack'] == 'diverse') & (df['condition']==condition)] / np.nanmean(df['sum of shortest paths'][(df['attack'] == 'none') & (df['condition']==condition)])
		df['sum of shortest paths'][(df['attack'] == 'rich') & (df['condition']==condition)] = df['sum of shortest paths'][(df['attack'] == 'rich') & (df['condition']==condition)] / np.nanmean(df['sum of shortest paths'][(df['attack'] == 'none') & (df['condition']==condition)])
	df = df[df.attack != 'none']
	sns.set(context="paper",font='Helvetica',style='white')
	df.rename(columns={'sum of shortest paths': 'sum of shortest paths increase'},inplace=True)
	sns.boxplot(data=df,x='condition',hue='attack',y="sum of shortest paths increase",hue_order=['diverse','rich'],palette={'diverse':sns.color_palette()[0],'rich':sns.color_palette()[1]},showfliers=False)
	for i,c in enumerate(conditions):
		stat = tstatfunc(df['sum of shortest paths increase'][(df['attack'] == 'diverse') & (df['condition']==c)],df['sum of shortest paths increase'][(df['attack'] == 'rich') & (df['condition']==c)],len(np.unique(df.condition.values)))
		maxvaly = np.mean(df['sum of shortest paths increase'][df.condition==c]) +(np.std(df['sum of shortest paths increase'][df.condition==c]) * .5)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig(savestr,dpi=3600)
	# sns.plt.show()
	sns.plt.close()

def plot_betweenness(n,savestr):
	n.betweenness = n.betweenness.sort_values('condition')
	n.betweenness.condition = n.betweenness.condition.str.lower()
	conditions = np.unique(n.betweenness.condition.values)
	conditions.sort()
	sns.set(context="paper",font='Helvetica',style='white')
	sns.boxplot(data=n.betweenness,x='condition',hue='club',y="betweenness",hue_order=['diverse','rich'],palette={'diverse':sns.color_palette()[0],'rich':sns.color_palette()[1]},showfliers=False)
	for i,c in enumerate(conditions):
		stat = tstatfunc(n.betweenness['betweenness'][(n.betweenness['club'] == 'diverse') & (n.betweenness['condition']==c)],n.betweenness['betweenness'][(n.betweenness['club'] == 'rich') & (n.betweenness['condition']==c)],len(np.unique(n.betweenness.condition.values)))
		maxvaly = np.mean(n.betweenness['betweenness'][n.betweenness.condition==c]) + (np.std(n.betweenness['betweenness'][n.betweenness.condition==c]) * .5)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig(savestr,dpi=3600)
	# sns.plt.show()
	sns.plt.close()

def plot_edge_betweenness(n,savestr):
	n.edge_betweenness.condition = n.edge_betweenness.condition.str.lower()
	n.edge_betweenness = n.edge_betweenness.sort_values('condition')
	conditions = np.unique(n.edge_betweenness.condition.values)
	conditions.sort()
	
	sns.set(context="paper",font='Helvetica',style='white')
	sns.boxplot(data=n.edge_betweenness,x='condition',hue='club',y="edge_betweenness",hue_order=['diverse','rich'],palette={'diverse':sns.color_palette()[0],'rich':sns.color_palette()[1]},showfliers=False)
	for i,c in enumerate(conditions):
		stat = tstatfunc(n.edge_betweenness['edge_betweenness'][(n.edge_betweenness['club'] == 'diverse') & (n.edge_betweenness['condition']==c)],n.edge_betweenness['edge_betweenness'][(n.edge_betweenness['club'] == 'rich') & (n.edge_betweenness['condition']==c)],len(np.unique(n.edge_betweenness.condition.values)))
		maxvaly = np.mean(n.edge_betweenness['edge_betweenness'][n.edge_betweenness.condition==c]) + (np.std(n.edge_betweenness['edge_betweenness'][n.edge_betweenness.condition==c]) * .5)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig(savestr,dpi=3600)
	# sns.plt.show()
	sns.plt.close()

def plot_clubness(n,savestr):
	sns.set(context="paper",font='Helvetica',style='white')
	nconditions = len(np.unique(n.clubness[n.subsets[0]]))
	fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
	subplots = subplots.reshape(-1)
	for idx,diffnetwork in enumerate(np.unique(n.clubness[n.subsets[0]])):
		ax = sns.tsplot(data=n.clubness[(n.clubness[n.subsets[0]]==diffnetwork)],condition='club',color={'diverse':sns.color_palette()[0],'rich':sns.color_palette()[1]},unit=n.subsets[1],time='rank',value='clubness',ax=subplots[idx])
		ax.set_title(diffnetwork.lower(),fontdict={'fontsize':'large'})
		ax.set_xticklabels(np.around(np.linspace(0,95,len(ax.get_xticklabels()))))
		n_nodes = int(ax.lines[0].get_data()[0][-1] + 1)
		xcutoff = int(n_nodes*1)
		ax.set_xlim(0,xcutoff)
		ylims_array = np.array([ax.lines[1].get_data()[1][:xcutoff],ax.lines[0].get_data()[1][:xcutoff]])
		ymax = np.nanmax(ylims_array)
		ymin = np.nanmin(ylims_array)
		ax.set_ylim(ymin,ymax)
		ax.set_xlabel('percentile cut off for club')
	if len(subplots) != nconditions:
		fig.delaxes(subplots[-1])
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig(savestr)
	# sns.plt.show()
	sns.plt.close()

def plot_intersect(n,savestr):
	n.intersect.condition = n.intersect.condition.str.lower()
	sns.barplot(data=n.intersect,x='percent overlap',y='condition')
	sns.plt.xlim((0,1))
	sns.plt.savefig(savestr.replace('REPLACE','percent_overlap'))
	sns.plt.close()
	sns.barplot(data=n.intersect,x='percent community, diverse club',y='condition')
	sns.plt.xlim((0,1))
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	sns.plt.savefig(savestr.replace('REPLACE','percent_community_diverse'))
	sns.plt.close()
	sns.barplot(data=n.intersect,x='percent community, rich club',y='condition')
	sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	sns.plt.xlim((0,1))
	sns.plt.savefig(savestr.replace('REPLACE','percent_community_rich'))
	sns.plt.close()
	newintdf = pd.DataFrame()
	for t in n.intersect.condition.values:
		newintdf = newintdf.append(pd.DataFrame({'condition':t,'percent community':n.intersect['percent community, diverse club'][n.intersect.condition==t],'club':'diverse'}),ignore_index=True)
		newintdf = newintdf.append(pd.DataFrame({'condition':t,'percent community':n.intersect['percent community, rich club'][n.intersect.condition==t],'club':'rich'}),ignore_index=True)
	sns.barplot(data=newintdf,x='percent community',y='condition',hue='club',palette={'diverse':sns.color_palette()[0],'rich':sns.color_palette()[1]})
	sns.plt.savefig(savestr.replace('REPLACE','percent_community_combined'))
	sns.plt.close()

def human_network_membership(n):
	sns.set(context="paper",font='Helvetica',style='white')
	network_names = np.array(pd.read_csv('%s/diverse_club/Consensus264.csv'%(homedir),header=None)[36].values)
	network_df = pd.DataFrame(columns=["club", "network",'number of club nodes'])
	for idx,network in enumerate(n.networks):
		rich_rank = np.argsort(network.community.graph.strength(weights='weight'))[n.ranks[idx]:]
		diverse_rank = np.argsort(network.pc)[n.ranks[idx]:]
		for network_name in network_names:
			diverse_networks = network_names[diverse_rank]==network_name
			rich_networks = network_names[rich_rank]==network_name
			network_df = network_df.append({'task':n.names[idx].split('_')[0],'club':'diverse','network':network_name.lower(),'number of club nodes':len(diverse_networks[diverse_networks==True])},ignore_index=True)
			network_df = network_df.append({'task':n.names[idx].split('_')[0],'club':'rich', 'network':network_name.lower(),'number of club nodes':len(rich_networks[rich_networks==True])},ignore_index=True)
	sns.barplot(data=network_df,x='network',y='number of club nodes',hue='club',hue_order=['diverse','rich'],palette={'diverse':sns.color_palette()[0],'rich':sns.color_palette()[1]})
	sns.plt.xticks(rotation=90)
	sns.plt.ylabel('mean number of club nodes in network')
	sns.plt.title('rich and diverse club network membership across tasks and costs')
	sns.plt.tight_layout()
	sns.plt.savefig('/%s/diverse_club/figures/human_both_club_membership_%s.pdf'%(homedir,n.community_alg),dpi=3600)
	sns.plt.close()

def partition_network(variables):
	matrix = variables[0]
	community_alg = variables[1]
	cost = variables[2]
	make_graph= variables[3]
	if make_graph == True:
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
	elif make_graph == False:
		graph = matrix
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
		vc = brain_graphs.brain_graph(graph.community_spinglass(weights='weight'))
	if community_alg == 'louvain':		
		louvain_vc = louvain.best_partition(nxg)
		vc = brain_graphs.brain_graph(VertexClustering(graph,louvain_vc.values()))
	if community_alg == 'louvain_res':		
		louvain_vc = louvain.best_partition(nxg,resolution=cost)
		vc = brain_graphs.brain_graph(VertexClustering(graph,louvain_vc.values()))
	return vc

def make_networks(network,rankcut,community_alg):
	if network == 'human':
		try: n = load_object('/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
		except:
			if community_alg == 'louvain_res':
				subsets = ['task','resolution']
				subsetiters = np.linspace(.25,.75,15)
			elif community_alg == 'walktrap_n':
				subsets = ['task','n_clusters']
				subsetiters = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
			else:
				subsets = ['task','cost']
				subsetiters = costs
			variables = []
			names = []
			pool = Pool(cores)
			for task in tasks:
				matrix = np.load('/%s/diverse_club/graphs/%s.npy'%(homedir,task))
				for idx,cost in enumerate(subsetiters):
					variables.append([matrix,community_alg,cost,True])
					names.append('%s_%s'%(task.lower(),cost))
			networks = pool.map(partition_network,variables)
			n = Network(networks,rankcut,names,subsets,community_alg)
			save_object(n,'/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
	if network == 'f_c_elegans':
		try: n = load_object('/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
		except:	
			if community_alg == 'louvain_res':
				subsets = ['worm','resolution']
				subsetiters = np.linspace(.25,.75,15)
			elif community_alg == 'walktrap_n':
				subsets = ['worm','n_clusters']
				subsetiters = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
			else:
				subsets = ['worm','cost']
				subsetiters = costs
			worms = ['Worm1','Worm2','Worm3','Worm4']
			variables = []
			names = []
			pool = Pool(cores)
			for worm in worms:
				matrix = np.array(pd.read_excel('/%s/diverse_club/graphs/pnas.1507110112.sd01.xls'%(homedir),sheetname=worm).corr())[4:,4:]
				for idx,cost in enumerate(subsetiters):
					variables.append([matrix,community_alg,cost,True])
					names.append('%s_%s'%(worm,cost))
			networks = pool.map(partition_network,variables)
			n = Network(networks,rankcut,names,subsets,community_alg)
			save_object(n,'/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))	
	if network == 'structural_networks':
		try: n = load_object('/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
		except:
			if community_alg == 'louvain_res':
				subsets = ['stuctural network','resolution']
				subsetiters = np.linspace(.25,.75,15)
			elif community_alg == 'walktrap_n':
				subsets = ['stuctural','n_clusters']
				subsetiters = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
			else:
				subsets = ['stuctural','cost']
				subsetiters = [1]
			raw_networks = [get_macaque_graph(),get_structural_c_elegans_graph(),get_power_graph(),get_airport_graph()]
			raw_names = ['macaque','c_elegans','US power grid','flight traffic']
			names = []
			networks = []
			pool = Pool(cores)
			variables = []
			for nameidx,n in enumerate(raw_networks):
				for cost in subsetiters:
					variables.append([n,community_alg,cost,False])
					names.append(raw_names[nameidx] + str(cost))
			networks = pool.map(partition_network,variables)
			n = Network(networks,rankcut,names,subsets,community_alg)
			save_object(n,'/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))		
	return n

def rename():
	files = glob.glob('/home/despoB/mb3152/diverse_club/results/**')
	for f in files:
		if f.split('.obj')[0][-5:] == 'False':
			continue
		if f.split('.obj')[0][-4:] == 'True':
			continue
		new = f.split('.obj')[0] + '_True_False.obj'
		os.system ('mv %s %s' %(f,new))

def run_networks(network,run=False,nrandomiters=1000,rankcut=.8,community_alg='infomap',randomize_topology=True,permute_strength=False,plot_it=False):
	# network = 'human'
	# run=False
	# nrandomiters=1000
	# rankcut=.8
	# community_alg='infomap'
	# randomize_topology=True
	# permute_strength=False
	# plot_it=True
	if community_alg == 'louvain_res':
		filter_name = '0.5'
	elif community_alg == 'walktrap_n':
		filter_name = '12'
	else:
		filter_name = '0.1'
	if run == False: n = load_object('/%s/diverse_club/results/%s_%s_%s_%s_%s.obj'%(homedir,network,rankcut,community_alg,randomize_topology,permute_strength))
	else:
		# n = load_object('/%s/diverse_club/results/%s_%s_%s_%s_%s.obj'%(homedir,network,rankcut,community_alg,randomize_topology,permute_strength))
		if network == 'human':
			n = make_networks(network,rankcut,community_alg)
		if network == 'f_c_elegans':
			n = make_networks(network,rankcut,community_alg)
		if network == 'human' or network == 'f_c_elegans':
			if community_alg == 'louvain_res' or community_alg == 'walktrap_n':
				n.attack(None,1000)
			else:
				n.attack('0.2',10000)
		if network == 'structural_networks':
			n = make_networks(network,rankcut,community_alg)
			if community_alg == 'louvain_res' or community_alg == 'walktrap_n':
				n.attack(None,1000)
			else:
				n.attack(None,10000)
		# save_object(n,'/%s/diverse_club/results/%s_%s_%s_%s_%s.obj'%(homedir,network,rankcut,community_alg,randomize_topology,permute_strength))
		# 1/0
		n.calculate_clubness(niters=nrandomiters,randomize_topology=randomize_topology,permute_strength=permute_strength)
		n.calculate_intersect()
		n.calculate_betweenness()
		n.calculate_edge_betweenness()
		save_object(n,'/%s/diverse_club/results/%s_%s_%s_%s_%s.obj'%(homedir,network,rankcut,community_alg,randomize_topology,permute_strength))
	if plot_it == True:
		plot_clubness(n,'%s/diverse_club/figures/%s_clubness_community_alg_%s_rt_%s_ps_%s.pdf'% \
			(homedir,network,community_alg,randomize_topology,permute_strength))
		plot_intersect(n,'%s/diverse_club/figures/%s_REPLACE_community_alg_%s_rt_%s_ps_%s.pdf'% \
			(homedir,network,community_alg,randomize_topology,permute_strength))
		plot_betweenness(n,'%s/diverse_club/figures/%s_betweennness_community_alg_%s_rt_%s_ps_%s.pdf'% \
			(homedir,network,community_alg,randomize_topology,permute_strength))
		plot_edge_betweenness(n,'%s/diverse_club/figures/%s_edge_betweennness_community_alg_%s_rt_%s_ps_%s.pdf'% \
			(homedir,network,community_alg,randomize_topology,permute_strength))
		plot_attacks(n,'%s/diverse_club/figures/%s_attacks_community_alg_%s_rt_%s_ps_%s.pdf'% \
			(homedir,network,community_alg,randomize_topology,permute_strength))
		plot_matrices(n,filter_name,'%s/diverse_club/figures/%s_matrices_%s_community_alg_%s_rt_%s_ps_%s.pdf'% \
			(homedir,network,filter_name,community_alg,randomize_topology,permute_strength))
		# if network == 'human':
		# 	human_network_membership(n)

def get_power_graph():
	graph = Graph.Read_GML('/%s/diverse_club/graphs/power.gml'%(homedir))
	graph.es["weight"] = np.ones(graph.ecount())
	return graph

def get_structural_c_elegans_graph():
	graph = Graph.Read_GML('%s/diverse_club/graphs/celegansneural.gml'%(homedir))
	graph.es["weight"] = np.ones(graph.ecount())
	graph.to_undirected()
	graph.es["weight"] = np.ones(graph.ecount())
	matrix = np.array(graph.get_adjacency(attribute='weight').data)
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	return graph 

def get_macaque_graph():
	matrix = loadmat('/%s/diverse_club/graphs/macaque.mat'%(homedir))['CIJ']
	graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
	return graph

def get_airport_graph():
	return igraph.read('/%s/diverse_club/graphs/airport.gml'%(homedir))

def airport_analyses():
	graph = get_airport_graph()
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

def submit_2_sge(network='human',cores=cores):
	for algorithm in algorithms:
		if algorithm != 'spectral': continue
		command = 'qsub -pe threaded %s -V -l mem_free=1G -j y -o /%s/diverse_club/sge/ -e /%s/diverse_club/sge/ -N %s diverse_club.py run %s %s ' \
		%(cores,homedir,homedir,algorithm,network,algorithm)
		os.system(command)

# run_networks('f_c_elegans',run=True,nrandomiters=1000,rankcut=.8,community_alg='spectral',randomize_topology=True,permute_strength=False,plot_it=False)

# for community_alg in algorithms:
	# if community_alg != 'louvain_res': continue
	# run_networks('human',run=False,nrandomiters=1000,rankcut=.8,community_alg=community_alg,randomize_topology=True,permute_strength=False,plot_it=True)
	# run_networks('f_c_elegans',run=False,nrandomiters=1000,rankcut=.8,community_alg=community_alg,randomize_topology=True,permute_strength=False,plot_it=True)

if len(sys.argv) > 1:
	if sys.argv[1] == 'run':
		run_networks(sys.argv[2],run=True,nrandomiters=1000,rankcut=.8,community_alg=sys.argv[3],randomize_topology=True,permute_strength=False)






