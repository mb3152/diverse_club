#!/home/despoB/mb3152/anaconda2/bin/python

# rich club stuff
from richclub import preserve_strength, RC

# graph theory stuff you might have
import brain_graphs
import networkx as nx
import community as louvain
import igraph
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering, arpack_options
arpack_options.maxiter=300000 #spectral community detection fails without this

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
global algorithm_names
global cores
cores = 2
algorithm_names = np.array(['infomap','walktrap','spectral','edge betweenness','label propogation','louvain','spin glass','walktrap (n)','louvain (resolution)'])
algorithms = np.array(['infomap','walktrap','spectral','edge_betweenness','label_propogation','louvain','spin_glass','walktrap_n','louvain_res'])
order = np.array(['infomap','walktrap','spectral','edge_betweenness','label_propogation','louvain','spin_glass','rich club, thresholded','rich club, dense','walktrap_n','louvain_res'])

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

def small_tstatfunc(x, y,bc=False):
	t, p = scipy.stats.ttest_ind(x,y)
	if bc != False:
		bfc = (p * bc)
		
		if p < 1e-5: pstr = '*!'
		elif p < .05: pstr = '*'
		else: pstr = None

		if bfc <= 0.05:
			if bfc < 1e-5: bfc = '*!'
			else: bfc = '*'
			return "%s \n p%s \n bf%s" %(np.around(t,3),pstr,bfc)

		elif p<0.05: return "t=%s \n p%s" %(np.around(t,3),pstr)
		else: return ""
	else:
		if p < 1e-5: pst = '*!'
		elif p < .05: pst = '*'
		else: pstr = None
		if pstr == None: return "%s" %(np.around(t,3))
		else: return "%s \n p%s" %(np.around(t,3),p)

def really_small_tstatfunc(x, y,bc=False):
	t, p = scipy.stats.ttest_ind(x,y)
	t = int(t)
	if bc != False:
		bfc = (p * bc)
		if bfc < 1e-5: bfc = str(t) + '**' 
		elif bfc < 0.05: bfc = str(t) + '*'
		else: bfc = ''
		return bfc

	else:
		if p < 1e-5: pst = str(t) + '**'
		elif p < .05: pst = str(t) + '* '
		else: pstr = ''
		return pst

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
		if s in dests: continue
		try: vs.append(int(s))
		except: continue
	for s in dests:
		try: vs.append(int(s))
		except: continue
	graph.add_vertices(np.unique(vs).astype(str))
	sources = pd.read_csv('/%s/diverse_club/graphs/routes.dat'%(homedir),header=None)[3].values
	dests = pd.read_csv('/%s/diverse_club/graphs//routes.dat'%(homedir),header=None)[5].values
	for s,d in zip(sources,dests):
		if s == d:continue
		try:
			int(s)
			int(d)
		except: continue
		if int(s) not in vs: continue
		if int(d) not in vs: continue
		s = str(s)
		d = str(d)
		eid = graph.get_eid(s,d,error=False)
		if eid == -1: graph.add_edge(s,d,weight=1)
		else: graph.es[eid]['weight'] = graph.es[eid]["weight"] + 1
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
	assert np.nanmax(np.array(network.pc)) < 1.

def plot_community_stats(network,measure='sizes'):
	network_objects = []
	for community_alg in algorithms:
		network_objects.append(load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,community_alg)))
	if network == 'f_c_elegans':
		iters = ['Worm1','Worm2','Worm3','Worm4']
	if network == 'human':
		iters = tasks
	if network == 'structural_networks':
		iters = ['c elegans','macaque','flight traffic','US power grid']
	
	sns.set(context="paper",font='Helvetica',style='white')
	nconditions = len(iters)

	if nconditions/2 == nconditions/2.:
		fig,subplots = sns.plt.subplots(int(np.ceil((nconditions+1)/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil((nconditions+1)/2.))))
		subplots = subplots.reshape(-1)	
	else:
		fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
		subplots = subplots.reshape(-1)

	for sidx,name_idx in enumerate(iters):
		sizes = np.zeros((9,16))
		q_values = np.zeros((9,16))
		q_values[:] = np.nan
		sizes[:] = np.nan
		for idx,n in enumerate(network_objects):
			niter = 0
			for nidx,nn in enumerate(n.networks):
				if n.names[nidx].split('_')[0] != '%s'%(name_idx) and  n.names[nidx].split('_')[0] != '%s'%(name_idx.lower()):
					continue
				sizes[idx,niter] = int(len(nn.community.sizes()))
				q_values[idx,niter] = nn.community.modularity
				niter += 1
		sizes[-2] = np.flip(sizes[-2],axis=0)
		q_values[-2] = np.flip(q_values[-2],axis=0)
		if measure == 'sizes':
			plot_matrix = sizes
			fmt = ".0f"
		if measure == 'q':
			plot_matrix = q_values
			fmt = ".2f"
		plot_matrix[plot_matrix<0.0] = 0.0
		heatfig = sns.heatmap(plot_matrix,annot=True,xticklabels=[''],yticklabels=np.flip(algorithm_names,0),square=True,rasterized=True,ax=subplots[sidx],cbar=False,fmt=fmt,annot_kws={"size": 5})
		heatfig.set_yticklabels(np.flip(heatfig.axes.get_yticklabels(),0),rotation=360,fontsize=6)
		heatfig.set_xlabel('density, resolution (louvain), or n (walktrap)')
		heatfig.set_title(name_idx.lower(),fontdict={'fontsize':'large'})
	
	if len(subplots) - nconditions == 2:
		for ax in subplots[-2:]:
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.spines['left'].set_color('white')
			ax.spines['bottom'].set_color('white')
			ax.spines['right'].set_color('white')
			ax.spines['top'].set_color('white')
	else:
		ax = subplots[-1]
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.spines['left'].set_color('white')
		ax.spines['bottom'].set_color('white')
		ax.spines['right'].set_color('white')
		ax.spines['top'].set_color('white')
	sns.plt.tight_layout()
	sns.plt.savefig('%s/diverse_club/figures/%s_all_%s.pdf'%(homedir,network,measure))
	sns.plt.close()

def plot_pc_similarity(network):
	def plot_pc_correlations(matrix,labels,nlabels,title,savestr):
		heatmap = sns.heatmap(pd.DataFrame(matrix,columns=labels),square=True,yticklabels=nlabels,xticklabels=nlabels,cmap = "RdBu_r",**{'vmin':-1,'vmax':1.0})
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
		for community_alg in algorithms:
			network_objects.append(load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,community_alg)))
		matrices = []
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
					if task in network_objects[i].names[j] or task.lower() in network_objects[i].names[j]:
						plot_network_objects.append(network_objects[i].networks[j])
						labels.append(algorithms[i] + ',' + str(np.around(subsetiters[label_idx],4)))
						alglabels.append(algorithms[i])
						label_idx += 1
			task_pc_matrix = np.zeros((len(plot_network_objects),len(plot_network_objects)))
			for i in range(len(plot_network_objects)):
				for j in range(len(plot_network_objects)):
					task_pc_matrix[i,j] = scipy.stats.spearmanr(plot_network_objects[i].pc,plot_network_objects[j].pc)[0]
			matrices.append(task_pc_matrix)
			savestr = '/%s/diverse_club/figures/individual/pc_similarity_algorithms_human_%s.pdf'%(homedir,task)
			plot_pc_correlations(task_pc_matrix,labels,4,'human (%s)'%(task.lower()),savestr)
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
			savestr = '/%s/diverse_club/figures/individual/pc_similarity_algorithms_f_c_elegans_%s.pdf'%(homedir,worm)
			plot_pc_correlations(task_pc_matrix,labels,4,'functional c elegans, %s'%(worm.lower()),savestr)
		savestr = '/%s/diverse_club/figures/pc_similarity_algorithms_f_c_elegans_mean.pdf'%(homedir)
		plot_pc_correlations(np.nanmean(matrices,axis=0),labels,4,'functional c elegans, mean across worms',savestr)
	if network == 'structural_networks':
		network_objects = []
		matrices = []
		for community_alg in algorithms:
			network_objects.append(load_object('/%s/diverse_club/graphs/graph_objects/structural_networks_0.8_%s.obj'%(homedir,community_alg)))
		for structural_network in ['macaque','c_elegans','US power grid','flight traffic']:
			if structural_network == 'c_elegans':
				structural_network = 'c elegans'
			labels = []
			alglabels = []
			plot_network_objects = []
			for i in range(len(network_objects)):
				if algorithms[i] == 'louvain_res' or algorithms[i] == 'walktrap_n':
					meanpc = []
					for j in range(len(network_objects[i].names)):
						if structural_network in network_objects[i].names[j]:
							meanpc.append(network_objects[i].networks[j].pc)
					plot_network_objects.append(np.nanmean(meanpc,axis=0))
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
			# np.fill_diagonal(pc_matrix,0.0)
			heatmap = sns.heatmap(pc_matrix,square=True,cmap = "RdBu_r",**{'vmin':-1,'vmax':1.0})
			heatmap.set_xticklabels(algorithms,rotation=90,fontsize=6)
			heatmap.set_yticklabels(np.flip(algorithms,0),rotation=360,fontsize=6)
			mean=np.nanmean(pc_matrix[pc_matrix!=1])
			heatmap.set_title(title + ', mean=%s'%(np.around(mean,3)))
			sns.plt.tight_layout()
			sns.plt.savefig(savestr,dpi=600)
			sns.plt.close()

def plot_degree_distribution(network):
	network_objects = []
	network_objects.append(load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,'infomap')))
	network_objects.append(load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,'walktrap_n')))
	if network == 'f_c_elegans': iters = ['Worm1','Worm2','Worm3','Worm4']
	if network == 'human': iters = tasks
	if network == 'structural_networks': iters = ['c elegans','macaque','flight traffic','US power grid']

	sns.set_style('dark')
	sns.set(font='Helvetica',rc={'axes.facecolor':'.5','axes.grid': False})
	nconditions = len(iters)
	if network == 'f_c_elegans': nconditions = nconditions + 1
	fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
	subplots = subplots.reshape(-1)

	for idx,name in enumerate(iters):
		bw = 'scott'
		if name =='US power grid': bw = 1
		if name == 'flight traffic': bw = 5
		if network != 'structural_networks':
			degrees = []
			for nidx,nn in enumerate(network_objects[0].networks):
				if network_objects[0].names[nidx].split('_')[0] != '%s'%(name) and  network_objects[0].names[nidx].split('_')[0] != '%s'%(name.lower()):
					continue
				degrees.append(nn.community.graph.strength(weights='weight'))

			degrees = np.array(degrees)
			colors = sns.light_palette("red",degrees.shape[0],reverse=True)
			sns.plt.sca(subplots[idx])
			means = []
			for i in range(degrees.shape[0]):
				f = sns.kdeplot(degrees[i],color=colors[i],bw=bw,**{'alpha':.5})
				means.append(f.lines[0].get_data()[1])
			mean = np.nanmean(means,axis=0)
			m = sns.tsplot(mean,color='black',ax=subplots[idx].twiny(),**{'alpha':.5})
			m.set_yticklabels('')
			m.set_xticklabels('')
		for nidx,nn in enumerate(network_objects[1].networks):
			if network_objects[1].names[nidx].split('_')[0] != '%s'%(name) and  network_objects[1].names[nidx].split('_')[0] != '%s'%(name.lower()):
				continue
			degree = nn.community.graph.strength(weights='weight')
		if network != 'structural_networks': 
			d = sns.kdeplot(np.array(degree),ax=subplots[idx].twiny(),color='purple',bw=bw,**{'alpha':.5})
			d.set_yticklabels('')
			d.set_xticklabels('')
		if network == 'structural_networks': 
			d = sns.kdeplot(np.array(degree),ax=subplots[idx],color='red',bw=bw,label='degree',**{'alpha':.5})
			d.legend()
			d.set_yticklabels('')
		d.set_title(name.lower())
	if network != 'structural_networks':
		patches = []
		for color,name in zip(colors,np.arange(5,21)*0.01):
			patches.append(mpl.patches.Patch(color=color,label=name,alpha=.75))
		patches.append(mpl.patches.Patch(color='black',label='mean, thresholded'))
		patches.append(mpl.patches.Patch(color='purple',label='dense'))		
		
		ax = subplots[-1]
		ax.legend(handles=patches,ncol=2,loc=10)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.set_axis_bgcolor("white")
		ax.spines['left'].set_color('white')
		ax.spines['bottom'].set_color('white')
		ax.set_axis_bgcolor('white')
		sns.despine()
		if network != 'human':
			ax = subplots[-2]
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.spines['left'].set_color('white')
			ax.spines['bottom'].set_color('white')
			ax.set_axis_bgcolor("white")
			sns.despine()
			ax.set_axis_bgcolor('white')
	savestr = '/%s/diverse_club/figures/individual/%s_degree.pdf'%(homedir,network)
	sns.plt.tight_layout()
	sns.plt.savefig(savestr)
	sns.plt.show()
	sns.plt.close()

def plot_structural_pc_distribution(network):
	network_objects = []
	for community_alg in algorithms: network_objects.append(load_object('/%s/diverse_club/results/structural_networks_0.8_%s_True_False.obj'%(homedir,community_alg)))
	iters = ['c elegans','macaque','flight traffic','US power grid']
	sns.set_style("white")
	sns.set(font='Helvetica',rc={'axes.facecolor':'.5','axes.grid': False})
	for name_idx,name in enumerate(iters):

		colors = np.zeros((9,3))
		colors[np.ix_([0,1])] = sns.cubehelix_palette(8)[-3],sns.cubehelix_palette(8)[3]
		colors[np.ix_([2,3,4,5,6,7,8])] = np.array(sns.color_palette("cubehelix", 18))[-7:]
		fig,subplots = sns.plt.subplots(2,2,figsize=(mm_2_inches(183),mm_2_inches(61.75*4)))
		subplots = subplots.reshape(-1)
		pcs = []
		for algidx,plotalg in enumerate(algorithms[:-2]):
	
			for nidx,nn in enumerate(network_objects[algidx].networks):
				if network_objects[algidx].names[nidx].split('_')[0] != '%s'%(name) and network_objects[algidx].names[nidx].split('_')[0] != '%s'%(name.lower()):
					continue
				pcs.append(nn.pc)
		pcs = np.array(pcs)
		sns.plt.sca(subplots[0])
		means = []
		for i in range(pcs.shape[0]):
			f = sns.kdeplot(pcs[i],color=colors[i],bw=.05,**{'alpha':.5,'label':algorithm_names[i]})
			means.append(f.lines[0].get_data()[1])
		mean = np.nanmean(means,axis=0)
		newax = subplots[0].twiny()
		m = sns.tsplot(mean,color='black',ax=newax,**{'alpha':.5})
		sns.plt.legend(ncols=2)
		sns.plt.title('%s\nparticipation coefficient\ndistributions'%(name))
		# newax.legend(loc=(.315,.53))
		m.set_yticklabels('')
		m.set_xticklabels('')
		f.set_yticklabels('')
		f.set_xticklabels(['',0,0.2,.4,.6,.8,''])
		
		pcs = []
		n_community_sizes = []
		for nidx,nn in enumerate(network_objects[7].networks):
			if network_objects[7].names[nidx].split('_')[0] != '%s'%(name) and network_objects[7].names[nidx].split('_')[0] != '%s'%(name.lower()):
				continue
			pcs.append(nn.pc)
			n_community_sizes.append(len(nn.community.sizes()))
		n_community_sizes.append(n_community_sizes[-1])
		n_community_sizes = np.flip(n_community_sizes,0)
		
		pcs = np.array(pcs)
		sns.plt.sca(subplots[1])
		means = []
		colors = sns.light_palette("red",pcs.shape[0],reverse=True)
		for i in range(pcs.shape[0]):
			f = sns.kdeplot(pcs[i],color=colors[i],bw=.05,**{'alpha':.5,'label':n_community_sizes[i]})
			means.append(f.lines[0].get_data()[1])
		mean = np.nanmean(means,axis=0)
		newax = subplots[1].twiny()
		m = sns.tsplot(mean,color='black',ax=newax,**{'alpha':.5})
		sns.plt.legend(ncols=2)
		# newax.legend(loc=(.70,.05))
		m.set_yticklabels('')
		m.set_xticklabels('')
		f.set_yticklabels('')
		f.set_xticklabels(['',0,0.2,.4,.6,.8,''])
		sns.plt.title('walktrap (n)')
		

		pcs = []
		n_community_sizes = []
		for nidx,nn in enumerate(network_objects[8].networks):
			if network_objects[8].names[nidx].split('_')[0] != '%s'%(name) and network_objects[8].names[nidx].split('_')[0] != '%s'%(name.lower()):
				continue
			pcs.append(nn.pc)
			n_community_sizes.append(len(nn.community.sizes()))
		
		pcs = np.array(pcs)
		sns.plt.sca(subplots[2])
		means = []
		colors = sns.light_palette("red",pcs.shape[0],reverse=True)
		for i in range(pcs.shape[0]):
			f = sns.kdeplot(pcs[i],color=colors[i],bw=.05,**{'alpha':.5,'label':n_community_sizes[i]})
			means.append(f.lines[0].get_data()[1])
		mean = np.nanmean(means,axis=0)
		newax = subplots[2].twiny()
		m = sns.tsplot(mean,color='black',ax=newax,**{'alpha':.5})
		sns.plt.legend(ncols=2)
		# newax.legend(loc=(.70,.05))
		m.set_yticklabels('')
		m.set_xticklabels('')
		f.set_yticklabels('')
		f.set_xticklabels(['',0,0.2,.4,.6,.8,''])
		sns.plt.title('louvain (resolution)')
	
		savestr = '/%s/diverse_club/figures/individual/%s_dist_%s.pdf'%(homedir,measure,name)
		ax = subplots[-1]
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.set_axis_bgcolor("white")
		ax.spines['left'].set_color('white')
		ax.spines['bottom'].set_color('white')
		ax.spines['top'].set_color('white')
		ax.spines['right'].set_color('white')
		ax.set_axis_bgcolor('white')
		sns.plt.tight_layout()
		sns.plt.savefig(savestr)
		sns.plt.close()

def plot_pc_distribution(network):
	network_objects = []
	bw = 0.05
	for community_alg in algorithms: network_objects.append(load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,community_alg)))
	if network == 'f_c_elegans': iters = ['Worm1','Worm2','Worm3','Worm4']
	if network == 'human': iters = tasks
	algs = algorithms
	sns.set_style('dark')
	sns.set(font='Helvetica',rc={'axes.facecolor':'.5','axes.grid': False})
	for name_idx in iters:
		fig,subplots = sns.plt.subplots(5,2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
		subplots = subplots.reshape(-1)
		plt.text(1.05, 1.13, name_idx.lower() + ' participation coefficient distributions', transform = subplots[0].transAxes, horizontalalignment='center', fontsize=12,)
		for idx,n in enumerate(network_objects):
			pcs = []
			for nidx,nn in enumerate(n.networks):
				if n.names[nidx].split('_')[0] != '%s'%(name_idx) and  n.names[nidx].split('_')[0] != '%s'%(name_idx.lower()):
					continue
				pcs.append(nn.pc)
			pcs = np.array(pcs)
			if algs[idx] == 'walktrap_n':
				pcs = np.flip(pcs,axis=0)
				n_community_sizes = []
				for nidx,nn in enumerate(n.networks):
					if n.names[nidx].split('_')[0] != '%s'%(name_idx) and  n.names[nidx].split('_')[0] != '%s'%(name_idx.lower()):
						continue
					n_community_sizes.append(len(nn.community.sizes()))
			if algs[idx] == 'louvain_res':
				l_community_sizes = []
				for nidx,nn in enumerate(n.networks):
					if n.names[nidx].split('_')[0] != '%s'%(name_idx) and  n.names[nidx].split('_')[0] != '%s'%(name_idx.lower()):
						continue
					l_community_sizes.append(len(nn.community.sizes()))
				l_community_sizes.append(l_community_sizes[-1])
			colors = sns.light_palette("red",pcs.shape[0],reverse=True)
			sns.plt.sca(subplots[idx])
			means = []
			for i in range(pcs.shape[0]):
				f = sns.kdeplot(pcs[i],color=colors[i],bw=bw,**{'alpha':.5})
				means.append(f.lines[0].get_data()[1])
			mean = np.nanmean(means,axis=0)
			m = sns.tsplot(mean,color='black',ax=subplots[idx].twiny(),**{'alpha':.5})
			m.set_yticklabels('')
			m.set_xticklabels('')
			f.set_title(algorithm_names[idx])
			f.set_yticklabels('')
			f.set_xticklabels(['',0,0.2,.4,.6,.8,''])


		patches = []
		colors = sns.light_palette("red",16,reverse=True)
		for color,name,ls,ns, in zip(colors,np.arange(5,21)*0.01,l_community_sizes,np.flip(n_community_sizes,0)):
			name = str(name) + ', ' + str(ls) + ', ' + str(ns)
			patches.append(mpl.patches.Patch(color=color,label=name,alpha=.85))
		patches.append(mpl.patches.Patch(color='black',label='mean'))		
		savestr = '/%s/diverse_club/figures/individual/%s_dist_%s.pdf'%(homedir,measure,'%s_%s'%(network,name_idx))
		ax = subplots[-1]
		ax.legend(handles=patches,ncol=2,loc=10,title='graph density,\n$\it{n}$ communities walktrap (n),\n$\it{n}$ communities louvain (resolution)')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.set_axis_bgcolor("white")
		ax.spines['left'].set_color('white')
		ax.spines['bottom'].set_color('white')
		ax.set_axis_bgcolor('white')
		sns.despine()
		sns.plt.tight_layout()
		sns.plt.savefig(savestr)
		sns.plt.close()

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
	pc_randomized_phis = np.zeros((niters,graph.vcount()))
	for i in range(niters):
		pc_randomized_phis[i] = np.array(RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=pc).phis())
	pc_average_randomized_phis = np.nanmean(pc_randomized_phis,axis=0)
	pc_normalized_phis = pc_emperical_phis/ pc_average_randomized_phis.astype(float)
	pc_normalized_phis_std = pc_normalized_phis / np.std(pc_randomized_phis,axis=0).astype(float)

	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	degree_randomized_phis = np.zeros((niters,graph.vcount()))
	for i in range(niters):
		degree_randomized_phis[i] = np.array(RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=graph.strength(weights='weight')).phis())
	degree_average_randomized_phis = np.nanmean(degree_randomized_phis,axis=0)
	degree_normalized_phis = degree_emperical_phis / degree_average_randomized_phis.astype(float)
	degree_normalized_phis_std = degree_normalized_phis / np.std(degree_randomized_phis,axis=0).astype(float)
	return np.array(pc_normalized_phis),np.array(degree_normalized_phis),np.array(pc_normalized_phis_std),np.array(degree_normalized_phis_std)

def plot_clubness_by_club_value(network,algorithm):
	if network == 'human':
		condition_name = 'task'
		percent_cutoff = .95
	if network == 'structural_networks':
		condition_name = 'network'	
		percent_cutoff = .95
	if network == 'f_c_elegans':
		condition_name = 'worm'
		percent_cutoff = .95
	df = load_object('/%s/diverse_club/results/%s_0.8_%s.obj'%(homedir,network,community_alg)).clubness
	df[condition_name] = df[condition_name].str.lower()
	df['percentile'] = np.nan
	for c in np.unique(df[condition_name]):
		df.percentile[df[condition_name] == c] = df[df[condition_name] == c]['rank'].rank(pct=True)
	df = df[df.permute_strength==False]
	df = df[df.randomize_topology==True]
	nconditions = len(np.unique(df[condition_name]))
	fig,subplots = sns.plt.subplots(nconditions/2,2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil((nconditions)/2.))))
	subplots = subplots.reshape(-1)

	for sidx,diffnetwork in enumerate(np.unique(df[condition_name])):
		temp_df = df[(df[condition_name]==diffnetwork)].copy()
		temp_df = temp_df[temp_df.percentile<percent_cutoff]
		dcorr = str(np.array(scipy.stats.spearmanr(temp_df.clubness[temp_df.club=='diverse'],temp_df.club_value[temp_df.club=='diverse']))[0])[1:4]
		rcorr = str(np.array(scipy.stats.spearmanr(temp_df.clubness[temp_df.club=='rich'],temp_df.club_value[temp_df.club=='rich']))[0])[1:4]
		subplots[sidx].scatter(temp_df.clubness[temp_df.club=='diverse'],temp_df.club_value[temp_df.club=='diverse'],color=sns.color_palette()[0],label='diverse club')
		twin = subplots[sidx].twinx()
		twin.scatter(temp_df.clubness[temp_df.club=='rich'],temp_df.club_value[temp_df.club=='rich'],color=sns.color_palette()[1],label='rich club')
		patches = []
		patches.append(mpl.patches.Patch(color=sns.color_palette()[0],label='diverse club (r=%s)'%(dcorr)))
		patches.append(mpl.patches.Patch(color=sns.color_palette()[1],label='rich club (r=%s)'%(rcorr)))
		subplots[sidx].legend(handles=patches,loc=4)
		subplots[sidx].set_xlabel('clubness')
		subplots[sidx].set_ylabel('participation coefficient')
		twin.set_ylabel('strength')
	sns.plt.tight_layout()
	sns.plt.show()

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
	def calculate_clubness(self,niters,randomize_topology,permute_strength):
		variables = []
		pool = Pool(cores)
		for idx, network in enumerate(self.networks):
			variables.append([network,self.names[idx],self.community_alg,niters,randomize_topology,permute_strength])
		results = pool.map(clubness,variables)
		for idx, result in enumerate(results):
			diverse_clubness,rich_clubness,diverse_clubness_std,rich_clubness_std = result[0],result[1],result[2],result[3]
			temp_df = pd.DataFrame()
			temp_df["rank"] = np.arange((self.vcounts[idx]))
			temp_df['club_value'] = self.networks[idx].pc[np.argsort(self.networks[idx].pc)]
			temp_df['clubness'] = diverse_clubness
			temp_df['club'] = 'diverse'
			temp_df['clubness_std'] = diverse_clubness_std
			temp_df['randomize_topology'] = randomize_topology
			temp_df['permute_strength'] = permute_strength
			for cidx,c in enumerate(self.subsets):
				temp_df[c] = self.names[idx].split('_')[cidx]
			if idx == 0: df = temp_df.copy()
			else: df = df.append(temp_df)
			temp_df = pd.DataFrame()
			temp_df["rank"] = np.arange((self.vcounts[idx]))
			temp_df['club_value'] = np.array(self.networks[idx].community.graph.strength(weights='weight'))[np.argsort(self.networks[idx].community.graph.strength(weights='weight'))]
			temp_df['clubness'] = rich_clubness
			temp_df['club'] = 'rich'
			temp_df['clubness_std'] = rich_clubness_std
			temp_df['randomize_topology'] = randomize_topology
			temp_df['permute_strength'] = permute_strength
			for cidx,c in enumerate(self.subsets):
				temp_df[c] = self.names[idx].split('_')[cidx]
			df = df.append(temp_df)
		df.clubness.loc[df.clubness==np.inf] = np.nan
		if hasattr(self,'clubness'): self.clubness = self.clubness.append(df.copy())
		else: self.clubness = df.copy()
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
	def attack(self,attack_name,n_attacks,n_iters):
		variables = []
		attack_conditions = []
		for n_iter in range(n_iters):
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

def get_whiskers(data):
	whisker_lim = 1.5 * sns.utils.iqr(data)
	q25, q50, q75 = np.percentile(data, [25, 50, 75])
	h1 = np.min(data[data >= (q25 - whisker_lim)])
	h2 = np.max(data[data <= (q75 + whisker_lim)])
	return h1,h2

def plot_all_betweenness(network,measure='betweenness'):
	order = np.array(['infomap','walktrap','spectral','edge_betweenness','label_propogation','louvain','spin_glass','rich club'])
	colors = np.zeros((11,3))
	colors[np.ix_([0,1])] = sns.cubehelix_palette(8)[-3],sns.cubehelix_palette(8)[3]
	colors[np.ix_([2,3,4,5,6,8,9])] = np.array(sns.color_palette("cubehelix", 18))[-7:]
	colors[np.ix_([7,10])] = sns.light_palette("green")[5][:3],sns.light_palette("green")[4][:3]
	colors = colors[:-2]

	if network == 'human':
		condition_name = 'condition'
		percent_cutoff = .95
	if network == 'structural_networks':
		condition_name = 'condition'
		percent_cutoff = .95
	if network == 'f_c_elegans':
		condition_name = 'condition'
		percent_cutoff = .9
	df = pd.DataFrame()
	for community_alg in algorithms[:-2]:
		temp_df = load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,community_alg))
		if measure == 'edge betweenness': 
			temp_df = temp_df.edge_betweenness
			temp_df.rename(columns={'edge_betweenness': 'edge betweenness'},inplace=True)
		elif measure == 'betweenness': temp_df = temp_df.betweenness
		temp_df['community algorithm'] = community_alg
		df = df.append(temp_df)
	sns.set(context="paper",font='Helvetica',style='white')
	df[condition_name] =  df[condition_name].str.lower()
	nconditions = len(np.unique(df[condition_name]))

	if nconditions/2 == nconditions/2.:
		fig,subplots = sns.plt.subplots(int(np.ceil((nconditions+1)/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil((nconditions+1)/2.))))
		subplots = subplots.reshape(-1)	
	else:
		fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
		subplots = subplots.reshape(-1)

	conditions = np.unique(df[condition_name].values)

	for sidx,diffnetwork in enumerate(np.unique(df[condition_name])):
		diverse_df = df[(df[condition_name]==diffnetwork)&(df.club=='diverse')].copy()
		rich_df = df[(df[condition_name]==diffnetwork)&(df.club=='rich')].copy()
		rich_df['community algorithm'] = 'rich club'
		temp_df = rich_df.append(diverse_df)

		ax = sns.boxplot(data=temp_df,width=1,x='community algorithm',y=measure,order=order,palette=colors,showfliers=False,ax=subplots[sidx])
		ax.set_title(diffnetwork + '\n',fontdict={'fontsize':'large'})
		ax.get_xaxis().set_visible(False)
		for i,c in enumerate(order):
			if i <= 6:
				data = (temp_df[measure][(temp_df['community algorithm']==c)])
				h1,h2 = get_whiskers(data)
				stat = really_small_tstatfunc(temp_df[measure][(temp_df['community algorithm']==c)],temp_df[measure][(temp_df['community algorithm']=='rich club')],bc=nconditions)
				ax.text(i,h2+(ax.get_ylim()[1] - ax.get_ylim()[0])/50.,stat,ha='center',color='black',fontsize='small')

	if len(subplots) - nconditions == 2:
		for ax in subplots[-2:]:
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.spines['left'].set_color('white')
			ax.spines['bottom'].set_color('white')
	else:
		ax = subplots[-1]
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.spines['left'].set_color('white')
		ax.spines['bottom'].set_color('white')

	ax = subplots[-1]
	patches = []
	for color,name in zip(colors,order):
		if name == 'walktrap_n': name = 'walktrap (n)'
		if name == 'louvain_res': name = 'louvain (resolution)'
		if name == 'spin_glass': name = 'spin glass'
		if name == 'edge_betweenness': name = 'edge betweenness'
		if name == 'label_propogation': name = 'label propogation'
		patches.append(mpl.patches.Patch(color=color,label=name))
	patches.append(mpl.patches.Patch(label='* : p < 0.05',color='white',edgecolor='white',facecolor='white'))
	patches.append(mpl.patches.Patch(label='** : p < 1e-5',color='white',edgecolor='white',facecolor='white'))
	ax.legend(handles=patches,loc=10)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.spines['left'].set_color('white')
	ax.spines['bottom'].set_color('white')
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('%s/diverse_club/figures/%s_all_%s.pdf'%(homedir,network,measure))
	sns.plt.close()

def plot_all_clubness(network,std=False,randomize_topology=True,permute_strength=False):
	if std == True: value = 'clubness_std'
	else: value = 'clubness'	
	if network == 'human':
		condition_name = 'task'
		percent_cutoff = .95
	if network == 'structural_networks':
		condition_name = 'network'	
		percent_cutoff = .95
	if network == 'f_c_elegans':
		condition_name = 'worm'
		percent_cutoff = .9
	colors = np.zeros((11,3))
	colors[np.ix_([0,1])] = sns.cubehelix_palette(8)[-3],sns.cubehelix_palette(8)[3]
	colors[np.ix_([2,3,4,5,6,7,8])] = np.array(sns.color_palette("cubehelix", 18))[-7:]
	colors[10] = sns.light_palette("green")[4][:3]
	colors[9] = sns.light_palette("green")[5][:3]

	df = pd.DataFrame()
	for community_alg in algorithms:
		temp_df = load_object('/%s/diverse_club/results/%s_0.8_%s.obj'%(homedir,network,community_alg))
		temp_df = temp_df.clubness
		temp_df['community algorithm'] = community_alg
		if community_alg == 'walktrap_n':
			temp_df.rename(columns={'n_clusters':'cost'}, inplace=True)
		if community_alg == 'louvain_res':
			temp_df.rename(columns={'resolution':'cost'}, inplace=True)
		df = df.append(temp_df)
	df['percentile'] = np.nan
	for c in np.unique(df[condition_name]):
		df.percentile[df[condition_name] == c] = df[df[condition_name] == c]['rank'].rank(pct=True)
	df = df[df.permute_strength==permute_strength]
	df = df[df.randomize_topology==randomize_topology]
	sns.set(context="paper",font='Helvetica',style='white')
	df[condition_name] =  df[condition_name].str.lower()
	nconditions = len(np.unique(df[condition_name]))
	if nconditions/2 == nconditions/2.:
		fig,subplots = sns.plt.subplots(int(np.ceil((nconditions+1)/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil((nconditions+1)/2.))))
		subplots = subplots.reshape(-1)	
	else:
		fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
		subplots = subplots.reshape(-1)

	for sidx,diffnetwork in enumerate(np.unique(df[condition_name])):

		temp_df = df[df[condition_name]==diffnetwork].copy()
		temp_df = temp_df[temp_df.percentile<percent_cutoff]

		ymax = np.nanmax(temp_df[value].values)
		ymin = np.nanmin(temp_df[value].values)

		diverse_colors = np.zeros((9,3))
		diverse_colors[np.ix_([0,1])] = sns.cubehelix_palette(8)[-3],sns.cubehelix_palette(8)[3]
		diverse_colors[2:] = np.array(sns.color_palette("cubehelix", 18))[-7:]

		temp_df = df[(df[condition_name]==diffnetwork)&(df.club=='diverse')].copy()
		temp_df = temp_df[(temp_df['community algorithm']!='walktrap_n')&(temp_df['community algorithm']!='louvain_res')]
		ax2 = sns.tsplot(data=temp_df,condition='community algorithm',unit='cost',time='percentile',value=value,color=diverse_colors,ax=subplots[sidx])
		
		# if network != 'structural_networks': 
		# 	rich_df = df[(df[condition_name]==diffnetwork)&(df.club=='rich')].copy()
		# 	rich_df = rich_df[(rich_df['community algorithm']=='walktrap_n')]
		# 	rich_df['community algorithm'] = 'degree, full matrix'
		# 	ax = sns.tsplot(data=rich_df,condition='community algorithm',unit='cost',time='percentile',value=value,color=sns.light_palette("green")[4],ax=subplots[sidx])

		rich_df = df[(df[condition_name]==diffnetwork)&(df.club=='rich')&(df['community algorithm']=='infomap')].copy()
		rich_df['percentile'] = rich_df['rank'].rank(pct=True)
		rich_df['community algorithm'] = 'degree, thresholded'
		ax1 = sns.tsplot(data=rich_df,condition='community algorithm',unit='cost',time='percentile',value=value,color=sns.light_palette("green")[5],ax=subplots[sidx])
		ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(.10))
		
		ax2.set_xlabel('percentile cut off for the club')
		ax2.set_title(diffnetwork.lower(),fontdict={'fontsize':'large'})	
		ax2.set_xlim(0,percent_cutoff)
		final_max = 0
		final_min = 0
		for i in range(len(ax2.lines)):
			temp_max = np.nanmax(ax2.lines[i].get_data()[1][ax2.lines[i].get_data()[0]<percent_cutoff])
			if temp_max > final_max:
				final_max = temp_max
		for i in range(len(ax2.lines)):
			temp_min = np.nanmin(ax2.lines[i].get_data()[1][ax2.lines[i].get_data()[0]<percent_cutoff])
			if temp_min < final_min:
				final_min = temp_min
		ax2.set_ylim(final_min,final_max)
		ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(.15))
		ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
		ax2.legend_.remove()
	# sns.plt.show()

	if len(subplots) - nconditions == 2:
		for ax in subplots[-2:]:
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.spines['left'].set_color('white')
			ax.spines['bottom'].set_color('white')
	else:
		ax = subplots[-1]
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.spines['left'].set_color('white')
		ax.spines['bottom'].set_color('white')
	
	ax = subplots[-1]
	patches = []
	order = np.array(['infomap','walktrap','spectral','edge_betweenness','label_propogation','louvain','spin_glass','walktrap_n','louvain_res','rich club','rich club, dense'])
	for color,name in zip(colors,order):
		if name == 'walktrap_n': name = 'walktrap (n)'
		if name == 'louvain_res': name = 'louvain (resolution)'
		if name == 'spin_glass': name = 'spin glass'
		if name == 'edge_betweenness': name = 'edge betweenness'
		if name == 'label_propogation': name = 'label propogation'
		if name == 'rich club, dense':
			if network == 'structural_networks': continue
		patches.append(mpl.patches.Patch(color=color,label=name))
	ax.legend(handles=patches,loc=10)
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.show()
	sns.plt.savefig('%s/diverse_club/figures/%s_all_clubness.pdf'%(homedir,network))
	# sns.plt.show()
	sns.plt.close()

def plot_all_attacks(network):
	order = np.array(['infomap','walktrap','spectral','edge_betweenness','label_propogation','louvain','spin_glass','rich club'])
	colors = np.zeros((11,3))
	colors[np.ix_([0,1])] = sns.cubehelix_palette(8)[-3],sns.cubehelix_palette(8)[3]
	colors[np.ix_([2,3,4,5,6,8,9])] = np.array(sns.color_palette("cubehelix", 18))[-7:]
	colors[np.ix_([7,10])] = sns.light_palette("green")[5][:3],sns.light_palette("green")[4][:3]
	colors = colors[:-2]
	if network == 'human': condition_name = 'condition'
	if network == 'structural_networks': condition_name = 'condition'	
	if network == 'f_c_elegans': condition_name = 'condition'
	df = pd.DataFrame()
	for community_alg in algorithms[:-2]:
		print community_alg
		temp_df = load_object('/%s/diverse_club/results/%s_0.8_%s.obj'%(homedir,network,community_alg))
		temp_df = temp_df.attacked
		temp_df['community algorithm'] = community_alg
		df = df.append(temp_df)
	# z-score everything
	for condition in np.unique(df[condition_name]):
		df.loc['sum of shortest paths'][(df['attack'] == 'diverse') & (df['condition']==condition)] = df['sum of shortest paths'][(df['attack'] == 'diverse') & (df['condition']==condition)] / np.nanmean(df['sum of shortest paths'][(df['attack'] == 'none') & (df['condition']==condition)])
		df.loc['sum of shortest paths'][(df['attack'] == 'rich') & (df['condition']==condition)] = df['sum of shortest paths'][(df['attack'] == 'rich') & (df['condition']==condition)] / np.nanmean(df['sum of shortest paths'][(df['attack'] == 'none') & (df['condition']==condition)])
	df = df[df.attack != 'none']
	df.rename(columns={'sum of shortest paths': 'sum of shortest paths increase'},inplace=True)
	df[condition_name] =  df[condition_name].str.lower()
	nconditions = len(np.unique(df[condition_name]))
	
	sns.set(context="paper",font='Helvetica',style='white')
	if nconditions/2 == nconditions/2.:
		fig,subplots = sns.plt.subplots(int(np.ceil((nconditions+1)/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil((nconditions+1)/2.))))
		subplots = subplots.reshape(-1)	
	else:
		fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
		subplots = subplots.reshape(-1)

	for sidx,diffnetwork in enumerate(np.unique(df[condition_name])):
		
		diverse_df = df[(df[condition_name]==diffnetwork)&(df.attack=='diverse')].copy()
		rich_df = df[(df[condition_name]==diffnetwork)&(df.attack=='rich')].copy()
		rich_df['community algorithm'] = 'rich club'
		temp_df = rich_df.append(diverse_df)

		ax = sns.boxplot(data=temp_df,width=1,x='community algorithm',y='sum of shortest paths increase',order = order,palette=colors ,showfliers=False,ax=subplots[sidx])
		ax.set_title(diffnetwork + '\n',fontdict={'fontsize':'large'})
		ax.get_xaxis().set_visible(False)

		for i,c in enumerate(order):
			if i <= 6:
				data = temp_df['sum of shortest paths increase'][(temp_df['community algorithm']==c)].values
				h1,h2 = get_whiskers(data)
				stat = really_small_tstatfunc(temp_df['sum of shortest paths increase'][(temp_df['community algorithm']==c)],temp_df['sum of shortest paths increase'][(temp_df['community algorithm']=='rich club')],bc=nconditions)
				ax.text(i,h2+(ax.get_ylim()[1] - ax.get_ylim()[0])/50.,stat,ha='center',color='black',fontsize='x-small')
	if len(subplots) - nconditions == 2:
		for ax in subplots[-2:]:
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.spines['left'].set_color('white')
			ax.spines['bottom'].set_color('white')
	else:
		ax = subplots[-1]
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.spines['left'].set_color('white')
		ax.spines['bottom'].set_color('white')
	

	ax = subplots[-1]
	patches = []
	for color,name in zip(colors,order):
		if name == 'walktrap_n': name = 'walktrap (n)'
		if name == 'louvain_res': name = 'louvain (resolution)'
		if name == 'spin_glass': name = 'spin glass'
		if name == 'edge_betweenness': name = 'edge betweenness'
		if name == 'label_propogation': name = 'label propogation'
		patches.append(mpl.patches.Patch(color=color,label=name))
	patches.append(mpl.patches.Patch(label='* : p < 0.05',color='white',edgecolor='white',facecolor='white'))
	patches.append(mpl.patches.Patch(label='** : p < 1e-5',color='white',edgecolor='white',facecolor='white'))
	ax.legend(handles=patches,loc=10)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	ax.spines['left'].set_color('white')
	ax.spines['bottom'].set_color('white')
	sns.despine()
	sns.plt.tight_layout()
	sns.plt.savefig('%s/diverse_club/figures/all_attacks_%s.pdf'%(homedir,network))
	# sns.plt.show()
	sns.plt.close()	

def plot_all_intersect(network):
	colors = np.zeros((9,3))
	colors[np.ix_([0,1])] = sns.cubehelix_palette(8)[-3],sns.cubehelix_palette(8)[3]
	colors[np.ix_([2,3,4,5,6,7,8])] = np.array(sns.color_palette("cubehelix", 18))[-7:]
	df = pd.DataFrame()
	for community_alg in algorithms:
		temp_df = load_object('/%s/diverse_club/results/%s_0.8_%s_True_False.obj'%(homedir,network,community_alg))
		temp_df = temp_df.intersect
		if community_alg == 'walktrap_n': community_alg = 'walktrap (n)'
		if community_alg == 'louvain_res': community_alg = 'louvain (resolution)'
		if community_alg == 'spin_glass': community_alg = 'spin glass'
		if community_alg == 'edge_betweenness': community_alg = 'edge betweenness'
		if community_alg == 'label_propogation': community_alg = 'label propogation'
		temp_df['community algorithm'] = community_alg
		df = df.append(temp_df)
	df.condition = df.condition.str.lower()
	newintdf = pd.DataFrame()
	for t in df.condition.values:
		newintdf = newintdf.append(pd.DataFrame({'condition':t,'percent community':df['percent community, diverse club'][df.condition==t],'club':'diverse','community algorithm':df['community algorithm'][df.condition==t]}),ignore_index=True)
		newintdf = newintdf.append(pd.DataFrame({'condition':t,'percent community':df['percent community, rich club'][df.condition==t],'club':'rich','community algorithm':df['community algorithm'][df.condition==t]}),ignore_index=True)
	final_df = pd.DataFrame()
	final_df['difference, percentage of communities with a diverse club node minus percentage of communities with a rich club node'] = newintdf['percent community'][newintdf.club == 'diverse'].values - newintdf['percent community'][newintdf.club == 'rich'].values
	final_df['community algorithm'] = newintdf['community algorithm'][newintdf.club == 'diverse']
	final_df['condition'] = newintdf['condition'][newintdf.club == 'diverse']

	fig,subplots = sns.plt.subplots(2,1,figsize=(mm_2_inches(183),mm_2_inches(61.75*4)))
	sns.set(context="paper",font='Helvetica',style='white')
	ax = sns.barplot(y='condition', x="percent overlap", hue='community algorithm', palette=colors,data=df,orient="h",ax=subplots[0])
	ax.set_xlabel('percentage of club nodes in both clubs')
	ax.set_xlim((0,1))
	ax1 = sns.barplot(y='condition', x='difference, percentage of communities with a diverse club node minus percentage of communities with a rich club node', hue='community algorithm', palette=colors,data=final_df,orient="h",ax=subplots[1])
	ax1.legend_.remove()
	ax1.set_xlabel('difference, percentage of communities with a diverse club node\nminus percentage of communities with a rich club node')
	sns.plt.savefig('/%s/diverse_club/figures/%s_all_intersect.pdf'%(homedir,network))

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

def plot_clubness(n,std,savestr):
	if std == True: value = 'clubness_std'
	else: value = 'clubness'
	sns.set(context="paper",font='Helvetica',style='white')
	nconditions = len(np.unique(n.clubness[n.subsets[0]]))
	fig,subplots = sns.plt.subplots(int(np.ceil(nconditions/2.)),2,figsize=(mm_2_inches(183),mm_2_inches(61.75*np.ceil(nconditions/2.))))
	subplots = subplots.reshape(-1)
	for idx,diffnetwork in enumerate(np.unique(n.clubness[n.subsets[0]])):
		df = n.clubness[(n.clubness[n.subsets[0]]==diffnetwork)].copy()
		df = df[df.permute_strength==permute_strength]
		df = df[df.randomize_topology==randomize_topology]
		df['percentile'] = df['rank'].rank(pct=True)
		ax = sns.tsplot(data=df,condition='club',color={'diverse':sns.color_palette()[0],'rich':sns.color_palette()[1]},unit=n.subsets[1],time='percentile',value=value,ax=subplots[idx])
		ax.set_title(diffnetwork.lower(),fontdict={'fontsize':'large'})
		len(ax.lines[1].get_data()[1])
		xcutoff = int(len(ax.lines[1].get_data()[1])*.95)

		ax.set_xlim(0,xcutoff)
		# labels = [item.get_text() for item in ax.get_xticklabels()]
		# 1/0
		# ax.set_xticklabels(np.around(np.linspace(0,100,len(ax.get_xticklabels()))))
		# ax.get_xaxis().get_major_formatter().set_useOffset(False)
		ax.autoscale(True,'both',True)
		ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(.15))
		ax.set_xlabel('percentile cut off for the club')
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
	for t in df.condition.values:
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
		elif community_alg == 'random':
			graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=.1,mst=True)
		else:
			graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=cost,mst=True)
	elif make_graph == False:
		graph = matrix
		if community_alg[:7] == 'louvain':
			nxg = igraph_2_networkx(graph)
	while True:
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
		if community_alg == 'random':
			vc = brain_graphs.brain_graph(VertexClustering(graph,np.random.randint(0,16,graph.vcount())))
		if len(vc.community.sizes()) > 1:
			break
	return vc

def make_networks(network,rankcut,community_alg):
	if network == 'human':
		try:
			1/0
			n = load_object('/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
		except:
			if community_alg == 'louvain_res':
				subsets = ['task','resolution']
				subsetiters = np.linspace(.4,.8,16)
			if community_alg == 'random':
				subsets = ['task','random_iteration']
				subsetiters = np.arange(0,100)
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
		try:
			1/0
			n = load_object('/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
		except:	
			if community_alg == 'louvain_res':
				subsets = ['worm','resolution']
				subsetiters = np.linspace(.4,.8,16)
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
		try:
			1/0
			n = load_object('/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
		except:
			if community_alg == 'louvain_res':
				subsets = ['network','resolution']
				subsetiters = np.linspace(.4,.8,16)
			elif community_alg == 'walktrap_n':
				subsets = ['network','n_clusters']
				subsetiters = np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
			else:
				subsets = ['network','cost']
				subsetiters = [1]
			raw_networks = [get_macaque_graph(),get_structural_c_elegans_graph(),get_power_graph(),get_airport_graph()]
			raw_names = ['macaque','c elegans','US power grid','flight traffic']
			names = []
			networks = []
			pool = Pool(cores)
			variables = []
			for nameidx,n in enumerate(raw_networks):
				for cost in subsetiters:
					variables.append([n,community_alg,cost,False])
					names.append(raw_names[nameidx] + '_' + str(cost))
			networks = pool.map(partition_network,variables)
			n = Network(networks,rankcut,names,subsets,community_alg)
			save_object(n,'/%s/diverse_club/graphs/graph_objects/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))		
	return n

def run_networks(network,run=False,nrandomiters=1000,rankcut=.8,community_alg='infomap',plot_it=False):
	# network='f_c_elegans'
	# run=False
	# nrandomiters=100
	# rankcut=.8
	# community_alg='walktrap_n'
	# randomize_topology=True
	# permute_strength=False
	# plot_it=True

	if run == False: 
		n = load_object('/%s/diverse_club/results/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
	else:
		if network == 'human':
			n = make_networks(network,rankcut,community_alg)
		if network == 'f_c_elegans':
			n = make_networks(network,rankcut,community_alg)
		if network == 'human' or network == 'f_c_elegans':
			if community_alg == 'louvain_res' or community_alg == 'walktrap_n':
				n.attack(None,100,10)
			else:
				n.attack('0.2',1000,10)
		if network == 'structural_networks':
			n = make_networks(network,rankcut,community_alg)
			if community_alg == 'louvain_res' or community_alg == 'walktrap_n':
				n.attack(None,100,10)
			else:
				n.attack(None,1000,10)
		sys.stdout.flush()
		n.calculate_clubness(niters=nrandomiters,randomize_topology=True,permute_strength=False)
		n.calculate_clubness(niters=nrandomiters,randomize_topology=True,permute_strength=True)
		n.calculate_intersect()
		n.calculate_betweenness()
		n.calculate_edge_betweenness()
		save_object(n,'/%s/diverse_club/results/%s_%s_%s.obj'%(homedir,network,rankcut,community_alg))
	if plot_it == True:
		if community_alg == 'louvain_res': filter_name = '0.5'
		elif community_alg == 'walktrap_n': filter_name = '12'
		else:
			if network == 'structural_networks': filter_name = '1'
			else: filter_name = '0.1'
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

def make_graph(variables):
	n_nodes = variables[0]
	np.random.seed(variables[1])
	graph = Graph()
	graph.add_vertices(n_nodes)
	while True:
		i = np.random.randint(0,n_nodes)
		j = np.random.randint(0,n_nodes)
		if i == j:
			continue
		if graph.get_eid(i,j,error=False) == -1:
			graph.add_edge(i,j,weight=1)
		if graph.density() > .05 and graph.is_connected() == True:
			break
	graph.es["weight"] = np.ones(graph.ecount())
	return graph

def generative(variables):
	metric = variables[0]
	n_nodes = variables[1]
	density = variables[2]
	graph = variables[3]
	np.random.seed(variables[4])
	all_shortest = variables[5]
	print variables[4],variables[0]
	q_ratio = variables[6]
	rccs = []
	for idx in range(150):
		delete_edges = graph.get_edgelist()
		if metric != 'none':
			vc = graph.community_fastgreedy().as_clustering()
			orig_q = vc.modularity
			membership = vc.membership
			orig_sps = np.sum(np.array(graph.shortest_paths()))
			community_matrix = brain_graphs.community_matrix(membership,0)
			np.fill_diagonal(community_matrix,1)
			orig_bc_sps = np.sum(np.array(graph.shortest_paths())[community_matrix!=1])
			q_edge_scores = []
			sps_edge_scores = []
			for edge in delete_edges:
				eid = graph.get_eid(edge[0],edge[1],error=False)
				graph.delete_edges(eid)
				q_edge_scores.append(VertexClustering(graph,membership).modularity-orig_q)
				if all_shortest == 'all':
					sps_edge_scores.append(orig_sps-np.sum(np.array(graph.shortest_paths())))
				if all_shortest == 'bc':
					sps_edge_scores.append(orig_bc_sps-np.sum(np.array(graph.shortest_paths())[community_matrix!=1]))
				graph.add_edge(edge[0],edge[1],weight=1)
			q_edge_scores = np.array(q_edge_scores)#Q when edge removed - original Q. High means increase in Q when edge removed.
			sps_edge_scores = np.array(sps_edge_scores)#original sps minus sps when edge removed. Higher value means more efficient.
			assert np.isnan(q_edge_scores).any() == False
			assert np.isnan(sps_edge_scores).any() == False
			if len(np.unique(sps_edge_scores)) > 1:
				q_edge_scores = scipy.stats.zscore(scipy.stats.rankdata(q_edge_scores,method='min'))
				sps_edge_scores = scipy.stats.zscore(scipy.stats.rankdata(sps_edge_scores,method='min'))
				scores = (q_edge_scores*q_ratio) + (sps_edge_scores*(1-q_ratio))
			else:
				scores = scipy.stats.rankdata(q_edge_scores,method='min')
			
		if metric == 'q':
			edges = np.array(delete_edges)[np.argsort(scores)][int(-(graph.ecount()*.05)):]
			edges = np.array(list(edges)[::-1])
		if metric == 'none':
			scores = np.random.randint(0,100,(int(graph.ecount()*.05))).astype(float)
			edges = np.array(delete_edges)[np.argsort(scores)]
		for edge in edges:
			eid = graph.get_eid(edge[0],edge[1],error=False)
			graph.delete_edges(eid)
			if graph.is_connected() == False:
				graph.add_edge(edge[0],edge[1],weight=1)
				continue
			while True:
				i = np.random.randint(0,n_nodes)
				j = np.random.randint(0,n_nodes)
				if i == j:
					continue
				if graph.get_eid(i,j,error=False) == -1:
					graph.add_edge(i,j,weight=1)
					break
		sys.stdout.flush()
		vc = brain_graphs.brain_graph(graph.community_fastgreedy().as_clustering())
		pc = vc.pc
		pc[np.isnan(pc)] = 0.0
		pc_emperical_phis = RC(graph,scores=pc).phis()
		pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=pc).phis() for i in range(25)],axis=0)
		pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
		degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
		average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=True),scores=graph.strength(weights='weight')).phis() for i in range(25)],axis=0)
		degree_normalized_phis = degree_emperical_phis/average_randomized_phis
		rcc = pc_normalized_phis[-10:]
		if np.isfinite(np.nanmean(rcc)):
			rccs.append(np.nanmean(rcc))	
	return [metric,pc_normalized_phis,degree_normalized_phis,graph]

def generative_model(n_nodes=100,iters=100,cores=40,all_shortest='all',q_ratio=.9):
	if n_nodes == 100:
		density= 0.05
	if n_nodes == 200:
		density = 0.025
	if n_nodes == 264:
		density = 0.02
	if n_nodes == 300:
		density = 0.015
	pool = Pool(cores)
	none_deg_rc = []
	none_pc_rc = []
	none_graphs = []
	both_deg_rc = []
	both_pc_rc = []
	both_graphs = []
	variables = []
	for i in range(iters):
		variables.append([n_nodes,i])
	graphs = pool.map(make_graph,variables)
	variables = []

	for i,g in enumerate(graphs):
		variables.append(['none',n_nodes,density,g.copy(),i,all_shortest,q_ratio])
	for i,g in enumerate(graphs):
		variables.append(['q',n_nodes,density,g.copy(),i,all_shortest,q_ratio])
	sys.stdout.flush()
	results = pool.map(generative,variables)
	for r in results:
		if r[0] == 'none':
			none_pc_rc.append(r[1])
			none_deg_rc.append(r[2])
			none_graphs.append(r[3])
		else:
			both_pc_rc.append(r[1])
			both_deg_rc.append(r[2])
			both_graphs.append(r[3])
	with open('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_graphs_none_%s_%s_%s_%s'%(iters,n_nodes,all_shortest,q_ratio),'w+') as f:
		pickle.dump(none_graphs,f)
	with open('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_graphs_both_%s_%s_%s_%s'%(iters,n_nodes,all_shortest,q_ratio),'w+') as f:
		pickle.dump(both_graphs,f)
	np.save('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_pc_none_%s_%s_%s_%s.npy'%(iters,n_nodes,all_shortest,q_ratio),np.array(none_pc_rc))
	np.save('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_deg_none_%s_%s_%s_%s.npy'%(iters,n_nodes,all_shortest,q_ratio),np.array(none_deg_rc))
	np.save('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_pc_both_%s_%s_%s_%s.npy'%(iters,n_nodes,all_shortest,q_ratio),np.array(both_pc_rc))
	np.save('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_deg_both_%s_%s_%s_%s.npy'%(iters,n_nodes,all_shortest,q_ratio),np.array(both_deg_rc))

	with open('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_graphs_none_%s_%s_%s_%s'%(iters,n_nodes,all_shortest,q_ratio),'r') as f:
		none_graphs = pickle.load(f)
	with open('/home/despoB/mb3152/dynamic_mod/results/new_gen_results/rich_club_gen_graphs_both_%s_%s_%s_%s'%(iters,n_nodes,all_shortest,q_ratio),'r') as f:
		both_graphs = pickle.load(f)
	none_mods = []
	for g in none_graphs:
		none_mods.append(g.community_fastgreedy().as_clustering().modularity)
	both_mods = []
	for g in both_graphs:
		both_mods.append(g.community_fastgreedy().as_clustering().modularity)
	print 'Q results, real | random'
	print 'means: ', scipy.stats.ttest_ind(both_mods,none_mods)
	print 't_test:', np.mean(both_mods),np.mean(none_mods)
	none_mods = []
	for g in none_graphs:
		none_mods.append(np.sum(g.shortest_paths()))
	both_mods = []
	for g in both_graphs:
		both_mods.append(np.sum(g.shortest_paths()))
	print 'SP results, real | random'
	print 'means: ', scipy.stats.ttest_ind(both_mods,none_mods)
	print 't_test:', np.mean(both_mods),np.mean(none_mods)

def submit_2_sge(network='human',cores=20):
	for algorithm in algorithms:
		if algorithm == 'walktrap_n' or algorithm == 'louvain_res':
			command = 'qsub -pe threaded %s -V -l mem_free=4G -j y -o /%s/diverse_club/sge/ -e /%s/diverse_club/sge/ -N %s diverse_club.py run %s %s ' \
			%(cores,homedir,homedir,network[0] + '_' + algorithm,network,algorithm)
			os.system(command)
		else:
			command = 'qsub -pe threaded %s -V -l mem_free=1G -j y -o /%s/diverse_club/sge/ -e /%s/diverse_club/sge/ -N %s diverse_club.py run %s %s ' \
			%(cores,homedir,homedir,network[0] + '_' + algorithm,network,algorithm)
			os.system(command)

# networks = ['f_c_elegans','human','structural_networks']
# for network in networks[:1]
	# print network
	# plot_all_intersect(network)
	# plot_all_attacks(network)
	# plot_all_clubness(network)
	# plot_all_betweenness(network,measure='betweenness')
	# plot_all_betweenness(network,measure='edge betweenness')

	# plot_community_stats('f_c_elegans',measure='q')
	# plot_degree_distribution(network)
	# n = make_networks(network,0.8,'infomap')
	# for nn in n.networks:
	# 	check_network(nn)
	# plot_distribution(network,measure='degree')
	# plot_distribution(network,measure='pc')

if len(sys.argv) > 1:
	cores = 40
	if sys.argv[1] == 'run':
		run_networks(sys.argv[2],run=True,nrandomiters=1000,rankcut=.8,community_alg=sys.argv[3])

# random community assingment
