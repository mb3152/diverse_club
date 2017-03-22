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
prop = mpl.font_manager.FontProperties(fname=path)
mpl.rcParams['font.family'] = prop.get_name()

#some globals that we want to have everywhere
global homedir
homedir = '/home/despoB/mb3152/'
global tasks
tasks = ['WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL','REST']
global costs
costs = np.arange(5,21) *0.01


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

def make_static_matrix(subject,task,project,atlas,scrub=False):
	hcp_subject_dir = '/home/despoB/connectome-data/SUBJECT/*TASK*/*reg*'
	parcel_path = '/home/despoB/mb3152/dynamic_mod/atlases/%s_template.nii' %(atlas)
	MP = None
	# try:
	# 	MP = np.load('/home/despoB/mb3152/dynamic_mod/motion_files/%s_%s.npy' %(subject,task))
	# except:
	# 	run_fd(subject,task)
	# 	MP = np.load('/home/despoB/mb3152/dynamic_mod/motion_files/%s_%s.npy' %(subject,task))
	subject_path = hcp_subject_dir.replace('SUBJECT',subject).replace('TASK',task)
	if scrub == True:
		subject_time_series = brain_graphs.load_subject_time_series(subject_path,dis_file=MP,scrub_mm=0.2)
		brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file='/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_%s_matrix_scrubbed_0.2.npy' %(atlas,subject,atlas,task))
	if scrub == False:
		subject_time_series = brain_graphs.load_subject_time_series(subject_path,dis_file=None,scrub_mm=False)
		brain_graphs.time_series_to_matrix(subject_time_series,parcel_path,voxel=False,fisher=False,out_file='/home/despoB/mb3152/dynamic_mod/%s_matrices/%s_%s_%s_matrix.npy' %(atlas,subject,atlas,task))

def between_community_centrality(graph,vc=None):
	if vc == None:
		vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
	rank = int(graph.vcount()/5)
	pc = vc.pc
	pc[np.isnan(pc)] = 0.0
	deg = np.array(vc.community.graph.strength(weights='weight'))
	return [np.array(graph.betweenness())[np.argsort(pc)[-rank:]],
	np.array(graph.betweenness())[np.argsort(deg)[-rank:]]]

def between_community_edge_centrality(graph,vc=None,n_runs=False):
	if vc == None:
		pc = []
		vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
		t_pc = vc.pc
		t_pc[np.isnan(t_pc)] = 0.0
		pc.append(t_pc)
		deg = np.array(vc.community.graph.strength(weights='weight'))
		if n_runs > 1:
			vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			for n in range(n_runs):
				t_pc[np.isnan(t_pc)] = 0.0
				pc.append(t_pc)
		pc = np.nanmean(pc,axis=0)
	pc_matrix = np.zeros((graph.vcount(),graph.vcount()))
	deg_matrix = np.zeros((graph.vcount(),graph.vcount()))
	between = graph.edge_betweenness()
	edge_matrix =  np.zeros((graph.vcount(),graph.vcount()))
	pc_thresh = np.percentile(pc,80)
	deg_thresh = np.percentile(deg,80)
	for idx,edge in enumerate(graph.get_edgelist()):
		edge_matrix[edge[0],edge[1]] = between[idx]
		edge_matrix[edge[1],edge[0]] = between[idx]
		if pc[edge[0]] > pc_thresh:
			if pc[edge[1]] > pc_thresh:
				pc_matrix[edge[0],edge[1]] = 1
				pc_matrix[edge[1],edge[0]] = 1
		if deg[edge[0]] > deg_thresh:
			if deg[edge[1]] > deg_thresh:
				deg_matrix[edge[0],edge[1]] = 1
				deg_matrix[edge[1],edge[0]] = 1
	return  edge_matrix[pc_matrix>0], edge_matrix[deg_matrix>0]

def attack(variables):
	graph = variables[1]
	np.random.seed(variables[0])
	try: vc = variables[2]
	except: vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
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
		num_kill = np.random.choice(range(dmin,dmax),1)
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
		# delete_edges = []
		# for node1,node2 in d_edges.reshape(d_edges.shape[0],2):
		# 	delete_edges.append(graph.get_eid(node1,node2))
		# d_graph.delete_edges(delete_edges)
		# if d_graph.is_connected() == False:
		# 	continue
		# delete_edges = []
		# for node1,node2 in c_edges.reshape(c_edges.shape[0],2):
		# 	delete_edges.append(graph.get_eid(node1,node2))
		# c_graph.delete_edges(delete_edges)
		# if c_graph.is_connected() == False:
		# 	continue
		deg_sp = np.array(d_graph.shortest_paths()).astype(float)
		c_sp = np.array(c_graph.shortest_paths()).astype(float)
		attack_degree_sps.append(np.nansum(deg_sp))
		attack_pc_sps.append(np.nansum(c_sp))
		# attack_degree_mods.append(d_graph.community_infomap(edge_weights='weight').modularity)
		# attack_pc_mods.append(c_graph.community_infomap(edge_weights='weight').modularity)
		idx = idx + 1
		if idx == 500:
			break
	# return [attack_pc_sps,attack_degree_sps,healthy_sp,attack_pc_mods,attack_degree_mods,np.mean(healthy_mods)]
	return [attack_pc_sps,attack_degree_sps,healthy_sp]

def human_attacks():
	tasks = ['REST','WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']
	try:
		df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/human_attack')
		btw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/human_bwt')
		betw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/human_ebwt')
	except:
		pool = Pool(20)
		tasks = ['REST','WM','GAMBLING','RELATIONAL','MOTOR','LANGUAGE','SOCIAL']
		df = pd.DataFrame(columns=['Attack Type','Sum of Shortest Paths'])
		btw_df = pd.DataFrame(columns=['Betweenness','Node Type','Task'])
		betw_df = pd.DataFrame(columns=['Edge Betweenness','Node Type','Task'])
		for task in tasks:
			atlas = 'power'
			print task
			subjects = np.array(hcp_subjects).copy()
			subjects = list(subjects)
			subjects = remove_missing_subjects(subjects,task,atlas)
			static_results = graph_metrics(subjects,task,atlas,'fz')
			subject_pcs = static_results['subject_pcs']
			matrices = static_results['matrices']


			variables = []
			for i in np.arange(5,21):
				cost = (i)*0.01
				temp_matrix = np.nanmean(static_results['matrices'],axis=0)
				graph = brain_graphs.matrix_to_igraph(temp_matrix.copy(),cost=cost,mst=True)
				variables.append(graph)


			b_results = pool.map(between_community_edge_centrality,variables)
			pc_btw = []
			deg_btw = []
			for r in b_results:
				pc_btw = np.append(r[0],pc_btw)
				deg_btw = np.append(r[1],deg_btw)
			print 'Edge Betweenness, PC Versus Degree', scipy.stats.ttest_ind(pc_btw,deg_btw)	

			tbtw_df = pd.DataFrame()
			tbtw_df['Edge Betweenness'] = pc_btw
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(pc_btw)).astype(str)
			task_str[:] = task
			tbtw_df['Task'] = task_str
			betw_df = betw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(deg_btw)).astype(str)
			task_str[:] = task
			tbtw_df['Task'] = task_str
			tbtw_df['Edge Betweenness'] = deg_btw
			tbtw_df['Node Type'] = 'Degree'
			betw_df = betw_df.append(tbtw_df)

			b_results = pool.map(between_community_centrality,variables)
			pc_btw = []
			deg_btw = []
			for r in b_results:
				pc_btw = np.append(r[0],pc_btw)
				deg_btw = np.append(r[1],deg_btw)	
			print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(pc_btw,deg_btw)	

			tbtw_df = pd.DataFrame()
			tbtw_df['Betweenness'] = pc_btw
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(pc_btw)).astype(str)
			task_str[:] = task
			tbtw_df['Task'] = task_str
			btw_df = btw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(deg_btw)).astype(str)
			task_str[:] = task
			tbtw_df['Task'] = task_str
			tbtw_df['Betweenness'] = deg_btw
			tbtw_df['Node Type'] = 'Degree'
			btw_df = btw_df.append(tbtw_df)

			variables = []
			temp_matrix = np.nanmean(static_results['matrices'],axis=0).copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=.2,mst=True)
			for i in np.arange(20):
				variables.append([i,graph.copy(),brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))])

			pool = Pool(20)
			results = pool.map(attack,variables)
			attack_degree_sps = np.array([])
			attack_pc_sps = np.array([])
			healthy_sp = np.array([])
			for r in results:
				attack_pc_sps = np.append(r[0],attack_pc_sps)
				attack_degree_sps = np.append(r[1],attack_degree_sps)
				healthy_sp = np.append(r[2],healthy_sp)
			attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
			attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
			print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)


			sys.stdout.flush()
			hdf = pd.DataFrame()
			task_str = np.ones(len(healthy_sp)).astype(str)
			task_str[:] = task
			hdf['Task'] = task_str
			hdf['Sum of Shortest Paths'] = healthy_sp
			hdf['Attack Type'] = 'None'
			df = df.append(hdf)
			d_df = pd.DataFrame()
			task_str = np.ones(len(attack_degree_sps)).astype(str)
			task_str[:] = task
			d_df['Task'] = task_str
			d_df['Sum of Shortest Paths'] = attack_degree_sps
			d_df['Attack Type'] = 'Degree Rich Club'
			df = df.append(d_df)
			pc_df = pd.DataFrame()
			task_str = np.ones(len(attack_pc_sps)).astype(str)
			task_str[:] = task
			pc_df['Task'] = task_str
			pc_df['Sum of Shortest Paths'] = attack_pc_sps
			pc_df['Attack Type'] = 'PC Rich Club'
			df = df.append(pc_df)

		df.to_csv('/home/despoB/mb3152/dynamic_mod/results/human_attack')
		btw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/human_bwt')
		betw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/human_btw_e')
	
	for task in tasks:
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Task']==task)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Task']==task)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Task']==task)])
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Task']==task)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Task']==task)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Task']==task)])
	colors= sns.color_palette(['#3F6075', '#FFC61E'])
	sns.set(font='Helvetica')
	sns.set(style="whitegrid")
	sns.boxplot(data=df[df['Attack Type']!='None'],x='Task',hue='Attack Type',y="Sum of Shortest Paths",palette=colors)
	for i,task in enumerate(tasks):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Task']==task)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Task']==task)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['Attack Type'] != 'None') & (df['Task']==task)])) + (np.nanstd(np.array(df['Sum of Shortest Paths'][(df['Task']==task)&(df['Attack Type'] != 'None')]))/2)
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.tight_layout()
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_human.pdf',dpi=3600)
	sns.plt.close()
	sns.set(style="whitegrid")
	sns.set(font='Helvetica')
	sns.set(style="whitegrid")
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.boxplot(data=btw_df,x='Task',hue='Node Type',y="Betweenness",palette=colors,showfliers=False)
	for i,task in enumerate(tasks):
		stat = tstatfunc(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['Task']==task)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['Task']==task)],15)
		maxvaly = np.mean(btw_df['Betweenness'])
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/humanbtw.pdf',dpi=3600)
	sns.plt.show()
	sns.plt.close()
	scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'PC Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'])

	sns.set(style="whitegrid")
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.boxplot(data=betw_df,x='Task',hue='Node Type',y="Edge Betweenness",palette=colors,showfliers=False)
	# sns.violinplot(data=betw_df,x='Task',hue='Node Type',y="Edge Betweenness",palette=colors)
	for i,task in enumerate(tasks):
		stat = tstatfunc(betw_df['Edge Betweenness'][(betw_df['Node Type'] == 'PC') & (betw_df['Task']==task)],betw_df['Edge Betweenness'][(betw_df['Node Type'] == 'Degree') & (betw_df['Task']==task)],15)
		# maxvaly = np.nanmax(np.array(betw_df['Edge Betweenness'][betw_df['Task']==task])) + (np.nanstd(np.array(betw_df['Edge Betweenness'][betw_df['Task']==task]))/2)
		maxvaly = np.mean(betw_df['Edge Betweenness'])
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/humanbetw_e.pdf',dpi=3600)
	sns.plt.show()

def structural_attacks(networks=['macaque','c_elegans','power_grid','air_traffic']):

	try:
		1/0
		df =pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/stuc_attack')
		df = df[(df.network!='c_elegans')&(df.network!='air_traffic')]
		btw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_bwt')
		btw_df = btw_df[(btw_df.network!='c_elegans')&(btw_df.network!='air_traffic')]
		intdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_int')
		intdf = intdf[(intdf.network!='c_elegans')&(intdf.network!='air_traffic')]
		ebtw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_ebwt')
		ebtw_df = ebtw_df[(ebtw_df.network!='c_elegans')&(ebtw_df.network!='air_traffic')]
		
		cdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/stuc_attack_new')
		cbtw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_bwt_new')
		cintdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_int_new')
		cebtw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_ebwt_new')

		adf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/stuc_attack_new_a')
		abtw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_bwt_new_a')
		aintdf = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_int_new_a')
		aebtw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/struc_ebwt_new_a')

		# df[df['network'] == 'c_elegans'] = cdf[cdf['network'] == 'c_elegans']
		# btw_df[btw_df['network'] == 'c_elegans'] = cbtw_df[cbtw_df['network'] == 'c_elegans']
		# ebtw_df[ebtw_df['network'] == 'c_elegans'] = cebtw_df[cebtw_df['network'] == 'c_elegans']
		# intdf[intdf['network'] == 'c_elegans'] = cintdf[cintdf['network'] == 'c_elegans']

		df = df.append(cdf)
		btw_df = btw_df.append(cbtw_df)
		ebtw_df = ebtw_df.append(cebtw_df)
		intdf= intdf.append(cintdf)

		df = df.append(adf)
		btw_df = btw_df.append(abtw_df)
		ebtw_df = ebtw_df.append(aebtw_df)
		intdf= intdf.append(aintdf)
	except:
		networks = ['macaque','c_elegans','air_traffic','power_grid']
		# networks = ['air_traffic']
		intdf = pd.DataFrame()
		df = pd.DataFrame(columns=['Attack Type','Sum of Shortest Paths','network'])
		btw_df = pd.DataFrame(columns=['Betweenness','Node Type','network'])
		ebtw_df = pd.DataFrame(columns=['Edge Betweenness','Node Type','network'])
		for network in networks[:2]:
			if network == 'macaque':
				matrix = loadmat('/home/despoB/mb3152/dynamic_mod/%s.mat'%(network))['CIJ']
				temp_matrix = matrix.copy()
				graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=1.)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			if network == 'c_elegans':
				graph = Graph.Read_GML('/home/despoB/mb3152/dynamic_mod/celegansneural.gml')
				graph.es["weight"] = np.ones(graph.ecount())
				graph.to_undirected()
				graph.es["weight"] = np.ones(graph.ecount())
				matrix = np.array(graph.get_adjacency(attribute='weight').data)
				graph = brain_graphs.matrix_to_igraph(matrix,cost=1.)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			if network == 'power_grid':
				graph = power_rich_club(return_graph=True)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
			if network == 'air_traffic':
				graph = airlines_RC(return_graph=True)
				v_to_d = []
				for i in range(3281):
					if len(graph.subcomponent(i)) < 3000:
						v_to_d.append(i)
				graph.delete_vertices(v_to_d)
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))

			b_results = between_community_edge_centrality(graph.copy(),None,5)
			print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(b_results[0],b_results[1])			

			tbtw_df = pd.DataFrame()
			tbtw_df['Edge Betweenness'] = b_results[0]
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(b_results[0])).astype(str)
			task_str[:] = network
			tbtw_df['network'] = task_str
			ebtw_df = ebtw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(b_results[1])).astype(str)
			task_str[:] = network
			tbtw_df['network'] = task_str
			tbtw_df['Edge Betweenness'] = b_results[1]
			tbtw_df['Node Type'] = 'Degree'
			ebtw_df = ebtw_df.append(tbtw_df)


			b_results = between_community_centrality(graph.copy(),vc)
			print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(b_results[0],b_results[1])				

			tbtw_df = pd.DataFrame()
			tbtw_df['Betweenness'] = b_results[0]
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(b_results[0])).astype(str)
			task_str[:] = network
			tbtw_df['network'] = task_str
			btw_df = btw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(b_results[1])).astype(str)
			task_str[:] = network
			tbtw_df['network'] = task_str
			tbtw_df['Betweenness'] = b_results[1]
			tbtw_df['Node Type'] = 'Degree'
			btw_df = btw_df.append(tbtw_df)

			print 'Rich Club Intersection'
			sys.stdout.flush()
			inters = rich_club_intersect(graph.copy(),(graph.vcount())-(graph.vcount()/5),vc)
			1/0
			temp_df = pd.DataFrame(columns=["Percent Overlap", 'Percent Community, PC','Percent Community, Degree','Task'],index=np.arange(1))
			temp_df["Percent Overlap"] = inters[0]
			temp_df['Percent Community, PC'] = inters[1]
			temp_df['Percent Community, Degree'] = inters[2]
			temp_df['network'] = network
			intdf = intdf.append(temp_df)
			intdf.to_csv('/home/despoB/mb3152/dynamic_mod/results/struc_int_new_a')	
			variables = []
			for i in np.arange(20):
				if network == 'power_grid':
					variables.append([i,graph.copy(),vc])
					continue
				if network == 'air_traffic':
					variables.append([i,graph.copy(),vc])
					continue
				else:
					variables.append([i,graph.copy(),brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))])
			pool = Pool(20)
			print 'Rich Club Attacks'
			sys.stdout.flush()
			results = pool.map(attack,variables)
			attack_degree_sps = np.array([])
			attack_pc_sps = np.array([])
			healthy_sp = np.array([])
			# attack_degree_mods = np.array([])
			# attack_pc_mods = np.array([])
			# healthy_mods = np.array([])
			for r in results:
				attack_pc_sps = np.append(r[0],attack_pc_sps)
				attack_degree_sps = np.append(r[1],attack_degree_sps)
				healthy_sp = np.append(r[2],healthy_sp)
				# attack_pc_mods = np.append(r[3],attack_pc_mods)
				# attack_degree_mods = np.append(r[4],attack_degree_mods)
				# healthy_mods = np.append(r[5],healthy_mods)
			attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
			attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
			print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)
			# print scipy.stats.ttest_ind(attack_degree_mods,attack_pc_mods)

			sys.stdout.flush()
			sys.stdout.flush()
			hdf = pd.DataFrame()
			task_str = np.ones(len(healthy_sp)).astype(str)
			task_str[:] = network
			hdf['network'] = task_str
			hdf['Sum of Shortest Paths'] = healthy_sp
			hdf['Attack Type'] = 'None'
			df = df.append(hdf)
			d_df = pd.DataFrame()
			task_str = np.ones(len(attack_degree_sps)).astype(str)
			task_str[:] = network
			d_df['network'] = task_str
			d_df['Sum of Shortest Paths'] = attack_degree_sps
			d_df['Attack Type'] = 'Degree Rich Club'
			df = df.append(d_df)
			pc_df = pd.DataFrame()
			task_str = np.ones(len(attack_pc_sps)).astype(str)
			task_str[:] = network
			pc_df['network'] = task_str
			pc_df['Sum of Shortest Paths'] = attack_pc_sps
			pc_df['Attack Type'] = 'PC Rich Club'
			df = df.append(pc_df)

		df.to_csv('/home/despoB/mb3152/dynamic_mod/results/stuc_attack_new_a')
		btw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/struc_bwt_new_a')
		
		ebtw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/struc_ebwt_new_a')
	1/0
	df['Sum of Shortest Paths'] = df['Sum of Shortest Paths'].astype(float)
	for network in networks:
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['network']==network)])
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['network']==network)])
	# sns.set(style="whitegrid",font_scale=1)
	sns.set_style("white")
	sns.set_style("ticks")
	intdf["Percent Overlap"] = intdf["Percent Overlap"].astype(float)
	intdf['Percent Community, PC'] = intdf['Percent Community, PC'].astype(float)
	intdf['Percent Community, Degree'] = intdf['Percent Community, Degree'].astype(float)
	sns.barplot(data=intdf,x='Percent Overlap',y='network')
	sns.plt.xlim((0,1))
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_overlap_struc_x01.pdf')
	sns.plt.show()
	newintdf = pd.DataFrame()
	for network in networks:
		newintdf = newintdf.append(pd.DataFrame({'Network':network,'PercentCommunity':intdf['Percent Community, PC'][intdf.network==network],'Club':'Diverse'}),ignore_index=True)
		newintdf = newintdf.append(pd.DataFrame({'Network':network,'PercentCommunity':intdf['Percent Community, Degree'][intdf.network==network],'Club':'Rich'}),ignore_index=True)
	sns.barplot(data=intdf,x='Percent Community, PC',y='network')
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_pc_struc.pdf')
	sns.plt.close()
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.barplot(data=newintdf,x='PercentCommunity',y='Network',hue='Club',palette=colors)
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_both_struc.pdf')

	sns.plt.show()
	# sns.barplot(data=intdf,x='Percent Community, Degree',y='network')
	# sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
	# sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/percent_community_degree_struct.pdf')
	# sns.plt.show()
	colors= sns.color_palette(['#3F6075','#FFC61E'])
	sns.set(style="whitegrid",font='Helvetica')
	sns.boxplot(order=networks[:2],data=df[(df['Attack Type']!='None') & (df.network!='power_grid') & (df.network!='air_traffic') ],palette=colors,x='network',hue='Attack Type',y="Sum of Shortest Paths")
	for i,network in enumerate(networks[:2]):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['network']==network) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['network']==network)&(df['Attack Type'] != 'None')]))/2
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_struc1.pdf',dpi=3600)
	sns.plt.show()
	sns.boxplot(data=df[(df['Attack Type']!='None') & (df.network!='c_elegans') & (df.network!='macaque') & (df.network!='power_grid')],palette=colors,x='network',hue='Attack Type',y="Sum of Shortest Paths")
	i = 0
	network = 'air_traffic'
	stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)])
	maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['network']==network) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['network']==network)&(df['Attack Type'] != 'None')]))/2
	sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_struc2.pdf',dpi=3600)
	sns.plt.show()	
	colors= sns.color_palette(['#3F6075','#FFC61E'])
	sns.boxplot(data=df[(df['Attack Type']!='None') & (df.network!='c_elegans') & (df.network!='macaque') & (df.network!='air_traffic')],palette=colors,x='network',hue='Attack Type',y="Sum of Shortest Paths")
	i = 0
	network = 'power_grid'
	stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network']==network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network']==network)])
	maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['network']==network) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['network']==network)&(df['Attack Type'] != 'None')]))/2
	sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_struc3.pdf',dpi=3600)
	sns.plt.show()	
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.set(style="whitegrid",font='Helvetica')
	for network in networks:
		btw_df['Betweenness'][btw_df['network']==network] = (btw_df['Betweenness'][btw_df['network']==network] - np.nanmean(btw_df['Betweenness'][btw_df['network']==network])) / np.nanstd(btw_df['Betweenness'][btw_df['network']==network])
	sns.boxplot(order=networks,data=btw_df,palette=colors,x='network',hue='Node Type',y="Betweenness",showfliers=False)
	sns.plt.ylim(-3,3)
	for i,network in enumerate(networks):
		stat = tstatfunc(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['network']==network)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['network']==network)],bc=15)
		maxvaly = np.nanmax(np.array(btw_df['Betweenness'][btw_df['network']==network])) + (np.nanstd(np.array(btw_df['Betweenness'][btw_df['network']==network]))/2)
		if maxvaly > 3:
			maxvaly = 3
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/strucbtw.pdf',dpi=3600)
	sns.plt.show()
	for network in networks:
		print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['network'] == network)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['network'] == network)])	
	for network in networks:
		print 'Damage, PC Versus Degree', scipy.stats.ttest_ind(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['network'] == network)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['network'] == network)])	
	print scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'PC Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'])

	sns.set(style="whitegrid")
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	for network in networks:
		ebtw_df['Edge Betweenness'][ebtw_df['network']==network] = (ebtw_df['Edge Betweenness'][ebtw_df['network']==network] - np.nanmean(ebtw_df['Edge Betweenness'][ebtw_df['network']==network])) / np.nanstd(ebtw_df['Edge Betweenness'][ebtw_df['network']==network])
	sns.boxplot(order=networks,data=ebtw_df,x='network',hue='Node Type',y="Edge Betweenness",palette=colors,showfliers=False)
	sns.plt.ylim(-3,3)
	for i,network in enumerate(networks):
		stat = tstatfunc(ebtw_df['Edge Betweenness'][(ebtw_df['Node Type'] == 'PC') & (ebtw_df['network']==network)],ebtw_df['Edge Betweenness'][(ebtw_df['Node Type'] == 'Degree') & (ebtw_df['network']==network)],bc=15)
		maxvaly = np.nanmax(np.array(ebtw_df['Edge Betweenness'][ebtw_df['network']==network])) + (np.nanstd(np.array(ebtw_df['Edge Betweenness'][ebtw_df['network']==network]))/2)
		if maxvaly > 3:
			maxvaly = 3
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/struc_ebtw_e.pdf',dpi=3600)
	sns.plt.show()

def fce_attacks():
	worms = ['Worm1','Worm2','Worm3','Worm4']
	try:
		df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/fce_attack')
		btw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/fce_bwt')
		ebtw_df = pd.read_csv('/home/despoB/mb3152/dynamic_mod/results/fce_ebwt')
	except:
		worms = ['Worm1','Worm2','Worm3','Worm4']
		df = pd.DataFrame(columns=['Attack Type','Sum of Shortest Paths'])
		btw_df = pd.DataFrame(columns=['Betweenness','Node Type','Worm'])
		ebtw_df = pd.DataFrame(columns=['Edge Betweenness','Node Type','Worm'])
		for worm in worms:
			matrix = np.array(pd.read_excel('pnas.1507110112.sd01.xls',sheetname=worm).corr())[4:,4:]
			variables = []
			for i in np.arange(5,21):
				cost = i * 0.01
				temp_matrix = matrix.copy()
				graph = brain_graphs.matrix_to_igraph(temp_matrix,cost=cost,mst=True,binary=False)
				variables.append(graph)	
			b_results = pool.map(between_community_edge_centrality,variables)
			pc_btw = []
			deg_btw = []
			for r in b_results:
				pc_btw = np.append(r[0],pc_btw)
				deg_btw = np.append(r[1],deg_btw)	
			print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(pc_btw,deg_btw)				

			tbtw_df = pd.DataFrame()
			tbtw_df['Edge Betweenness'] = pc_btw
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(pc_btw)).astype(str)
			task_str[:] = worm
			tbtw_df['Worm'] = task_str
			ebtw_df = ebtw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(deg_btw)).astype(str)
			task_str[:] = worm
			tbtw_df['Worm'] = task_str
			tbtw_df['Edge Betweenness'] = deg_btw
			tbtw_df['Node Type'] = 'Degree'
			ebtw_df = ebtw_df.append(tbtw_df)


			pool = Pool(20)
			b_results = pool.map(between_community_centrality,variables)
			pc_btw = []
			deg_btw = []
			for r in b_results:
				pc_btw = np.append(r[0],pc_btw)
				deg_btw = np.append(r[1],deg_btw)	
			print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(pc_btw,deg_btw)	
			tbtw_df = pd.DataFrame()
			tbtw_df['Betweenness'] = pc_btw
			tbtw_df['Node Type'] = 'PC'
			task_str = np.ones(len(pc_btw)).astype(str)
			task_str[:] = worm
			tbtw_df['Worm'] = task_str
			btw_df = btw_df.append(tbtw_df)
			tbtw_df = pd.DataFrame()
			task_str = np.ones(len(deg_btw)).astype(str)
			task_str[:] = worm
			tbtw_df['Worm'] = task_str
			tbtw_df['Betweenness'] = deg_btw
			tbtw_df['Node Type'] = 'Degree'
			btw_df = btw_df.append(tbtw_df)

			variables = []
			temp_matrix = matrix.copy()
			graph = brain_graphs.matrix_to_igraph(temp_matrix.copy(),cost=.2,mst=True)
			for i in np.arange(20):
				vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
				variables.append([i,graph.copy(),vc])

			pool = Pool(20)
			results = pool.map(attack,variables)
			attack_degree_sps = np.array([])
			attack_pc_sps = np.array([])
			healthy_sp = np.array([])
			for r in results:
				attack_pc_sps = np.append(r[0],attack_pc_sps)
				attack_degree_sps = np.append(r[1],attack_degree_sps)
				healthy_sp = np.append(r[2],healthy_sp)
			attack_degree_sps = np.array(attack_degree_sps).reshape(-1)
			attack_pc_sps = np.array(attack_pc_sps).reshape(-1)
			print scipy.stats.ttest_ind(attack_pc_sps,attack_degree_sps)

			sys.stdout.flush()
			hdf = pd.DataFrame()
			task_str = np.ones(len(healthy_sp)).astype(str)
			task_str[:] = worm
			hdf['Worm'] = task_str
			hdf['Sum of Shortest Paths'] = healthy_sp
			hdf['Attack Type'] = 'None'
			df = df.append(hdf)
			d_df = pd.DataFrame()
			task_str = np.ones(len(attack_degree_sps)).astype(str)
			task_str[:] = worm
			d_df['Worm'] = task_str
			d_df['Sum of Shortest Paths'] = attack_degree_sps
			d_df['Attack Type'] = 'Degree Rich Club'
			df = df.append(d_df)
			pc_df = pd.DataFrame()
			task_str = np.ones(len(attack_pc_sps)).astype(str)
			task_str[:] = worm
			pc_df['Worm'] = task_str
			pc_df['Sum of Shortest Paths'] = attack_pc_sps
			pc_df['Attack Type'] = 'PC Rich Club'
			df = df.append(pc_df)

		df.to_csv('/home/despoB/mb3152/dynamic_mod/results/fce_attack')
		btw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/fce_bwt')
		ebtw_df.to_csv('/home/despoB/mb3152/dynamic_mod/results/fce_ebwt')
	for Worm in worms:
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm']==Worm)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm']==Worm)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Worm']==Worm)])
		df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm']==Worm)] = df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm']==Worm)] / np.nanmean(df['Sum of Shortest Paths'][(df['Attack Type'] == 'None') & (df['Worm']==Worm)])
	colors= sns.color_palette(['#3F6075', '#FFC61E'])
	sns.set(style="whitegrid",font='Helvetica')
	sns.boxplot(data=df[df['Attack Type']!='None'],x='Worm',hue='Attack Type',y="Sum of Shortest Paths",palette=colors)
	for i,Worm in enumerate(worms):
		stat = tstatfunc(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm']==Worm)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm']==Worm)])
		maxvaly = np.nanmax(np.array(df['Sum of Shortest Paths'][(df['Worm']==Worm) & (df['Attack Type'] != 'None')])) + np.nanstd(np.array(df['Sum of Shortest Paths'][(df['Worm']==Worm)&(df['Attack Type'] != 'None')]))/2
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/attack_fce.pdf',dpi=3600)
	sns.plt.close()
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.set(style="whitegrid",font='Helvetica')
	sns.boxplot(data=btw_df,x='Worm',hue='Node Type',y="Betweenness",palette=colors,showfliers=False)
	for i,Worm in enumerate(worms):
		stat = tstatfunc(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['Worm']==Worm)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['Worm']==Worm)],15)
		maxvaly = np.mean(btw_df['Betweenness'])
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/fcebtw.pdf',dpi=3600)
	sns.plt.show()
	for worm in worms:
		print 'Betweenness, PC Versus Degree', scipy.stats.ttest_ind(btw_df['Betweenness'][(btw_df['Node Type'] == 'PC') & (btw_df['Worm'] == worm)],btw_df['Betweenness'][(btw_df['Node Type'] == 'Degree') & (btw_df['Worm'] == worm)])	
	for worm in worms:
		print 'Damage, PC Versus Degree', scipy.stats.ttest_ind(df['Sum of Shortest Paths'][(df['Attack Type'] == 'PC Rich Club') & (df['Worm'] == worm)],df['Sum of Shortest Paths'][(df['Attack Type'] == 'Degree Rich Club') & (df['Worm'] == worm)])	
	print scipy.stats.ttest_ind(df['Sum of Shortest Paths'][df['Attack Type'] == 'PC Rich Club'],df['Sum of Shortest Paths'][df['Attack Type'] == 'Degree Rich Club'])

	sns.set(style="whitegrid")
	colors= sns.color_palette(['#FFC61E','#3F6075'])
	sns.boxplot(data=ebtw_df,x='Worm',hue='Node Type',y="Edge Betweenness",palette=colors,showfliers=False)
	# sns.violinplot(data=betw_df,x='Task',hue='Node Type',y="Edge Betweenness",palette=colors)
	for i,Worm in enumerate(worms):
		stat = tstatfunc(ebtw_df['Edge Betweenness'][(ebtw_df['Node Type'] == 'PC') & (ebtw_df['Worm']==Worm)],ebtw_df['Edge Betweenness'][(ebtw_df['Node Type'] == 'Degree') & (ebtw_df['Worm']==Worm)],15)
		# maxvaly = np.nanmax(np.array(betw_df['Edge Betweenness'][betw_df['Task']==task])) + (np.nanstd(np.array(betw_df['Edge Betweenness'][betw_df['Task']==task]))/2)
		maxvaly = np.mean(ebtw_df['Edge Betweenness'])
		sns.plt.text(i,maxvaly,stat,ha='center',color='black',fontsize=sns.plotting_context()['font.size'])
	sns.plt.savefig('/home/despoB/mb3152/dynamic_mod/figures/c_elegans_f_ebtw_e.pdf',dpi=3600)
	sns.plt.show()

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

def igraph_2_networkx(g):
	return nx.Graph(g.get_edgelist())

def clubness(network,niters=100,randomize_topology=True,permute_strength=False):
	graph = network.community.graph
	pc = np.array(network.pc)
	assert np.isnan(pc).any() == False
	pc_emperical_phis = RC(graph, scores=pc).phis()
	pc_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=pc).phis() for i in range(niters)],axis=0)
	pc_normalized_phis = pc_emperical_phis/pc_average_randomized_phis
	degree_emperical_phis = RC(graph, scores=graph.strength(weights='weight')).phis()
	degree_average_randomized_phis = np.nanmean([RC(preserve_strength(graph,randomize_topology=randomize_topology,permute_strength=permute_strength),scores=graph.strength(weights='weight')).phis() for i in range(niters)],axis=0)
	degree_normalized_phis = degree_emperical_phis/degree_average_randomized_phis
	return np.array(pc_normalized_phis),np.array(degree_normalized_phis)

def plot_clubness(diverse_clubness,rich_clubness,savestr,cutoff=0.90):
	sns.set_style("white")
	sns.set_style("ticks")
	with sns.plotting_context("paper",font_scale=1.5):
		try: #assume it is average across densities
			sns.tsplot(np.array(rich_clubness)[:,:int((diverse_clubness.shape[1]* cutoff))],color='b',condition='rich club',ci=95)
			sns.tsplot(np.array(diverse_clubness)[:,:int((diverse_clubness.shape[1]* cutoff))],color='r',condition='diverse club',ci=95)
		except: #if it's not, it's not
			sns.tsplot(np.array(rich_clubness)[:int((diverse_clubness.shape[1]* cutoff))],color='b',condition='rich club',ci=95)
			sns.tsplot(np.array(diverse_clubness)[:int((diverse_clubness.shape[1]* cutoff))],color='r',condition='diverse club',ci=95)
		plt.ylabel('clubness')
		plt.xlabel('rank')
		sns.despine()
		plt.legend()
		plt.tight_layout()
		sns.plt.savefig('%s.pdf'%(savestr),dpi=3600)
		sns.plt.close()

def check_network(network):
	loops = np.array(network.community.graph.is_loop())
	multiples = np.array(network.community.graph.is_multiple())
	assert np.isnan(network.pc).any() == False
	assert len(loops[loops==True]) == 0.0
	assert len(multiples[multiples==True]) == 0.0
	assert np.min(network.community.graph.degree()) > 0
	assert np.isnan(network.community.graph.degree()).any() == False

def test_louvain(matrix,savestr):
	graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=1,mst=True)
	rs = np.arange(1,16)*.05
	pcs = np.zeros((len(rs),graph.vcount()))
	community_sizes = np.zeros(len(rs))
	for idx,r in enumerate(rs):
		nxg = igraph_2_networkx(graph)
		louvain_vc = louvain.best_partition(nxg,resolution=r)
		vc = brain_graphs.brain_graph(VertexClustering(graph,louvain_vc.values()))
		print len(vc.community.sizes())
		community_sizes[idx] = len(vc.community.sizes())
		pcs[idx] = vc.pc
	rmatrix = np.zeros((len(pcs),len(pcs)))
	for i,j in permutations(range(len(pcs)),2):
		rmatrix[i,j] = scipy.stats.spearmanr(pcs[i],pcs[j])[0]
	np.fill_diagonal(rmatrix,1)
	heatmap = sns.heatmap(rmatrix)
	heatmap.set_yticklabels(np.flip(community_sizes,0).astype(int),rotation=360)
	heatmap.set_xticklabels(community_sizes.astype(int),rotation=90)
	sns.plt.savefig(savestr)

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

class Network:
	def __init__(self,networks,rankcut,names,columns):
		vcounts = []
		for network in networks:
			check_network(network)
			vcounts.append(network.community.graph.vcount())
		self.vcounts = vcounts
		self.networks = networks
		self.names = names
		self.columns = columns
		self.rankcut = int(networks[0].community.graph.vcount()*rankcut)
	def calculate_clubness(self,niters=100,randomize_topology=True,permute_strength=False):
		for idx,network in enumerate(self.networks):
			diverse_clubness = np.zeros((self.vcounts[idx]))
			rich_clubness = np.zeros((self.vcounts[idx]))
			diverse_clubness,rich_clubness = clubness(network,niters=niters,randomize_topology=randomize_topology,permute_strength=permute_strength)
			temp_df = pd.DataFrame()
			temp_df["rank"] = np.arange((self.vcounts[idx]))
			temp_df['clubness'] = diverse_clubness
			temp_df['club'] = 'diverse'
			for cidx,c in enumerate(self.columns):
				temp_df[c] = self.names[idx].split('_')[cidx]
			if idx == 0: df = temp_df.copy()
			else: df = df.append(temp_df)
			temp_df = pd.DataFrame()
			temp_df["rank"] = np.arange((self.vcounts[idx]))
			temp_df['clubness'] = rich_clubness
			temp_df['club'] = 'rich'
			for cidx,c in enumerate(self.columns):
				temp_df[c] = self.names[idx].split('_')[cidx]
			df = df.append(temp_df)
		self.clubness = df
	def calculate_intersect(self):
		for idx,network in enumerate(self.networks):
			intersect_results = club_intersect(network,self.rankcut)
			temp_df = pd.DataFrame(index=np.arange(1))
			temp_df["percent overlap"] = intersect_results[0]
			temp_df['percent community, diverse club'] = intersect_results[1]
			temp_df['percent community, rich club'] = intersect_results[2]
			temp_df['condition'] = network.names[idx].split("_")[0]
			if idx == 0:
				df = temp_df.copy()
				continue
			df = df.append(temp_df)
		df["percent overlap"] = df["percent overlap"].astype(float)
		df['percent community, diverse club'] = df['percent community, diverse club'].astype(float)
		df['percent community, rich club'] = df['percent community, rich club'].astype(float)
		self.intersect = df

def run_networks(network,nrandomiters=100,attack=False,rankcut=.8,community_alg='infomap',randomize_topology=True,permute_strength=False,resolution=1):

	network='human'
	nrandomiters=100
	rankcut=.80
	community_alg='infomap'
	randomize_topology = True
	permute_strength = False

	sns.set_style("white")
	sns.set_style("ticks")
	if network == 'human':
		networks = []
		names = []
		df = pd.DataFrame(columns=["percent overlap", 'percent community, diverse club','percent community, rich club','task'])
		network_names = np.array(pd.read_csv('%s/diverse_club/Consensus264.csv'%(homedir),header=None)[36].values)
		network_df = pd.DataFrame(columns=['task',"club", "network",'number of club nodes'])
		# rankcut = int(graph.vcount()*rankcut)
		for task in tasks:
			matrix = np.load('/%s/diverse_club/graphs/%s.npy'%(homedir,task))
			diverse_clubness = np.zeros((len(costs),264))
			rich_clubness = np.zeros((len(costs),264))
			mean_pc = np.zeros((len(costs),264))
			mean_degree = np.zeros((len(costs),264))
			for idx,cost in enumerate(costs):
				graph = brain_graphs.matrix_to_igraph(matrix.copy(),cost=cost,mst=True)
				if community_alg == 'infomap':
					vc = brain_graphs.brain_graph(graph.community_infomap(edge_weights='weight'))
				if community_alg == 'walktrap':
					vc = brain_graphs.brain_graph(graph.community_walktrap(weights='weight').as_clustering())
				if community_alg == 'louvain':
					nxg = igraph_2_networkx(graph)
					louvain_vc = louvain.best_partition(nxg,resolution=1)
					vc = brain_graphs.brain_graph(VertexClustering(graph,louvain_vc.values()))
				networks.append(vc)
				names.append('%s_%s'%(task,cost))
		n = Network(networks,.8,names)
		n.calculate_intersect()
		n.calculate_clubness(niters=nrandomiters,randomize_topology=randomize_topology,permute_strength=permute_strength)
		plot_clubness(n.diverse_clubness,n.rich_clubness,'%s/diverse_club/figures/human_clubness_task_%s_community_alg_%s_%s_rt_%s_ps_%s'% \
			(homedir,task,community_alg,resolution,randomize_topology,permute_strength))

		# 		#human specific analyses, since we have nice names for the communities
		# 		rich_rank = np.argsort(graph.strength(weights='weight'))[rankcut:]
		# 		diverse_rank = np.argsort(pc)[rankcut:]
		# 		for network in network_names:
		# 			diverse_networks = network_names[diverse_rank]==network
		# 			rich_networks = network_names[rich_rank]==network
		# 			network_df = network_df.append({'task':task,'club':'diverse club','network':network,'number of club nodes':len(diverse_networks[diverse_networks==True])},ignore_index=True)
		# 			network_df = network_df.append({'task':task,'club':'rich club', 'network':network,'number of club nodes':len(rich_networks[rich_networks==True])},ignore_index=True)
		# 		#clubness	
		# 		diverse_clubness[idx], rich_clubness[idx] = clubness(graph,vc,
		# 	plot_clubness(diverse_clubness,rich_clubness,'%s/diverse_club/figures/human_clubness_task_%s_community_alg_%s_%s_rt_%s_ps_%s'%(homedir,task,community_alg,resolution,randomize_topology,permute_strength))
		# sns.barplot(data=network_df,x='network',y='number of club nodes',hue='club',palette=sns.color_palette(['#7EE062','#050081']))
		# sns.plt.xticks(rotation=90)
		# sns.plt.ylabel('mean number of club nodes in network')
		# sns.plt.title('rich and diverse club network membership across tasks and costs')
		# sns.plt.tight_layout()
		# sns.plt.savefig('/%s/diverse_club/figures/human_both_club_membership_%s.pdf'%(homedir,community_alg),dpi=3600)
		# sns.plt.close()
		# df["percent Overlap"] = df["percent overlap"].astype(float)
		# df['percent Community, diverse club'] = df['percent community, diverse club'].astype(float)
		# df['Percent Community, rich club'] = df['percent community, rich club'].astype(float)
		sns.barplot(data=n.intersect_df,x='percent overlap',y='condition')
		sns.plt.xlim((0,1))
		sns.plt.savefig('/%s/diverse_club/figures/%s_percent_overlap_human_range_%s.pdf'%(homedir,network,community_alg))
		sns.plt.close()
		sns.barplot(data=n.intersect_df,x='percent community, diverse club',y='condition')
		sns.plt.savefig('/%s/diverse_club/figures/%s_percent_community_diverse_%s.pdf'%(homedir,network,community_alg))
		sns.plt.close()
		sns.barplot(data=n.intersect_df,x='percent community, rich club',y='condition')
		sns.plt.xticks([0,0.2,0.4,0.6,0.8,1])
		sns.plt.savefig('/%s/diverse_club/figures/%s_percent_community_rich_%s.pdf'%(homedir,network,community_alg))
		sns.plt.close()

def corrfunc(x, y, **kws):
	r, _ = pearsonr(x, y)
	ax = plt.gca()
	ax.annotate("r={:.3f}".format(r) + ",p={:.3f}".format(_),xy=(.1, .9), xycoords=ax.transAxes)

def tstatfunc(x, y,bc=False):
	t, p = scipy.stats.ttest_ind(x,y)
	if bc != False:
		bfc = np.around((p * bc),5)
		if bfc <= 0.05:
			return "t=%s,p=%s,bf=%s" %(np.around(t,3),np.around(p,5),bfc)
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


if len(sys.argv) > 1:
	if sys.argv[1] == 'perf':
		performance_across_tasks()
	if sys.argv[1] == 'forever':
		a = 0
		while True:
			a = a - 1
			a = a + 1
	if sys.argv[1] == 'pc_edge_corr':
		task = sys.argv[2]
		atlas = 'power'
		subjects = np.array(hcp_subjects).copy()
		subjects = list(subjects)
		subjects = remove_missing_subjects(subjects,task,atlas)
		static_results = graph_metrics(subjects,task,atlas)
		subject_pcs = static_results['subject_pcs']
		matrices = static_results['matrices']
		pc_edge_corr = pc_edge_correlation(subject_pcs,matrices,path='/home/despoB/mb3152/dynamic_mod/results/hcp_%s_power_pc_edge_corr_z.npy' %(task))
	if sys.argv[1] == 'graph_metrics':
		# subjects = remove_missing_subjects(list(np.array(hcp_subjects).copy()),sys.argv[2],sys.argv[3])
		# subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',sys.argv[2],sys.argv[3]))
		# graph_metrics(subjects,task=sys.argv[2],atlas=sys.argv[3],run_version='fz_wc',run=True)
		subjects = []
		dirs = os.listdir('/home/despoB/connectome-data/')
		for s in dirs:
			try: int(s)
			except: continue
			subjects.append(str(s))
		graph_metrics(subjects,task=sys.argv[2],atlas=sys.argv[3],run_version='HCP_900',run=True)
	if sys.argv[1] == 'make_matrix':
		subject = str(sys.argv[2])
		task = str(sys.argv[3])
		atlas = str(sys.argv[4])
		make_static_matrix(subject,task,'hcp',atlas)
	if sys.argv[1] == 'calc_motion':
		subject = str(sys.argv[2])
		task = str(sys.argv[3])
		run_fd(subject,task)
	if sys.argv[1] == 'check_norm':
		atlas = sys.argv[3]
		task = sys.argv[2]
		# subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_fz.npy' %('hcp',sys.argv[2],sys.argv[3]))
		subjects = np.load('/home/despoB/mb3152/dynamic_mod/results/%s_%s_%s_subs_scrub_.2.npy' %('hcp',sys.argv[2],sys.argv[3]))
		check_scrubbed_normalize(subjects,task,atlas='power')
		print 'done checkin, all good!'
	if sys.argv[1] == 'mediation':
		local_mediation(sys.argv[2]) 
	if sys.argv[1] == 'alg_compare':
		subjects = []
		dirs = os.listdir('/home/despoB/connectome-data/')
		for s in dirs:
			try: int(s)
			except: continue
			subjects.append(str(s))
		alg_compare(subjects)
