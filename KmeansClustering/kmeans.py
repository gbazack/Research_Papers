"""
Implementation of k-means algorithm in 2-dimensional space, with fixed data points
"""
import numpy as np
import random
import matplotlib as plt
from scipy.spatial.distance import cdist
from numpy import linalg
import time as tm
from dataset import load_data
from initialisation.random_centers import generate_centers
from initialisation.geo_init import geometric_init




"""
Define a class for kmeans
"""
class _KMeans:
	def __init__(self, K, X, N=0):
		self.K = K
		if X == []:
			if N == 0:
				raise Exception("If no data is provided, a parameter N (number of points) is needed")
                        else:
                        	self.N = N
                        	self.X = self._init_board_gauss(N, K)
                else:
                	self.X = X
                	self.N = len(X)
                self.mu = None
                self.clusters = None
                self.method = None
                self.inter = 0.0
                self.separation=0.0
                self.cohesion=0.0
                self.convergence = 0.0
                
                #define functions
                
                """
                We define a function _kmeans which will be called in the class _KMeans defined below
                """
        ###########################################
        def _init_board_gauss(self, N, k):
               	n = float(N)/k
               	X = []
                	
               	for i in range(k):
               		c = (random.uniform(-1,1), random.uniform(-1,1),random.uniform(-1,1))
               		s = random.uniform(0.05,0.15)
               		x = []
                		
               		while len(x) < n:
               			a,b,c = np.array([np.random.normal(c[0],s),np.random.normal(c[1],s),np.random.normal(c[2],s)])
                			
               			if abs(a) and abs(b) and abs(c)<1:
               				x.append([a,b,c])
               		X.extend(x)
               	X = np.array(X)[:N]
               	return X


	################################--kmeans++
	def _kmeanspp(self):
		D2=[]
		K=self.K
		X=self.X
		mu=self.mu
		
		def _dist_from_centers(mu,X):
			cent = mu
			D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
		#        self.D2 = D2
		 
		 
		def _choose_next_center(D2):
			probs = D2/D2.sum()
			cumprobs = probs.cumsum()
			r = random.random()
			ind = np.where(cumprobs >= r)[0][0]

			return(X[ind])
		 
		def init_centers():
			mu = random.sample(X, 1)
			while len(mu) < K:
				_dist_from_centers()
				mu.append(_choose_next_center())
			
		self.mu=mu


	###################################--geo_init
	def geometric_init(self):
		X=self.X
		K=self.K
		
		#Generate the first center
		first_center=np.median(X,axis=0)
		centers=[first_center]
		#select the biggest distance between the 
		radius=np.amax(linalg.norm(X - first_center))
	
		#_theta=math.pi/(k-1)
		delta=radius/5
		theta=np.pi/(K-1)
		phi=np.pi/(K-1)
		
		#Getting coordinates of the remaining centers
		theta_j=[random.uniform((j-1)*theta, j*theta) for j in range(1,K)]
		x1=delta*np.array([np.cos(phi) for i in range(K)])
		x2=delta*np.sin(phi)*np.cos(theta_j)
		x3=delta*np.sin(phi)*np.sin(theta_j)
		
		
		for j in range(K-1):
			x=np.array([x1[i], x2[i], x3[i]])
			centers.append(x)
		#return np.array(centers)
		self.mu=centers

                	
        ##############################################--plot
        def plot_board(self):
               	X = self.X
               	fig = plt.figure(figsize=(5,5))
               	plt.xlim(-1,1)
               	plt.ylim(-1,1)
                	
               	if self.mu and self.clusters:
               		mu = self.mu
               		clus = self.clusters
               		K = self.K
                		
              		for m, clu in clus.items():
               			cs = cm.spectral(1.*m/self.K)
               			plt.plot(mu[m][0], mu[m][1], 'o', marker='*', markersize=12, color=cs)
               			plt.plot(zip(*clus[m])[0], zip(*clus[m])[1], '.', markersize=8, color=cs, alpha=0.5)
                			
               	else:
               		plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
                		
               	if self.method == '++':
               		tit = 'K-means++'
               	else:
               		tit = 'K-means with random initialization'
                		
               	pars = 'N=%s, K=%s' % (str(self.N), str(self.K))
               	plt.title('\n'.join([pars, tit]), fontsize=16)
               	plt.savefig('kpp_N%s_K%s.png' % (str(self.N), str(self.K)), bbox_inches='tight', dpi=200)

                	
        #####################################
        def _cluster_points(self):
               	mu = self.mu
               	clusters  = {}
               	
              	for x in self.X:
               		bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
                		
               		try:
               			clusters[bestmukey].append(x)
               		except KeyError:
               			clusters[bestmukey] = [x]
               	self.clusters = clusters
                	
                
        #########################################
        def _reevaluate_centers(self):
               	clusters = self.clusters
              	newmu = []
               	keys = sorted(self.clusters.keys())
                	
               	for k in keys:
               		newmu.append(np.mean(clusters[k], axis = 0))
               	self.mu = newmu
               	

	######################################	
	def _has_converged(self):
		K = len(self.oldmu)
               	return(set([tuple(a) for a in self.mu]) == set([tuple(a) for a in self.oldmu]) and len(set([tuple(a) for a in self.mu])) == K)
                	
                
                
       #############################################
	def find_centers(self, method='random'):
		self.method = method
               	X = self.X
               	K = self.K
               	self.oldmu = random.sample(X, K)
                	
               	if method == '++':
               		# Initialize with kmeans++
               		t1=tm.time()
               		self._kmeanspp()
               	elif method == 'geo':
               		# Initialize with geometric scheme
               		t1=tm.time()
               		self.geometric_init()
               	else:
               		# Initialize to K random centers
               		t1=tm.time()
               		self.mu = random.sample(X, K)
               	while not self._has_converged():
               		self.oldmu = self.mu
               		# Assign all points in X to clusters
               		self._cluster_points()
               		# Reevaluate centers
               		self._reevaluate_centers()
               	t2=tm.time()
               	self.convergence=t2-t1
               	
        #################################################--inter cluster       		
	def inter_cluster(self):
		clusters=self.clusters
		K=self.K
		mu=self.mu
		inter=self.inter
		_sum=[0]
		intra=[0]
		
		for i in range(len(mu)):
			for points in clusters[i]:
				d=np.square(linalg.norm(points - mu[i]))
				_sum.append(d)
			
			intra.append(np.sum(_sum))
		
		inter=np.sum(intra)
		self.inter=inter
		
	##########################################-----------separation	
	def find_separation(self):
		clusters=self.clusters
		separation=self.separation
		_sum=[]
		
		for i in range (len(clusters)-1):
			for j in range(i+1,len(clusters)):
				d=np.array([max([np.linalg.norm(x-c) for c in clusters[j]]) for x in clusters[i]])
				
			_sum.append(np.max(d))
		
		separation=(np.max(_sum))
		self.separation=separation



	##########################################-----------separation	
	def find_cohesion(self):
		clusters=self.clusters
		cohesion=self.cohesion
		_sum=[]
		
		for j in range (len(clusters)):
			d=np.array([max([np.linalg.norm(x-c) for c in clusters[j]]) for x in clusters[i]])
			_sum.append(np.max(d))
		
		cohesion=(np.max(_sum))
		self.cohesion=cohesion



