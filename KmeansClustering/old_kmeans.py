"""
Implementation of k-means algorithm in 2-dimensional space, with fixed data points
"""
import numpy as np
from scipy.spatial.distance import cdist
from dataset import load_data
from initialisation.random_centers import generate_centers
from initialisation.geo_init import geometric_init




"""
Define a class for kmeans
"""
class _KMeans:
	def __init__(self,nb_points,nb_centers,init='random',iteration=2):
		self.nb_points=nb_points
		self.nb_centers=nb_centers
		#Number of data points must equal or greater than to four times the number of centers
		try:
			if self.nb_points<4*self.nb_centers:
				raise ValueError("Number of data points lower than expected")
		except ValueError:
			print("Number of data points should be (n>4*k)")
		self.init=init
		self.iteration=iteration
		
	#define functions

	"""
	We define a function _kmeans which will be called in the class _KMeans defined below
	"""
	def _kmeans(self,n,k,init='random',iteration=2,a=-9,b=9):
	
		_iter=0
	
		#load data points
		_points=load_data(n)
		"""
		Select the method for generating the k initial centers
		"""	
		if init=='geo_init':
			#generate centers by geometric initialisation method
			_centers=geometric_init(_points,k)
			print _centers
		else:
			#generate centers randomly
			_centers=generate_centers(a,b,n,k)
			print _centers
			
		#compute euclidean distance between data points and centers
		_distances=cdist(_points,_centers,'euclidean')
		print _distances	

	
		while _iter<2:
			_iter+=1		
			intra=[0]  #variable for intra cluster criterion
			#create an ndarray to classify clusters
			_clusters=np.ones((k,n))*100
		
			for i in range(n):
				index_min=np.argmin(_distances[i])
				_clusters.put((index_min,i),index_min)
				_distances.put((i,index_min),100)
			#compute the intra cluster criterion
			squared_distances=np.square(_distances)
		
			for i in range(k):
				s=j=0
				while j<n:
					if _clusters[i][j]<100:
						 s+=squared_distances[j][i]
						 j+=1
					intra.append(s)
			inter=np.sum(intra)
		
			#recompute centers 
			_centers=[[]]
			for i in range(k):
				temp_list=[[]]
				for j in range(n):
					if _clusters[i][j]<100:
						temp_list.append(_points[j])
				del temp_list[0]
				_centers.append(np.median(temp_list,axis=0))
			del _centers[0]
		
			#_iter+=1
			_distances=cdist(_points,_centers,'euclidean')
	
		return inter,intra,_clusters



	def getCluster():
		print this._kmeans(nb_points,nb_centers,init,iteration)[2]
	

	def getIntracluster():
		return	this._kmeans(nb_points,nb_centers,init,iteration)[1]
		
	
	def getIntercluster():
		return	this._kmeans(nb_points,nb_centers,init,iteration)[0]
		
		
		
