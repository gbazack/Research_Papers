from __future__ import division
import numpy as np
import scipy.spatial.distance as distance
import math 

def geometric_init(set_of_data_points,k):
	#Generate the first center
	first_center=list(np.median(set_of_data_points,axis=0))
	centers=[first_center]
	#select the biggest distance between the 
	radius=np.amax(distance.cdist([first_center],set_of_data_points,'euclidean'))
	
	#_theta=math.pi/(k-1)
	d=radius/2
	for j in range(1,k):
		centers.append([d*math.cos(j*math.pi/(k-1)), d*math.sin(j*math.pi/(k-1))])
		
	
	return np.array(centers)
	
