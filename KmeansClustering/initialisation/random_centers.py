import numpy as np
import random
"""
Define function for generating centers randomly
"""

def generate_centers(a,b,n,k):
	"""
	a,b	: extremities of intervall [a,b]; and coordonnees' values of every center will be in taken from this intervall
	n	: number of data points
	k	: number of centers to be generated
	"""

	_list=[random.uniform(a,b) for i in range(n)]
	__list=[[0,0]]

	for i in range(k-1):
		__list.append(random.sample(_list,2))
		random.shuffle(_list)

	return np.array(__list)

