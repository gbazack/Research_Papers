from __future__ import division


def _kmeanspp(K,X,mu):
	D2=[]
	
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
		
	return mu
	
