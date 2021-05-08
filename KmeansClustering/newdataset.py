import numpy as np
import csv
import sys
import os
import random as rand
from os.path import dirname, exists, expanduser, isdir, join, splitext


_path=os.path.dirname('/home/gbazack/Documents/PhD/Simulation/data//')

def load_data(nb_points=2000):

	set_of_data_points=[[0,0,0]]
	"""
	By this function we load data points stored in csv files
	nb_points: is the number of data points required
	set_of_data_points: is a 2-array variable for storing the data points
	"""
	
	if nb_points==2000:
		with open(_path+'/opencellid2000.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[6]), float(row[7]), float(row[8])])
	
	elif nb_points==5000:
		with open(_path+'/opencellid5000.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[6]), float(row[7]), float(row[8])])

	elif nb_points==10000:
		with open(_path+'/opencellid10000.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[6]), float(row[7]), float(row[8])])

				
	else: print("Select the number data point in ",[2000,5000,10000])
	
	#Delete the second item of the ndarray in order to set its cardinality to nb_points
	del set_of_data_points[0]
	
	#return the set of nb_points data points
	return np.array(set_of_data_points)
