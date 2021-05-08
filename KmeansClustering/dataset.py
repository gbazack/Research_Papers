import numpy as np
import csv
import sys
import os
import random as rand
from os.path import dirname, exists, expanduser, isdir, join, splitext


_path=os.path.dirname('/home/gbazack/Documents/PhD/Simulation/data')

def load_data(nb_points=100):

	set_of_data_points=[[0,0]]
	"""
	By this function we load data points stored in csv files
	nb_points: is the number of data points required
	set_of_data_points: is a 2-array variable for storing the data points
	"""
	
	if nb_points==100:
		with open(_path+'/dataset100.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])
	
	elif nb_points==200:
		with open(_path+'/dataset200.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==300:
		with open(_path+'/dataset300.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==400:
		with open(_path+'/dataset400.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==500:
		with open(_path+'/dataset500.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==600:
		with open(_path+'/dataset600.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==700:
		with open(_path+'/dataset700.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==800:
		with open(_path+'/dataset800.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==900:
		with open(_path+'/dataset900.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])

	elif nb_points==1000:
		with open(_path+'/dataset1000.csv','rb') as _csvfile:
			_reader=csv.reader(_csvfile)
			for row in _reader:
				set_of_data_points.append([float(row[0]), float(row[1])])
				
	else: print("Select the number data point in ",[100,200,300,400,500,600,700,800,900,1000])
	
	#Delete the second item of the ndarray in order to set its cardinality to nb_points
	del set_of_data_points[1]
	
	#return the set of nb_points data points
	return np.array(set_of_data_points)
