"""
Functions to compute the spectrall efficiency C/B of 
the proposed CSS model
"""
import sys
import csv
import numpy as np
import math
#import random
#import matplotlib as plt


def spectral(ebno,Nio,Nb):
	B = 7     #Bandwidth capacity 7MHz
	ebno = np.array(ebno)    #Cast the list into array
	one  = np.ones(len(ebno)) #Create a list
	spectr = np.array(())
	
	var_tan = np.tan(((2*Nio + 1)*np.pi)/(4*Nio + 3))	
	spectr  = np.log2( one + (Nb/B**2)*ebno*var_tan)
	
	return spectr
	

#run the code
if __name__ == "__main__":
	ebno = [i for i in range(11)] #Initialization of the ebno variable
	Nio  = 2;	Nb = 3
	
	#Save the results in a text file
	#f=open("results/result.txt","w")
	#f.write("Spectral Efficiency / Ebno0\n")
	
	#Save the results in a csv file
	"""
	with open('results/results.csv', 'w', newline='') as resultcsv:
		fieldnames = ['Nb', 'Nio', 'ebno', 'Spectral']
		f = csv.DictWriter(resultcsv, fieldnames=fieldnames)
		f.writeheader()
		
		for Nb in range(1,11):
			#f.write(str(Nb)+"\n")
			for Nio in range(1,11):
				sp = spectral(ebno,Nio,Nb)
				for j in range(len(sp)):
					f.writerow({'Nb': Nb, 'Nio': Nio, 'ebno':ebno[j], 'Spectral': sp[j]})
	"""
	for Nb in range(1,11):
		with open('results/resultsNb'+str(Nb)+'.csv', 'w', newline='') as resultcsv:
			fieldnames = ['ebno', 'Spectral.Efficiency', 'Time.Slot']
			f = csv.DictWriter(resultcsv, fieldnames=fieldnames)
			f.writeheader()
			
			for Nio in range(1,11):
				sp = spectral(ebno,Nio,Nb)
				
				for j in range(len(sp)):
					f.writerow({'ebno':ebno[j], 'Spectral.Efficiency': sp[j], 'Time.Slot': 'Nio'+str(Nio)})
	
	print("Done!")
	
