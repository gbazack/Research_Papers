"""
Main functions to run the simulation
references: http://codehubpython.appspot.com/digcom
"""

import matplotlib.pyplot as plt
import numpy as np
from BerBPSK import BPSK

def run():
	bit_length  = 10000
	#bandwidth = 7
	iter_len = 100
	
	#Timeslot ranging from 1 to 3
	#timeslot = np.array(np.arange(1, 4, 1),int)
	timeslot = np.array([1,2,3])
	bit_len_tab = np.zeros(len(timeslot),int)	
	
	#noise calculations
	#SNR db range: -2 to 10
	SNR_db = np.array(np.arange(-12, 12, 2),float)
	SNR = np.zeros(len(SNR_db), float)
	
  #SNR expression based on our model
	#SNR_p = np.zeros(len(SNR_db), float)
	#var_tan = np.tan(((2*timeslot + 1)*np.pi)/(4*timeslot + 3))
	#SNR_p = 10**(-SNR_db*(bit_length/(bandwidth**2)*var_tan))
	
	#for i in range(len(SNR_p)):
	#	SNR_p[i] = 10**(-(SNR_db[i]*(bit_length/(bandwidth**2)*var_tan))/20)
  
	for i in range (len(SNR)):
		SNR[i] = 1/np.sqrt(2)*10**(-SNR_db[i]/20)
	
	for i in range(len(timeslot)):
		bit_len_tab[i] = int(np.ceil(bit_length*timeslot[i]*1))
  
  #instance of BPSK class
	bpsk = BPSK(bit_len_tab[0])
	bpsk1 = BPSK(bit_len_tab[1])
	bpsk2 = BPSK(bit_len_tab[2])
	#bpsk3 = BPSK(bit_len_tab[3])	
	#bpsk4 = BPSK(bit_len_tab[4])
	  
	#accumulate bit error rate for various iterations
	instantaneous_error = instantaneous_error1 = instantaneous_error2 = instantaneous_error3 = instantaneous_error4 = np.zeros((iter_len, len(SNR_db)), float)
	for iter in range(iter_len):
		error_matrix = error_matrix1 = error_matrix2 = error_matrix3 = error_matrix4 = np.zeros(len(SNR_db), float)
		
		for i in range(len(SNR)):
			#First time slot
			signal =  bpsk.signal_generator()
			noise = bpsk.noise_generator()
			recieved_signal = bpsk.recieved_signal(signal, SNR[i], noise)
			detected_signal = bpsk.detected_signal(recieved_signal)
			error = bpsk.error(signal, detected_signal)
			error_matrix[i] = error
			
			#Second timeslot
			signal1 =  bpsk1.signal_generator()
			noise1 = bpsk1.noise_generator()
			recieved_signal1 = bpsk1.recieved_signal(signal1, SNR[i], noise1)
			detected_signal1 = bpsk1.detected_signal(recieved_signal1)
			error1 = bpsk1.error(signal1, detected_signal1)
			error_matrix1[i] = error1

			#Third timeslot
			signal2 =  bpsk2.signal_generator()
			noise2 = bpsk2.noise_generator()
			recieved_signal2 = bpsk2.recieved_signal(signal2, SNR[i], noise2)
			detected_signal2 = bpsk2.detected_signal(recieved_signal2)
			error2 = bpsk2.error(signal2, detected_signal2)
			error_matrix2[i] = error2

			#Fourth timeslot
			#signal3 =  bpsk3.signal_generator()
			#noise3 = bpsk3.noise_generator()
			#recieved_signal3 = bpsk3.recieved_signal(signal3, SNR[i], noise3)
			#detected_signal3 = bpsk3.detected_signal(recieved_signal3)
			#error3 = bpsk3.error(signal3, detected_signal3)
			#error_matrix3[i] = error3

			#Fifth timeslot
			#signal4 =  bpsk4.signal_generator()
			#noise4 = bpsk4.noise_generator()
			#recieved_signal4 = bpsk4.recieved_signal(signal4, SNR[i], noise4)
			#detected_signal4 = bpsk4.detected_signal(recieved_signal4)
			#error4 = bpsk4.error(signal4, detected_signal4)
			#error_matrix4[i] = error4

  		
		instantaneous_error[iter]=error_matrix
		instantaneous_error1[iter]=error_matrix1
		instantaneous_error2[iter]=error_matrix2
		#instantaneous_error3[iter]=error_matrix3
		#instantaneous_error4[iter]=error_matrix4
  	
  #Average BER
	BerBPSK = instantaneous_error.sum(axis=0)/(iter_len*bit_len_tab[0])
	BerBPSK1 = instantaneous_error1.sum(axis=0)/(iter_len*bit_len_tab[1])
	BerBPSK2 = instantaneous_error2.sum(axis=0)/(iter_len*bit_len_tab[2])
	#BerBPSK3 = instantaneous_error3.sum(axis=0)/(iter_len*bit_len_tab[3])
	#BerBPSK4 = instantaneous_error4.sum(axis=0)/(iter_len*bit_len_tab[4])

  #calculate theoritical BER
	theoryBerBPSK = bpsk.theoryBerBPSK(SNR_db)
	#theoryBerBPSK1 = bpsk1.theoryBerBPSK(SNR_db)
	#theoryBerBPSK2 = bpsk2.theoryBerBPSK(SNR_db)
  
  #save data in .csv file
  #mention all data in following line
	data  = zip(theoryBerBPSK, BerBPSK, BerBPSK1, BerBPSK2, SNR_db)
  
  #add header(title) for each data, to recognize the column in .csv file
	#np.savetxt('results/bpsk.csv', data, fmt = '%s, %s, %s', header='Simulation, Theory, Eb/No', comments="")
  
  #plot data (not from the .csv file)
	plt.semilogy(SNR_db, theoryBerBPSK, 'o')
	plt.semilogy(SNR_db, BerBPSK,'-r')
	plt.semilogy(SNR_db, BerBPSK1,'--')
	plt.semilogy(SNR_db, BerBPSK2,'-b')
	#plt.semilogy(SNR_db, BerBPSK3,'-y')
	#plt.semilogy(SNR_db, BerBPSK4,'-g')
	#plt.semilogy(SNR_db, theoryBerBPSK1, 'v')
	#plt.semilogy(SNR_db, theoryBerBPSK2, 'p')
	plt.ylabel('BER')
	plt.xlabel('Eb/No (dB)')
	plt.title('BER Performance analysis for 3 bits')
	plt.legend(['Theory', 'Simulation, Nio1', 'Simulation, Nio2', 'Simulation, Nio3'], loc='lower left')#, 'Simulation, Nio5', 'Simulation, Nio10'], loc='lower left')
	plt.grid()
	plt.show()
	  
# Standard boilerplate to call the main() function.
if __name__ == '__main__':
	run()
	print("Done!")
