"""
Functions to compute the Bit Errot Rate (BER) of 
the proposed CSS model
references: http://codehubpython.appspot.com/digcom
"""
import numpy as np 
from scipy.special import erfc 


class BPSK(object):
	def __init__(self, bit_length):
		self.bit_length = bit_length
		
	#binary signal generator
	def signal_generator(self):
    #uniformly distributed sequence
		b = np.random.uniform(-1, 1, self.bit_length)
    	
    #convert to binary and save in signal
		signal = np.zeros((self.bit_length),float)
    	
		for i in range(self.bit_length):
			if b[i] < 0:
				signal[i]=-1
			else:
				signal[i]=1
    	
		return signal
    	
	def noise_generator(self):
    #Gaussian Noise
		noise = np.random.randn(self.bit_length)
    	
		return noise
    
	def recieved_signal(self, signal, SNR, noise):
		recieved_signal = signal + SNR*noise
  	
		return recieved_signal

	def detected_signal(self, recieved_signal):
		detected_signal = np.zeros((self.bit_length),float)
		
		for i in range(self.bit_length):
			if recieved_signal[i] < 0:
				detected_signal[i]=-1
			else:
				detected_signal[i]=1
				
		return detected_signal

	def error(self, signal, detected_signal):
		error_matrix = abs((detected_signal - signal)/2)
		error=error_matrix.sum()
    
		return error
    
	def theoryBerBPSK(self, SNR_db):
		#calculate theoritical BER
		theoryBER = np.zeros(len(SNR_db),float)
		
		for i in range(len(SNR_db)):
			theoryBER[i] = 0.5*erfc(np.sqrt(10**(SNR_db[i]/10)))
		
		return theoryBER
		
