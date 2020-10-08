"""
In this file, we evaluate the performance of our proposed Direct Modulation Chirp Spread Spectrum (DM CSS).
The first part will involve generating the sampling time.
Then we design the chirp signals s(t) and the noisy chirp signals x(t)=s(t) + n(t).
The second part of this simulation will consist of devising the Matched filter h(t)
to detect the signal at the receiver side.
Spectral and BER analysis will be implemented in the last part.
=====
@References: https://inst.eecs.berkeley.edu/~ee123/sp15/lab/lab1/lab1-TimeDomain-Sonar.html, https://scipy-cookbook.readthedocs.io/items/FIRFilter.html
*
**
***
@Author: gbazack (morenzack@gmail.com), Sept 2020
"""

#Importing libraries
from __future__ import division
import sys, csv
import numpy as np, matplotlib.pyplot as plt
from numpy import *
from matplotlib.pyplot import *
from scipy import signal
from scipy.special import erfc


class dmcss_modulation:
	def __init__(self, f_min, f_max, c_duration):
		self.fmin= f_min
		self.fmax= fmax
		self.chrip_duration= c_duration
		
	#Generating the sampling time
	def sampling_time(self, k, epsilon):
		dBm= (self.fmax -self.fmin)*k*epsilon
		
		return dBm/np.tan(np.pi/k)
	
	#Generating the up or down chirp signal
	def chirp_signal(self,Amplitude, direction):
		delta_t= self.sampling(k,epsilon)
		fs= int((self.fmax - self.fmin)/2)
		t= np.array([i for i in np.arange(0, delta_t, 0.01)])
		tsquare= np.array([i**2 for i in t])
		
		if direction==1:
			mu_u=(self.fmax - fs)/delta_t			
			c_signal= Amplitude*np.cos(2*np.pi*fs*t + np.pi*mu_u*tsquare)
			
		elif direction==0:
			mu_d=(fs - self.fmin)/delta_t
			c_signal= Amplitude*np.cos(2*np.pi*fs*t - np.pi*mu_d*tsquare)
		
		else:
			c_signal=np.zeros(len(t))
		
		return c_signal
	
	#Adding a white Gaussian noise to the signal
	def received_signal(self, K, mu, sigma, A, d):
		s_signal= self.chirp_signal(A, d)
		noise= np.random.normal(mu,sigma, len(s_signal))
		
		return K*s_signal + noise
	
	#Designing the Matched filter to detect the transmitted signal
	#from the noisy signal
	def mateched_filter(self,k,mu,sigma,A,d):
		s_signal= self.chirp_signal(A, d)
		x_signal=self.received_signal(k,mu,sigma,A,d)
		
		return signal.correlate(s_signal, x_signal, mode='same', method='fft')
	
	def ber_analysis():
		
