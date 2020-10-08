#Comparison of the maximum number of overlapped
k=np.arange(3,4,0.01)
Of1=np.tan(np.pi/k)/(20*k*epsilon1);
Of2=np.tan(np.pi/k)/(30*k*epsilon1);
Of3=np.tan(np.pi/k)/(50*k*epsilon1)

#Plotting
plt.plot(k,Of1,'r--', label="$B=20 KHz$");
plt.plot(k,Of2,'-o', label="$B=30 KHz$");
plt.plot(k,Of3,'b', label="$B=50 KHz$");
plt.ylabel('Maximum number of overlapped chirps ($O_{f_k}$)');
plt.xlabel('k');
plt.legend(loc="upper right", bbox_to_anchor=[1, 1],ncol=3);
plt.grid(True);
plt.xlim(3,4);
plt.show()


#Simulation Scenarios:
#Scenario 1
B1=20; k1=3; epsilon1=0.01
fs1=B1/2; dBm1=B1*k1*epsilon1;
delta_t1=dBm1/np.tan(np.pi/k1)
mu_u1=(B1 - fs1)/delta_t1; t1= np.array([i for i in np.arange(0, delta_t1, 0.001)])
tsquare1= np.array([i**2 for i in t1])
u1_signal1= np.cos(2*np.pi*fs1*t1 + np.pi*mu_u1*tsquare1)
u1_noise1= np.random.normal(0,1, len(u1_signal1))
xu1_signal1= u1_signal1 + u1_noise1
yu1_signal1=signal.correlate(u1_signal1, x1_signal1, mode='same', method='fft')

#Scenario1 signal2
t1_s2= np.array([i for i in np.arange(delta_t1, 2*delta_t1, 0.001)])
tsquare1_s2= np.array([i**2 for i in t1_s2])
u1_signal2= np.cos(2*np.pi*fs1*t1_s2 + np.pi*mu_u1*tsquare1_s2)
u1_noise2= np.random.normal(0,1, len(u1_signal2))
xu1_signal2= u1_signal2 + u1_noise2
yu1_signal2=signal.correlate(u1_signal2, xu1_signal2, mode='full', method='fft')

#Scenario1 signal3
t1_s3= np.array([i for i in np.arange(2*delta_t1, 3*delta_t1, 0.001)])
tsquare1_s3= np.array([i**2 for i in t1_s3])
u1_signal3= np.cos(2*np.pi*fs1*t1_s3 + np.pi*mu_u1*tsquare1_s3)
u1_noise3= np.random.normal(0,1, len(u1_signal3))
xu1_signal3= u1_signal3 + u1_noise3
yu1_signal3= signal.correlate(u1_signal3, xu1_signal3, mode='full', method='fft')


#Scenario1: Plotting the autocorrated impulse responses
#Rescaling
rabscissa= abscissa/100;
ryu1_signal1= yu1_signal1/100;
ryu1_signal2= yu1_signal2/100;
ryu1_signal3= yu1_signal3/100;
ryd1_signal1= yd1_signal1/100;
ryd1_signal2= yd1_signal2/100;
ryd1_signal3= yd1_signal3/100


#Plotting
plt.plot(rabscissa,ry1_signal1,'r--', label="$y^u_1(t)$");
plt.plot(rabscissa,ry1_signal2, label='$y^u_2(t)$');
plt.plot(rabscissa, ry1_signal3, 'g-.', label='$y^u_3(t)$');
plt.ylabel('Cross correlation coefficient');
plt.xlabel('Frequency (MHz)');
plt.legend(loc="upper right", bbox_to_anchor=[1, 1],ncol=4);
plt.grid(True);
plt.xlim(14,26);
plt.show()

#Scenario 1: Plotting the cross power spectral density
f1, Pxy1 = signal.csd(x1_signal1, y1_signal1, fs1, nperseg=1024)
f2, Pxy2 = signal.csd(x1_signal2, y1_signal2, fs1, nperseg=1024)
f3, Pxy3 = signal.csd(x1_signal3, y1_signal3, fs1, nperseg=1024)
f4, Pxy4 = signal.csd(x1_signal4, y1_signal4, fs1, nperseg=1024)

plt.semilogy(f1, np.abs(Pxy1), f2, np.abs(Pxy2), f3, np.abs(Pxy3), f4, np.abs(Pxy4));plt.xlabel('frequency [MHz]');plt.ylabel('Cross Spectral Density');plt.show()


#**************************************************************
#Scenario 2
B2=20; k2=3.35; epsilon2=0.01
fs2=B2/2; dBm2=B2*k2*epsilon2;
delta_t2=dBm2/np.tan(np.pi/k2)
mu_u2=(B2 - fs2)/delta_t2; t2= np.array([i for i in np.arange(0, delta_t2, 0.00001)])
tsquare2= np.array([i**2 for i in t2])
u2_signal1= np.cos(2*np.pi*fs*t2 + np.pi*mu_u2*tsquare2)
u2_noise1= np.random.normal(0,0.1, len(u2_signal1))
xu2_signal1= cu2_signal1 + c2_noise1
yu2_signal1=signal.correlate(u2_signal1, xu2_signal1, mode='same', method='fft')

#Scenario2 signal2
t2_s2= np.array([i for i in np.arange(delta_t2, 2*delta_t2, 0.00001)])
tsquare2_s2= np.array([i**2 for i in t2_s2])
c2_signal2= np.cos(2*np.pi*fs2*t2_s2 + np.pi*mu_u2*tsquare2_s2)
c2_noise2= np.random.normal(0,0.1, len(c2_signal2))
x2_signal2= c2_signal2 + c2_noise2
y2_signal2=signal.correlate(c2_signal2, x2_signal2, mode='same', method='fft')


#Scenario2: Plotting the autocorrated impulse responses
#Rescaling
rabscissa2= abscissa2/10000;
ry2_signal1= y2_signal1/10000;
ry2_signal2= y2_signal2/10000

#Plotting
plt.plot(rabscissa2,ry2_signal1,'r--', label="$y^u_1(t)$");
plt.plot(rabscissa2,ry2_signal2, label='$y^u_2(t)$');
plt.ylabel('Cross correlation coefficient');
plt.xlabel('Frequency (MHz)');
plt.legend(loc="upper right", bbox_to_anchor=[1, 1],ncol=2);
plt.grid(True);
plt.xlim(26.5,46);
plt.show()

#Scenario 2: Plotting the cross power spectral density
f21, Px2y1 = signal.csd(x2_signal1, y2_signal1, fs2, nperseg=1024)
f22, Px2y2 = signal.csd(x2_signal2, y2_signal2, fs2, nperseg=1024)

plt.semilogy(f21, np.abs(Px2y1), f22, np.abs(Px2y2));plt.xlabel('Frequency [MHz]');plt.ylabel('Cross Spectral Density');plt.show()


#**************************************************************
#Scenario 3
B3=20; k3=4; epsilon3=0.01
fs3=B3/2; dBm3=B3*k2*epsilon3;
delta_t3=dBm3/np.tan(np.pi/k3)
mu_u3=(B3 - fs3)/delta_t3; t3= np.array([i for i in np.arange(0, delta_t3, 0.00001)])
tsquare3= np.array([i**2 for i in t3])
u3_signal1= np.cos(2*np.pi*fs3*t3 + np.pi*mu_u3*tsquare3)
u3_noise1= np.random.normal(0,0.1, len(u3_signal1))
xu3_signal1= u3_signal1 + u3_noise1
yu3_signal1=signal.correlate(u3_signal1, xu3_signal1, mode='same', method='fft')

#Scenario3: Plotting the autocorrated impulse responses
#Rescaling
rabscissa2= abscissa2/10000;
ry3_signal1= y3_signal1/10000;

#Plotting
plt.plot(rabscissa2,ry3_signal1,'r--', label="$y^u_1(t)$");
plt.ylabel('Cross correlation coefficient');
plt.xlabel('Frequency (MHz)');
plt.legend(loc="upper right", bbox_to_anchor=[1, 1],ncol=2);
plt.grid(True);
plt.xlim(26.5,46);
plt.show()

#Scenario 3: Plotting the cross power spectral density
f31, Px3y1 = signal.csd(x3_signal1, y3_signal1, fs3, nperseg=1024)



#Plotting using subplots
fig, ((axe11, axe12), (axe21, axe22)) = plt.subplots(2, 2)
#First row
axe11.plot(rabscissa,ry1_signal1,'r--', label="$y^u_1(t)$");
axe11.plot(rabscissa,ry1_signal2, label='$y^u_2(t)$');
axe11.plot(rabscissa, ry1_signal3, 'g-.', label='$y^u_3(t)$');
axe11.axes.set_ylabel('Cross correlation');
axe11.axes.set_xlabel('Time');
axe11.legend(loc="upper right", bbox_to_anchor=[1, 1],ncol=3);
axe11.grid(True);
axe11.axes.set_xlim(0.18,0.35);


axe12.plot(rabscissa2,ry2_signal1,'r--', label="$y^u_1(t)$");
axe12.plot(rabscissa2,ry2_signal2, label='$y^u_2(t)$');
axe12.axes.set_ylabel('Cross correlation');
axe12.axes.set_xlabel('Time');
axe12.legend(loc="upper right", bbox_to_anchor=[1, 1],ncol=2);
axe12.grid(True);
axe12.axes.set_xlim(26.5,46);

#Second row
axe21.semilogy(f1, np.abs(Pxy1), f2, np.abs(Pxy2), f3, np.abs(Pxy3), f4, np.abs(Pxy4));
axe21.axes.set_xlabel('Frequency [MHz]');
axe21.axes.set_ylabel('Cross Power Spectral Density');

axe22.semilogy(f21, np.abs(Px2y1), f22, np.abs(Px2y2));
axe22.axes.set_xlabel('Frequency [MHz]');
axe22.axes.set_ylabel('Cross Spectral Density');

fig.show()

#############**************** BER ANALYSIS
#Scenario1: BER Analysis
SNR_min= 0; SNR_max=11; iteration=10
EbNo= k1*np.array(np.arange(SNR_min, SNR_max, 1),float)
EbNo2= k2*np.array(np.arange(SNR_min, SNR_max, 1),float)
EbNo3= k3*np.array(np.arange(SNR_min, SNR_max, 1),float)

SNR= np.zeros(len(EbNo), float);
SNR2= np.zeros(len(EbNo2), float);
SNR3= np.zeros(len(EbNo3), float)

ber_coef= 1/(2**(2*k1 -1));
ber_coef2= 1/(2**(2*k2 -1));
ber_coef3= 1/(2**(2*k3 -1))

SNR= ber_coef*(10**(EbNo/(k1*10-1)));
SNR2= ber_coef2*(10**(EbNo2/(k2*10-1)));
SNR3= ber_coef3*(10**(EbNo3/(k3*10-1)))

b,a=signal.butter(3,0.125)
instantaneous_error1= np.zeros((iteration, len(EbNo)), float)
instantaneous_error2= np.zeros((iteration, len(EbNo2)), float)
instantaneous_error3= np.zeros((iteration, len(EbNo3)), float)

for i in range(iteration):
	error_matrix1= error_matrix2= error_matrix3= np.zeros(len(EbNo), float)
	
	for j in range(len(SNR)):
		#First scenario
		u1_signal1= np.cos(2*np.pi*fs*t1 + np.pi*mu_u1*tsquare1)
		u1_noise1= SNR[j]*np.random.normal(0,0.1, len(u1_signal1))
		xu1_signal1= u1_signal1 + u1_noise1
		yu1_filfit1=signal.filtfilt(b,a,xu1_signal1)
		error1=abs(yu1_filfit1 - u1_signal1)/2
		error_matrix1[j]=error1.sum()
		
		#Second scenario
		u2_signal1= np.cos(2*np.pi*fs*t2 + np.pi*mu_u2*tsquare2)
		u2_noise1= SNR2[j]*np.random.normal(0,0.1, len(u2_signal1))
		xu2_signal1= cu2_signal1 + u2_noise1
		yu2_filfit1=signal.filtfilt(b,a,xu2_signal1)
		error2=abs(yu2_filfit1 - u2_signal1)/2
		error_matrix2[j]=error2.sum()
		
		#Third scenario
		u3_signal1= np.cos(2*np.pi*fs*t3 + np.pi*mu_u3*tsquare3)
		u3_noise1= SNR3[j]*np.random.normal(0,0.1, len(u3_signal1))
		xu3_signal1= u3_signal1 + u3_noise1
		yu3_filfit1=signal.filtfilt(b,a,xu3_signal1)
		error3=abs(yu3_filfit1 - u3_signal1)/2
		error_matrix3[j]=error3.sum()

		
	instantaneous_error1[i]=error_matrix1
	instantaneous_error2[i]=error_matrix2
	instantaneous_error3[i]=error_matrix3

BER1= instantaneous_error1.sum(axis=0)/k1;
BER2= instantaneous_error2.sum(axis=0)/k2;
BER3= instantaneous_error3.sum(axis=0)/k3

TheoryBER= 0.5*erfc(np.sqrt(10**(k1*EbNo/100)))

#Scenario 1: Plotting BER
plt.plot(EbNo, BER1, 'bs-', EbNo, BER2, 'g^-',EbNo, BER3, 'o-');
plt.ylabel('$\log_{10}(BER)$');
plt.xlabel('Eb/No (dB)');
plt.tick_params(labelsize='medium', width=3, length=3);
plt.title('BER Performance analysis');
plt.legend(['$O_f = 2.886$', '$O_f = 2.034$', '$O_f = 1.249$'], loc='lower left');
plt.margins(0.1,0.25);
plt.xlim(0,10);
plt.ylim(-6,-1);
plt.grid(True, linestyle='-.');
plt.show()

