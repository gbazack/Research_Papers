import numpy as np
import matplotlib.pyplot as plt

_lambda=0.25; _lambda2=0.95
x=interval= np.linspace(-1, 1, 1000)

#Hard thresholding
hard_th= np.piecewise(x, [abs(x)<=_lambda, abs(x)>_lambda], [0, lambda x:x])
#Soft thresholding
soft_th= np.piecewise(x, [abs(x)<=_lambda, x>_lambda, x<-_lambda], [0, lambda x:x-_lambda, lambda x:x+_lambda])
#Semi-soft thresholding
semisoft_th= np.piecewise(x, [abs(x)<=_lambda, _lambda<abs(x), abs(x)<=_lambda2, abs(x)>_lambda2], [0, lambda x:np.sign(x)*(_lambda2*(abs(x)-_lambda))/(_lambda2 -_lambda), lambda x:np.sign(x)*(_lambda2*(abs(x)-_lambda))/(_lambda2 -_lambda), lambda x:x])
#Non-Negative Garrote thresholding
garrote_th= np.piecewise(x, [abs(x)<=_lambda, abs(x)>_lambda], [0, lambda x:x-(pow(_lambda,2)/x)])

#Ploting thresholding functions
#fig, (axe11, axe12) = plt.subplots(nrows=1, ncols=2)
#
plt.plot(interval, hard_th, 'b', linewidth=2)+plt.plot(interval, interval, 'r--', linewidth=1)
#plt.vlines([-0.25, 0.5], -1,1, colors='r', linestyles='dashed')
#plt.annotate('$-\lambda$', xy=(-0.25, 0.25), xytext=(-0.3, 0.45), arrowprops=dict(facecolor='black', shrink=0.1))
#plt.annotate('$\lambda$', xy=(0.25, 0.75), xytext=(0.3, 0.85), arrowprops=dict(facecolor='black', shrink=0.1))
plt.ylabel('$\delta^h_{\lambda}(d_{ij})$')
plt.xlabel('Wavelet coefficients $d_{ij}$')
plt.title('Hard thresholding')
plt.grid(True, linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
#
plt.plot(interval, soft_th, 'b', linewidth=2)+plt.plot(interval, interval, 'r--', linewidth=1)
plt.ylabel('$\delta^s_{\lambda}(d_{ij})$')
plt.xlabel('Wavelet coefficients $d_{ij}$')
plt.title('Soft thresholding')
plt.grid(True, linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
#
plt.plot(interval, semisoft_th, 'b', linewidth=2)+plt.plot(interval, interval, 'r--', linewidth=0.5)
plt.ylabel('$\delta^{ss}_{\lambda}(d_{ij})$')
plt.xlabel('Wavelet coefficients $d_{ij}$')
plt.title('Semi-soft thresholding')
plt.grid(True, linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
#
plt.plot(interval, garrote_th, 'b', linewidth=2)+plt.plot(interval, interval, 'r--', linewidth=1)
plt.ylabel('$\delta^{ng}_{\lambda}(d_{ij})$')
plt.xlabel('Wavelet coefficients $d_{ij}$')
plt.title('Non-negative garrote thresholding')
plt.grid(True, linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()
#fig.show()
