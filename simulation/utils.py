#Define some functions outside loop
import random
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import random
# from Conditional_Generation import *
# from utils import *
# from Joint_Generation import *
import math
import scipy
import scipy.linalg
import scipy.stats
import k3d
from sklearn.linear_model import LinearRegression, Lasso

def vis_tensor(beta):
    volume = k3d.volume(beta)
    plot = k3d.plot(camera_auto_fit=True)
    plot += volume
    plot.lighting = 1
    plot.display()
    
def soft(n,tau,z):   #In the theory we have n, but here we do not need since we can adjust tau
    if z < -tau/2 :
        s = z+tau/2
    if z > tau/2 :
        s = z-tau/2
    if abs(z) <= tau/2 :
        s = 0
    return s

def hard(tau,z):
    if np.abs(z) < (tau):
        s = 0
    if np.abs(z) >= (tau):
        s = z
    return s

def cut_T_pro(T, Adim, img_type = "gray"):
    ''' R operator for matrix/tensor: (p1*d1, p2*d2, p3*d3) to (p1*p2*p3, d1*d2*d3) '''
    if len(Adim) == 2:
        N1, N2 = list(T.shape)[0], list(T.shape)[1]
        p1, p2 = Adim
        assert N1 % p1 == 0 and N2 % p2 == 0, "Dimension wrong"
        d1, d2 = N1 // p1, N2 // p2
        if img_type == "rgb":
            strides = T.itemsize * np.array([p2*d2*d1*3, d2*3, p2*d2*3, 3,1])
            T_blocked = np.lib.stride_tricks.as_strided(T, shape=(p1, p2, d1, d2,3), strides=strides)
        else:
            strides = T.itemsize * np.array([p2*d2*d1, d2, p2*d2, 1])
            T_blocked = np.lib.stride_tricks.as_strided(T, shape=(p1, p2, d1, d2), strides=strides)
    else:
        N1, N2, N3 = T.shape
        p1, p2, p3 = Adim
        assert N1 % p1 == 0 and N2 % p2 == 0 and N3 % p3 == 0, "Dimension wrong"
        d1, d2, d3 = N1 // p1, N2 // p2, N3 // p3
        strides = T.itemsize * np.array([N2 * N3 * d1, N3 * d2, d3, N2 * N3, N3, 1])  # 大层，大行，大列，小层，小行，小列
        T_blocked = np.lib.stride_tricks.as_strided(T, shape=(p1, p2, p3, d1, d2, d3), strides=strides)
    return T_blocked


def Cov_ystar(N,Yarray,b,R,p1,p2,d1,d2):
  Y_star = np.zeros(N*p1*p2*R).reshape(N,p1*p2*R)
  for r in range(R):
    for i in range(N):
        test = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2]).reshape(p1*p2,d1*d2) # Create Y_tilde
        Y_star[i,p1*p2*r:p1*p2*(r+1)] = test.dot(b[d1*d2*r:d1*d2*(r+1)])
  S_ys = np.cov(Y_star,rowvar=False)
  return S_ys

def Cov_yprime(N,Yarray,a,R,p1,p2,d1,d2):
  Y_prime = np.zeros(N*d1*d2*R).reshape(N,d1*d2*R)
  for r in range(R):
    for i in range(N):
        test = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2]).reshape(p1*p2,d1*d2) # Create Y_tilde
        Y_prime[i,d1*d2*r:d1*d2*(r+1)] = test.T.dot(a[p1*p2*r:p1*p2*(r+1)])
  S_yp = np.cov(Y_prime,rowvar=False)
  return S_yp

def Cov_ystar_tensor(N,Yarray,b,R,p1,p2,p3,d1,d2,d3):
  Y_star = np.zeros(N*p1*p2*p3*R).reshape(N,p1*p2*p3*R)
  for r in range(R):
    for i in range(N):
        test = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2,p3]).reshape(p1*p2*p3,d1*d2*d3) # Create Y_tilde
        Y_star[i,p1*p2*p3*r:p1*p2*p3*(r+1)] = test.dot(b[d1*d2*d3*r:d1*d2*d3*(r+1)])
  S_ys = np.cov(Y_star,rowvar=False)
  return S_ys

def Cov_yprime_tensor(N,Yarray,a,R,p1,p2,p3,d1,d2,d3):
  Y_prime = np.zeros(N*d1*d2*d3*R).reshape(N,d1*d2*d3*R)
  for r in range(R):
    for i in range(N):
        test = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2,p3]).reshape(p1*p2*p3,d1*d2*d3) # Create Y_tilde
        Y_prime[i,d1*d2*d3*r:d1*d2*d3*(r+1)] = test.T.dot(a[p1*p2*p3*r:p1*p2*p3*(r+1)])
  S_yp = np.cov(Y_prime,rowvar=False)
  return S_yp
