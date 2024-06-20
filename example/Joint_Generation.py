import random
import numpy as np
import numpy as np
# from matplotlib import pyplot as plt
import random
# from Conditional_Generation import *
# from utils import *
# from Joint_Generation import *
import math
import scipy
import scipy.linalg
import scipy.stats
from sklearn.linear_model import LinearRegression, Lasso

def generate_theta(nonzero,p):
    indx = random.sample(list(np.arange(0,p)),nonzero)
    theta = np.zeros((p,1))
    theta[indx] = math.sqrt(nonzero)/nonzero #make norm equal to 1
    return theta, indx


def generate_Z_from_YC_lm(YC,rho2):
    Ztemp = np.random.normal(loc=0,scale=1,size=YC.shape[0])
    model = LinearRegression()
    model.fit(YC.reshape(-1,1),Ztemp.reshape(-1,1))
    fitted = model.predict(YC.reshape(-1,1))
    resd = Ztemp-fitted.flatten()
    Z = rho2 * np.std(resd) * YC + resd*np.std(YC)*np.sqrt(1-rho2**2)
    Z = (Z - np.mean(Z))/np.std(Z)
    #print("Generated correlation z and YC is: ",np.corrcoef(Z,YC.flatten())[0,1])
    #print("mean of z is:",np.mean(Z))
    #print("variance of z is:",np.var(Z))
    return Z


def data_generation(N,rho1,rho2,p,D1,D2,mux,muy,C_vec,C_true,type,verbose):
  theta,indx = generate_theta(nonzero=5,p=p)
  
  if type == "identity":
    covx = np.eye(p)
    covy = np.eye(D1*D2)

  if type == "toeplitz":
    covx = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            covx[i,j] = 0.9**np.abs(i-j)
    covy = np.zeros((D1*D2,D1*D2))
    for i in range(D1*D2):
        for j in range(D1*D2):
            covy[i,j] = 0.9**np.abs(i-j)

  #Normalize the covxy
  covxy = rho1 * covx.dot(theta).dot(C_vec.T).dot(covy) / (np.sqrt(theta.T.dot(covx.dot(theta)))*np.sqrt(C_vec.T.dot(covy).dot(C_vec)))
  
  #Generate X and Y from a joint multivariate normal
  Sig = np.zeros((p+D1*D2)**2).reshape(p+D1*D2, p+D1*D2)
  Sig[0:p,0:p] = covx
  Sig[0:p,(p):(p+D1*D2)] = covxy
  Sig[(p):(p+D1*D2),0:p] = covxy.T
  Sig[(p):(p+D1*D2),(p):(p+D1*D2)]= covy
  
  joint = np.random.multivariate_normal(size=N,mean=np.concatenate([mux,muy]),cov=Sig)
  
  #Separate X and Y
  X = joint[:,0:p]
  Y = joint[:,p:(p+D1*D2)]
  Yarray = Y.reshape(N,D1,D2) #Different from R, slicing index is at the first position
  
  Xcor = X.dot(theta).flatten()
  
  #Check cannonical correlation
  Ytruecor = np.zeros(N)
  for s in range(N):
    Ytruecor[s] = np.sum(np.multiply(Yarray[s,:,:],C_true))
  z = generate_Z_from_YC_lm(YC=Ytruecor,rho2=rho2)
  #z = generate_Z_from_YC_orth(YC=Ytruecor,rho2=rho2)
  if verbose == True:
    print("The norm of theta is: ", theta.T.dot(theta))
    print("Variance of Xtheta is: ",theta.T.dot(covx).dot(theta))
    print("Variance of YC is: ",C_vec.T.dot(covy).dot(C_vec))
    print("cor(Xtheta,YC) is: ",np.corrcoef(Xcor,Ytruecor)[0,1])
    print("cor(z,Xtheta) is:",np.corrcoef((z,Xcor))[0,1])
    #Check induced correlation between z and YC
    print("cor(z,YC) is: ",np.corrcoef(z,Ytruecor)[0,1])
    print("The sum of 3 correlation is: ",np.corrcoef(Xcor,Ytruecor)[0,1]+np.corrcoef(Xcor,z)[1,0]+np.corrcoef(z,Ytruecor)[0,1])
  return X,Y,Yarray,theta,z,indx
  