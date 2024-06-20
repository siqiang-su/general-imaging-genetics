#Define some functions outside loop
import random
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
import random
# from Conditional_Generation import *
from utils import *
from Joint_Generation import *
import math
import scipy
import scipy.linalg
import scipy.stats
import k3d
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import KFold


############### Alternating Minimization ######################
def cca_alt_minimization(N,X,Yarray,z,lam1,lam2,R,tau_x,tau_y,p1,p2,d1,d2,p,niter,init):
  # Initial values for parameters
  if (init == 'one'):
    a = np.ones(p1*p2*R) 
    b = np.ones(d1*d2*R) 

  if (init == 'Gaussian'):
    a = (np.random.normal(loc=1,scale=0.1,size=p1*p2*R))
    b = (np.random.normal(loc=1,scale=0.1,size=d1*d2*R))
    
  if (init == 'uniform'):
    a = np.abs(np.random.uniform(low=0.5,high=1,size=p1*p2*R))
    b = np.abs(np.random.uniform(low=0.5,high=1,size=d1*d2*R))
    
    

  S_x = np.cov(X,rowvar=False) + tau_x * np.eye(X.shape[1])
  S_x_sqrt = scipy.linalg.sqrtm(S_x).real
  S_x_sqrt_inv = np.linalg.inv(S_x_sqrt)

  #### Iteration loop ####
  for k in range(niter):

    #Fix a and b, solve theta
    g = np.zeros(p)
    for i in range(N):
      test1 = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2]).reshape(p1*p2,d1*d2)
      ayb_temp = 0
      for r in range(R):
        ayb_temp += a[p1*p2*r:p1*p2*(r+1)].T.dot(test1).dot(b[d1*d2*r:d1*d2*(r+1)])
      g = np.column_stack((g,(ayb_temp-z[i])*(S_x_sqrt_inv.dot(X[i,:])/2)))

    g = g[:,1:]
    gbar = np.apply_along_axis(np.mean,1,g)

    #Coordinate Descent for Lasso to solve "theta"
    model1 = Lasso(alpha=lam2)
    model1.fit(X=S_x_sqrt,y=gbar)
    thetahat = model1.coef_
    thetahat = thetahat/math.sqrt(sum((S_x_sqrt.dot(thetahat))**2)) #Normalize theta such that theta norm is 1

    #Fix theta and b, solve a
    c = np.zeros(p1*p2*R)
    S_ys = Cov_ystar(N,Yarray,b,R,p1,p2,d1,d2) + tau_y * np.eye(p1*p2*R) # Compute sample covariance matrix for Y*
    S_ys_sqrt = scipy.linalg.sqrtm(S_ys).real
    S_ys_sqrt_inv = np.linalg.inv(S_ys_sqrt)

    for i in range(N):
      test = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2]).reshape(p1*p2,d1*d2) # Create Y_tilde
      yb_temp = np.zeros(p1*p2*R)
      for r in range(R):
        yb_temp[p1*p2*r:p1*p2*(r+1)] = test.dot(b[d1*d2*r:d1*d2*(r+1)])
      c = np.column_stack((c,(thetahat.T.dot(X[i,])+z[i])*(S_ys_sqrt_inv.dot(yb_temp)/2)))
    
    c = c[:,1:]
    cbar = np.apply_along_axis(np.mean,1,c) #Here we take row mean
    
    # Coordinate Descent for LASSO regression
    model0 = Lasso(alpha=lam1)
    model0.fit(X=S_ys_sqrt,y=cbar)
    a = model0.coef_
    a = a/math.sqrt(sum((S_ys_sqrt.dot(a))**2)) #Normalize a such that alpha norm is 1
    ### Force matrix a into an orthogonal matrix
    a = a.reshape(p1*p2,R)
    a_sqrt = scipy.linalg.sqrtm(a.T.dot(a))
    a_sqrt_inv = np.linalg.inv(a_sqrt+ 0.01*np.eye(R))
    a = a.dot(a_sqrt_inv)
    a = a.flatten()
    

    # Once we updated a, we can compute the covariance matrix Y':
    S_yp = Cov_yprime(N,Yarray,a,R,p1,p2,d1,d2) + tau_y * np.eye(d1*d2*R)
    S_yp_sqrt = scipy.linalg.sqrtm(S_yp).real
    S_yp_sqrt_inv = np.linalg.inv(S_yp_sqrt)

    #Fix a and theta, solve b
    d = np.zeros(d1*d2*R)
    for i in range(N):
      temp = (thetahat.T.dot(X[i,:])+z[i])/2
      test2 = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2]).reshape(p1*p2,d1*d2)
      yta_temp = np.zeros(d1*d2*R)
      for r in range(R):
          yta_temp[d1*d2*r:d1*d2*(r+1)] = test2.T.dot(a[p1*p2*r:p1*p2*(r+1)])
      d = np.column_stack((d,S_yp_sqrt_inv.dot(yta_temp)*temp))

    d = d[:,1:]
    b = S_yp_sqrt_inv.dot((np.apply_along_axis(np.mean,1,d)))
    b = b/math.sqrt(np.sum((S_yp_sqrt.dot(b))**2)) #Normalize b such that delta norm is 1

  return a,b,thetahat
    

############ This function computes the TPR, FPR and MSE measures ################
def cca_sim_evaluation(N,X,Yarray,z,a,b,R,theta,C_true,thetahat,indx,p1,p2,d1,d2,D1,D2,show):
  # Evaluate thetahat
  values = [thetahat[i] for i, x in enumerate(thetahat) if i not in indx]
  TPR_theta = np.size(np.nonzero(thetahat[indx]))/5 #Among 5 true non-zero, how many is predicted non-zero
  FPR_theta = np.size(np.nonzero(values))/(95)
  theta_err = np.sum((theta.flatten()-thetahat)**2)
  print("TPR of theta is: ",TPR_theta)
  print("FPR of theta is: ",FPR_theta)
  maxind = np.argpartition((thetahat).flatten(),-5)[-5:]

  # Evaluate C
  C_hat = np.zeros((D1,D2))
  for r in range(R):
      temp = np.kron(a[p1*p2*r:p1*p2*(r+1)].reshape(p1,p2),b[d1*d2*r:d1*d2*(r+1)].reshape(d1,d2))
      C_hat = C_hat + temp
  TPR_C = np.count_nonzero(C_hat[C_true>0])/np.count_nonzero(C_true)
  FPR_C = np.count_nonzero(C_hat[C_true==0])/(D1*D2 - np.count_nonzero(C_true))
  print("TPR of C is: ",TPR_C)
  print("FPR of C is: ",FPR_C)
  C_err = np.sum((C_hat - C_true)**2)
  #Check the recovery
  if show == True:
    r = d1
    plt.xticks(np.arange(0,D1,r),color = 'w') # ,color = 'w'
    plt.yticks(np.arange(0,D2,r),color = 'w')
    plt.grid(linestyle = '-.',linewidth = 0.5,which = "both")
    plt.imshow(C_hat,cmap="gray")
    plt.show() 
  
  # Evaluate correlation
  Ycor = np.zeros(N)
  for i in range(N):
    Ycor[i] = np.sum(np.multiply(Yarray[i,:,:],C_hat))

  canoncor = np.corrcoef(X.dot(thetahat),Ycor)[0,1]

  #Check canonical correlation of z and Xbeta
  
  zXcor = np.corrcoef(X.dot(thetahat),z)[0,1]

  #Check canonical correlation of z and YC
  
  zYcor = np.corrcoef(Ycor,z)[0,1]

  #Check the sum of correlations of 3 blocks
  
  sumcor = canoncor + (zXcor) + (zYcor)

  return TPR_theta, FPR_theta, theta_err, TPR_C, FPR_C, C_err, C_hat, canoncor, zXcor, zYcor, sumcor

def C_hat_vis(d1,D1,D2,C_hat):
  r = d1
  plt.xticks(np.arange(0,D1,r),color = 'w') # ,color = 'w'
  plt.yticks(np.arange(0,D2,r),color = 'w')
  plt.grid(linestyle = '-.',linewidth = 0.5,which = "both")
  plt.imshow(C_hat,cmap="gray")
  plt.show()

def cca_cross_val(N,fold,X,Yarray,z,lam1_grid,lam2_grid,p1,p2,d1,d2,p,niter):
  kf = KFold(n_splits=fold)
  sumcor_grid = np.zeros((len(lam1_grid),len(lam2_grid)))
  for i in range(len(lam1_grid)):
      for j in range(len(lam2_grid)):
        sumcor = []
        for train_index, test_index in kf.split(X):
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = Yarray[train_index], Yarray[test_index]
          z_train, z_test = z[train_index], z[test_index]

          #Now train the model
          a,b,thetahat = cca_alt_minimization(N=len(z_train),X=X_train,Yarray=y_train,z=z_train,lam1=lam1_grid[i],lam2=lam2_grid[j],tau_x=10,tau_y=0.01,p1=p1,p2=p2,d1=d1,d2=d2,p=p,niter=niter)
          A = a.reshape(p1,p2)
          B = b.reshape(d1,d2)
          C_hat = np.kron(A,B)

          # Compute the summation of correlations of 3 blocks in the test set

          Ycor = np.zeros(len(test_index))
          for k in range(len(test_index)):
            Ycor[k] = np.sum(np.multiply(y_test[k,:,:],C_hat))

          canoncor = np.corrcoef(X_test.dot(thetahat),Ycor)[0,1]

  #Check canonical correlation of z and Xbeta
  
          Zcor = np.corrcoef(X_test.dot(thetahat),z_test)[0,1]

  #Check canonical correlation of z and YC
  
          zycor = np.corrcoef(Ycor,z_test)[0,1]

  #Check the sum of correlations of 3 blocks
  
          sumcor.append(canoncor + (Zcor) + (zycor))

        sumcor_grid[i,j] = np.mean(sumcor)
        print("The sumcor of combination of", lam1_grid[i], "and", lam2_grid[j], "is: ", np.mean(sumcor))
  # Return the index of array that has max sumcor
  max_index = np.argmax(sumcor_grid)
  row_index = max_index // sumcor_grid.shape[1]
  col_index = max_index % sumcor_grid.shape[1]
  opt_lam1 = lam1_grid[row_index]
  opt_lam2 = lam2_grid[col_index]
  print("The best combination of hyperparameters are", lam1_grid[row_index], "and", lam2_grid[col_index], "with sumcor: ", sumcor_grid[row_index,col_index])
  return opt_lam1, opt_lam2


########### BIC tuning for hyper-parameter selection ################
def cca_BIC_tuning(N,X,Yarray,z,lam1_grid,lam2_grid,R_grid,p1,p2,d1,d2,p,niter):
  for i in range(len(lam1_grid)):
     for j in range(len(lam2_grid)):
        for k in range(len(R_grid)):
            a,b,thetahat = cca_alt_minimization(N=len(z),X=X,Yarray=Yarray,z=z,lam1=lam1_grid[i],lam2=lam2_grid[j],R=R_grid[k],tau_x=10,tau_y=0.1,p1=p1,p2=p2,d1=d1,d2=d2,p=p,niter=niter,init='uniform')
            a_l0 = np.count_nonzero(a)
            theta_l0 = np.count_nonzero(thetahat)
            R = R_grid[k]
            #Evaluate BIC value
            R_sum = np.zeros(N)
            obj = 0
            for s in range(N):
              Tilde_Y = cut_T_pro(Yarray[s,:,:],Adim=[p1,p2]).reshape(p1*p2,d1*d2)
              for r in range(R):
              # Summation over R terms
                R_sum[s] += a[p1*p2*r:p1*p2*(r+1)].T.dot(Tilde_Y).dot(b[d1*d2*r:d1*d2*(r+1)])
              obj += (R_sum[s] * (thetahat.T.dot(X[s,:])+z[s]) - (thetahat.T.dot(X[s,:])*z[s])) / (-N)
            BIC = (obj) + np.log(N)/N*a_l0 + np.log(N)/N*theta_l0
            print("The BIC value for combination ",lam1_grid[i],lam2_grid[j],R_grid[k],"is: ", BIC)

  return BIC


def cca_alt_minimization_tensor(N,X,Yarray,z,lam1,lam2,R,tau_x,tau_y,p1,p2,p3,d1,d2,d3,p,niter,init):
  # Initial values for parameters
  if (init == 'one'):
    a = np.ones(p1*p2*p3*R) 
    b = np.ones(d1*d2*d3*R) 

  if (init == 'Gaussian'):
    a = (np.random.normal(loc=1,scale=0.1,size=p1*p2*p3*R))
    b = (np.random.normal(loc=1,scale=0.1,size=d1*d2*d3*R))
    
  if (init == 'uniform'):
    a = np.abs(np.random.uniform(low=0.5,high=1,size=p1*p2*p3*R))
    b = np.abs(np.random.uniform(low=0.5,high=1,size=d1*d2*d3*R))

  S_x = np.cov(X,rowvar=False) + tau_x * np.eye(X.shape[1])
  S_x_sqrt = scipy.linalg.sqrtm(S_x).real
  S_x_sqrt_inv = np.linalg.inv(S_x_sqrt)

  for k in range(niter):
    #Fix a and b, solve theta
    g = np.zeros(p)
    for i in range(N):
      test1 = cut_T_pro(Yarray[i,:,:,:],Adim=[p1,p2,p3]).reshape(p1*p2*p3,d1*d2*d3)
      ayb_temp = 0
      for r in range(R):
        ayb_temp += a[p1*p2*p3*r:p1*p2*p3*(r+1)].T.dot(test1).dot(b[d1*d2*d3*r:d1*d2*d3*(r+1)])
      g = np.column_stack((g,(ayb_temp-z[i])*(S_x_sqrt_inv.dot(X[i,:])/2)))

    g = g[:,1:]
    gbar = np.apply_along_axis(np.mean,1,g)

    #Coordinate Descent for Lasso to solve "theta"
    model1 = Lasso(alpha=lam2)
    model1.fit(X=S_x_sqrt,y=gbar)
    thetahat = model1.coef_
    thetahat = thetahat/math.sqrt(sum((S_x_sqrt.dot(thetahat))**2)) #Normalize theta such that beta norm is 1..
    

    # Fix theta and b, solve a
    c = np.zeros(p1*p2*p3*R)
    S_ys = Cov_ystar_tensor(N,Yarray,b,R,p1,p2,p3,d1,d2,d3) + tau_y * np.eye(p1*p2*p3*R) # Compute sample covariance matrix for Y*
    S_ys_sqrt = scipy.linalg.sqrtm(S_ys).real
    S_ys_sqrt_inv = np.linalg.inv(S_ys_sqrt)

    for i in range(N):
      test = cut_T_pro(Yarray[i,:,:,:],Adim=[p1,p2,p3]).reshape(p1*p2*p3,d1*d2*d3) # Create Y_tilde
      yb_temp = np.zeros(p1*p2*p3*R)
      for r in range(R):
        yb_temp[p1*p2*p3*r:p1*p2*p3*(r+1)] = test.dot(b[d1*d2*d3*r:d1*d2*d3*(r+1)])
      c = np.column_stack((c,(thetahat.T.dot(X[i,])+z[i])*(S_ys_sqrt_inv.dot(yb_temp)/2))) # Create constants c
    
    c = c[:,1:]
    cbar = np.apply_along_axis(np.mean,1,c) #Here we take row mean
    
    # Coordinate Descent for LASSO regression with non-orthonormal design
    model0 = Lasso(alpha=lam1)
    model0.fit(X=S_ys_sqrt,y=cbar)
    a = model0.coef_
    a = a/math.sqrt(sum((S_ys_sqrt.dot(a))**2)) #Normalize a such that alpha norm is 1..
    ### Force matrix a into an orthogonal matrix
    a = a.reshape(p1*p2*p3,R)
    a_sqrt = scipy.linalg.sqrtm(a.T.dot(a))
    a_sqrt_inv = np.linalg.inv(a_sqrt+ 0.01*np.eye(R))
    a = a.dot(a_sqrt_inv)
    a = a.flatten()

    # Once we updated a, we can compute the covariance matrix Y':
    S_yp = Cov_yprime_tensor(N,Yarray,a,R,p1,p2,p3,d1,d2,d3) + tau_y * np.eye(d1*d2*d3*R)
    S_yp_sqrt = scipy.linalg.sqrtm(S_yp).real
    S_yp_sqrt_inv = np.linalg.inv(S_yp_sqrt)

    #Fix a and beta, solve b
    d = np.zeros(d1*d2*d3*R)
    for i in range(N):
      temp = (thetahat.T.dot(X[i,:])+z[i])/2
      test2 = cut_T_pro(Yarray[i,:,:],Adim=[p1,p2,p3]).reshape(p1*p2*p3,d1*d2*d3)
      yta_temp = np.zeros(d1*d2*d3*R)
      for r in range(R):
          yta_temp[d1*d2*d3*r:d1*d2*d3*(r+1)] = test2.T.dot(a[p1*p2*p3*r:p1*p2*p3*(r+1)])
      d = np.column_stack((d,S_yp_sqrt_inv.dot(yta_temp)*temp))

    d = d[:,1:]
    b = S_yp_sqrt_inv.dot((np.apply_along_axis(np.mean,1,d)))
    b = b/math.sqrt(np.sum((S_yp_sqrt.dot(b))**2)) #Normalize b such that delta norm is 1..

  return a,b,thetahat

def cca_BIC_tuning_tensor(N,X,Yarray,z,lam1_grid,lam2_grid,R_grid,p1,p2,p3,d1,d2,d3,p,niter):
  for i in range(len(lam1_grid)):
     for j in range(len(lam2_grid)):
        for k in range(len(R_grid)):
            a,b,thetahat = cca_alt_minimization_tensor(N=len(z),X=X,Yarray=Yarray,z=z,lam1=lam1_grid[i],lam2=lam2_grid[j],R=R_grid[k],tau_x=0.1,tau_y=0.1,p1=p1,p2=p2,p3=p3,d1=d1,d2=d2,d3=d3,p=p,niter=niter)
            a_l0 = np.count_nonzero(a)
            theta_l0 = np.count_nonzero(thetahat)
            R = R_grid[k]
            #Evaluate BIC value
            R_sum = np.zeros(N)
            obj = 0
            for s in range(N):
              Tilde_Y = cut_T_pro(Yarray[s,:,:,:],Adim=[p1,p2,p3]).reshape(p1*p2*p3,d1*d2*d3)
              for r in range(R):
              # Summation over R terms
                R_sum[s] += a[p1*p2*p3*r:p1*p2*p3*(r+1)].T.dot(Tilde_Y).dot(b[d1*d2*d3*r:d1*d2*d3*(r+1)])
              obj += (R_sum[s] * (thetahat.T.dot(X[s,:])+z[s]) - (thetahat.T.dot(X[s,:])*z[s])) / (-N)
            BIC = (obj) + np.log(N)/N*a_l0 + np.log(N)/N*theta_l0
            print("The BIC value for combination ",lam1_grid[i],lam2_grid[j],R_grid[k],"is: ", BIC)
            print("The number of nonzero value for thetahat ",lam1_grid[i],lam2_grid[j],R_grid[k],"is: ", np.count_nonzero(thetahat))

  return a,b,thetahat,BIC

def permutation_test_C_hat(C_hat,N,D1,D2,D3,X,Y_com,z,lam1,lam2,R,p1,p2,p3,d1,d2,d3,p):
  C_hat[C_hat !=0] = 1
  #First extract selected regions for all images
  perm_reg = np.zeros((N,D1,D2,D3))
  for i in range(N):
    perm_reg[i,:,:,:] = C_hat * Y_com[i,:,:,:] #elementwise multiplication
  #Permute the selected regions over N samples
  perm_ind = np.arange(1744)
  random.shuffle(perm_ind)
  perm_reg = perm_reg[perm_ind,:,:,:]
  #Create reverse C_hat
  C_rev = C_hat + 1
  C_rev[C_rev != 1] = 0
  #Clear the selected region in the original Y_com images
  Y_perm = np.zeros((N,D1,D2,D3))
  for i in range(N):
      Y_perm[i,:,:,:] = C_rev * Y_com[i,:,:,:]  
  #Overlay the permuted region onto the cleared region
  Y_perm = Y_perm + perm_reg
  #Implement algorithm on the permuted images
  a_perm,b_perm,thetahat_perm = cca_alt_minimization_tensor(N,X=X,Yarray=Y_perm,z=z,lam1=lam1,lam2=lam2,R=R,tau_x=10,tau_y=0.1,p1=p1,p2=p2,p3=p3,d1=d1,d2=d2,d3=d3,p=p,niter=10) 

  return perm_ind,a_perm,b_perm,thetahat_perm   
















  





 

 


  
    
    



  
