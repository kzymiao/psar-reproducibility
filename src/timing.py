# -*- coding: utf-8 -*-

from .CLE import lce_inference,lce_estimate_x
from .CLS import cls_inference,cls_estimate_x
from .generation import generate
import numpy as np
from scipy import stats
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from random import randint, sample
import networkx as nx
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


np.random.seed(123)
R=20
Ni=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
distri_e='N'
distri_ep='N'
type_w='Dyad'
p=2
p1=1
p2=1
beta = np.array([0.3,0.3]).reshape((p,1))
rho = 0.2
tbeta=np.append(beta,[rho])
t_cle,t_cls=[],[]
lambda2=0.5
lambdax=0.2
for i in range(0,len(Ni)):
    N=Ni[i]
    CP=np.zeros((p+1,1))
    guji_rho,guji_beta1,guji_beta2,guji_beta3=np.zeros((R,1)),np.zeros((R,1)),np.zeros((R,1)),np.zeros((R,1))
    dense=np.zeros((R,1))
    t1,t2,t3=0,0,0
    for r in range(0,R):
    
    
        X,Y1,W,mu_ep=generate(N,p,beta,rho,type_w,distri_e,distri_ep,lambda2)
        
        xe=np.random.normal(0,lambdax**0.5,N*p2).reshape(N,p2)
        mu_ex=3*lambdax*lambdax
        
        X[:,p1:p]=X[:,p1:p]+xe
        X1=X[:,0:p1]
        X2=X[:,p1:p]
        
    
        #estimation
        
        starttime = time.time()
        hrbeta,hsig,k=lce_estimate_x(X,Y1,W,p1,lambda2,lambdax)
        endtime = time.time()
        t2=t2+endtime -starttime
        
        starttime = time.time()
        hrbeta,hsig,D,dD,ddD=cls_estimate_x(X,Y1,W,p1,lambda2,lambdax)
        endtime = time.time()
        t3=t3+endtime -starttime 
    
        #print('\r Current progress: {:^3.0f}%'.format(((r + 1) / R * 100)), end='')
                
    print('N=',N)
    print('CLE:',t2/R)
    print('CLS:',t3/R)
    print("\n")
    t_cle.append(t2/R)
    t_cls.append(t3/R)

X=Ni


plt.figure(figsize=(7,9))
plt.plot(X, t_cle,linewidth=8.0,label="time of CLE method", linestyle="-")
plt.plot(X, t_cls,linewidth=8.0,label="time of CLS method", linestyle=":")
plt.legend(frameon=False,loc="upper left",fontsize='xx-large') #分别为图例无边框、图例放在右上角、图例大小
#plt.legend()
#plt.title(" ")
plt.xticks(fontsize=18) #x轴刻度字体大小
plt.yticks(fontsize=18) #y轴刻度字体大小
plt.xlabel("N",fontsize=18)
plt.ylabel("CPU time",fontsize=18)
plt.show()
