# -*- coding: utf-8 -*-

from CLE import lce_inference,lce_estimate_x
from CLS import cls_inference,cls_estimate_x,f_cls_biascorr
from generation import generate
import numpy as np
import scipy 
import random
from scipy import stats
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from random import randint, sample
import networkx as nx
import time
import warnings
warnings.filterwarnings("ignore")


np.random.seed(123)
R=100
Ni=[500,1000]
Distri_e=['N']
Distri_ep=['N']
type_w='PowerLaw'
#type_w='Dyad'
#type_w='SBM'
p=2
p1=1
p2=1
beta = np.array([0.3,0.3]).reshape((p,1))
rho = 0.2
tbeta=beta
tbeta=np.append(tbeta,[rho])
lambdax=0.2
lambda2=0.2



for j2 in range(0,len(Distri_e)):
    distri_ep=Distri_ep[j2]
    for j1 in range(0,len(Distri_ep)):
        distri_e=Distri_e[j1]
        for i in range(0,len(Ni)):
            N=Ni[i]
            CP=np.zeros((p+1,1))
            guji_rho,guji_beta1,guji_beta2,guji_beta3=np.zeros((R,1)),np.zeros((R,1)),np.zeros((R,1)),np.zeros((R,1))
            dense=np.zeros((R,1))
            t,SE,SE1=0,0,0
            for r in range(0,R):

                X,Y1,W,mu_ep=generate(N,p,beta,rho,type_w,distri_e,distri_ep,lambda2)
                if distri_ep == 'N':
                    xe=np.random.normal(0,lambdax**0.5,N*p2).reshape(N,p2)
                    mu_ex=3*lambdax*lambdax
                elif distri_ep == 'N':
                    prng = np.random.RandomState(12345678)
                    xe=prng.standard_t(6, size=N*p2).reshape(N,p2)/np.sqrt(3)
                    mu_ex=6*1.5*1.5 
                X[:,p1:p]=X[:,p1:p]+xe
                X1=X[:,0:p1]
                X2=X[:,p1:p]
                
                #estimation
                starttime = time.time()
                #lce
                #hrbeta,hsig,k=lce_estimate_x(X,Y1,W,p1,lambda2,lambdax)
                hrbeta,hsig,D,dD,ddD=cls_estimate_x(X,Y1,W,p1,lambda2,lambdax)
                
                #SE=lce_inference(X,Y1,hrbeta,hsig,lambda2,mu_ep,W,p1,lambdax,mu_ex)
                SE=cls_inference(X,Y1,hrbeta,hsig,lambda2,mu_ep,W,D,dD,ddD,p1,lambdax,mu_ex)
                
                
                endtime = time.time()
                t=t+endtime -starttime 
                #print(t)
                SE1=SE1+SE
                CI1=hrbeta-1.96*SE[0:p+1]
                CI2=hrbeta+1.96*SE[0:p+1]
                for i in range(p+1):
                    if tbeta[i]>=CI1[i] and tbeta[i]<=CI2[i]:
                        CP[i]+=1
                   
                guji_beta1[r,]=hrbeta[0,]
                guji_beta2[r,]=hrbeta[1,]
                guji_rho[r]=hrbeta[p]

                dense[r]=len(np.nonzero(W.toarray())[0])/N/(N-1)
                print('\r当前进度：{:^3.0f}%'.format(((r + 1) / R * 100)), end='')

            CP=CP/R
            SE1=SE1/R
            bias1=np.mean(guji_beta1)-tbeta[0]
            bias2=np.mean(guji_beta2)-tbeta[1]
            biasr=np.mean(guji_rho)-tbeta[p]
            
            se1=np.std(guji_beta1)
            se2=np.std(guji_beta2)
            ser=np.std(guji_rho)
            
            MSE1=(1.0/R*sum((guji_beta1-tbeta[0])**2))
            MSE2=(1.0/R*sum((guji_beta2-tbeta[1])**2))
            MSEr=(1.0/R*sum((guji_rho-tbeta[p])**2))
            print("\n")
            print('noise(E,epsilon):',distri_e,distri_ep)
            print('N=',N)
            print('type:',type_w)
            print('CP:',CP)
            print('MSE1:',MSE1)
            print('MSE2:',MSE2)
            print('MSEr:',MSEr)
            print('bias_beta1:',bias1)
            print('bias_beta2:',bias2)
            print('biasr:',biasr)
            print('se_beta1:',se1)
            print('se_beta2:',se2)
            print('ser:',ser)
            print('SE1:',SE1)
            print('estimate time',t/R)
            print('density:',np.mean(dense))
 