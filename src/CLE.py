# -*- coding: utf-8 -*-

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
def f_lambda_ture_sig(hbeta1,hbeta2,hrho,Y,X,W,lambda2,lambdax):  # 计算一阶导(Y)
    N=len(Y)
    p1=len(hbeta1)
    p2=len(hbeta2)
    p=p1+p2
    X1=X[:,0:p1]
    X2=X[:,p1:p]
    hS= np.eye(N)-hrho*W.toarray()
    hS1=np.linalg.inv(hS)
    V=hS.dot(Y)-(X1.dot(hbeta1)+X2.dot(hbeta2)).reshape(N,1)
    SS=hS.dot(hS.T)
    hsig=(V.T.dot(V)-lambdax*hbeta2.T.dot(hbeta2)*N-lambda2*(SS).trace())/N#-lambdax*hbeta2.T@hbeta2
    hsig=hsig[0,0]
    s1=1.0/hsig
    
    omega=hsig*np.eye(N)+lambda2*SS
    omega1=np.linalg.inv(omega)
    omega2=omega1.dot(omega1)
    
    f_r=lambda2*((omega1.dot(W.toarray()).dot(hS.T)).trace())-((W.toarray().dot(hS1)).trace())+(V.T.dot(omega1).dot(W.toarray()).dot(Y))-lambda2*(V.T.dot(omega1).dot(W.toarray()).dot(hS.T).dot(omega1).dot(V))+lambdax*hbeta2.T.dot(hbeta2)*(omega2.dot(W.toarray()).dot(hS.T)).trace()
    f_b1=X1.T.dot(omega1).dot(V)
    f_b2=X2.T.dot(omega1).dot(V)+lambdax*(omega1).trace()*hbeta2
    
    
    f=np.vstack((f_b1,f_b2,f_r))

    sim=W.dot(hS.T)+hS.dot((W.toarray()).T)
    
    g_rr=lambda2**2*(omega1.dot(sim).dot(omega1).dot(W.toarray()).dot(hS.T)).trace()-lambda2*(omega1.dot(W.toarray()).dot((W.toarray()).T)).trace()-(W.toarray().dot(hS1).dot(W.toarray()).dot(hS1)).trace()-(Y.T.dot((W.toarray()).T).dot(omega1).dot(W.toarray()).dot(Y))+2*lambda2*(V.T.dot(omega1).dot(sim).dot(omega1).dot(W.toarray()).dot(Y))-lambda2**2*(V.T.dot(omega1).dot(sim).dot(omega1).dot(sim).dot(omega1).dot(V))+lambda2*(V.T.dot(omega1).dot(W.toarray()).dot((W.toarray()).T).dot(omega1).dot(V))
    g_rr=g_rr+lambda2**2*lambdax*(hbeta2.T.dot(hbeta2))[0,0]*(omega2.dot(sim).dot(omega1).dot(sim)).trace()-lambdax*lambda2*(hbeta2.T.dot(hbeta2))[0,0]*(omega2.dot(W.toarray()).dot(W.toarray().T)).trace()
    g_b1b1=-X1.T.dot(omega1).dot(X1)
    g_b2b2=-X2.T.dot(omega1).dot(X2)+lambdax*(omega1).trace()*np.eye(p2)
    
    g_b1r=lambda2*(X1.T.dot(omega1).dot(sim).dot(omega1).dot(V))-(X1.T.dot(omega1).dot(W.toarray()).dot(Y))
    g_b2r=lambdax*lambda2*(omega2.dot(sim)).trace()*hbeta2+lambda2*X2.T.dot(omega1).dot(sim).dot(omega1).dot(V)-X2.T.dot(omega1).dot(W.toarray()).dot(Y)#-2*lambdax*(V.T.dot(omega2).dot(W.toarray()).dot(Y))[0,0]*hbeta2+2*lambdax*lambda2*(V.T.dot(omega1).dot(sim).dot(omega2).dot(V))[0,0]*hbeta2
    g_b1b2=-X1.T.dot(omega1).dot(X2)
    
    gt=np.vstack((g_b1b1, np.transpose(g_b1b2), np.transpose(g_b1r)))
    gtt=np.vstack((g_b1b2, g_b2b2, np.transpose(g_b2r)))
    g=np.hstack((gt,gtt,np.vstack((g_b1r, g_b2r, g_rr))))
    return f,g,hsig

def lce_estimate_x(X,Y1,W,p1,lambda2,lambdax):
    N=len(Y1)
    p=len(X[1])
    p2=p-p1
    hrbeta=(0.01*np.ones(p+1)).reshape((p+1,1))
    K=1000
    k=0
    e=10**(-8)
    f,g,hsig=f_lambda_ture_sig(hrbeta[0:p1,],hrbeta[p1:p,],hrbeta[p,],Y1,X,W,lambda2,lambdax)
    while np.linalg.norm(f)>e and k<=K :
        hrbeta-=np.dot(np.linalg.inv(g),f)
        k+=1
        f,g,hsig=f_lambda_ture_sig(hrbeta[0:p1,],hrbeta[p1:p,],hrbeta[p,],Y1,X,W,lambda2,lambdax)
    return hrbeta,hsig,k

def lce_inference(X,Y1,hrbeta,hsig,lambda2,mu_ep,W,p1,lambdax,mu_ex):
    """
    Obtain estimated standard error SE based on Theorem 4.
    :Y1: the observed response
    :hrbeta: the estimate vector of (rho, beta)
    :hsig: the estimate of sigma^2
    :lambda2: the variance of artificially created noise
    :mu_ep: the fourth moment of epsilon
    :W: the network weighting matrix
    :return: the estimated standard error SE.
    """
    N=len(Y1)
    p=len(X[1])
    p2=p-p1
    X1=X[:,0:p1]
    X2=X[:,p1:p]
    beta1=hrbeta[0:p1]
    beta2=hrbeta[p1:p]
    bb=(beta2.T@beta2)[0][0]
    
    W=W.toarray()
    S=np.eye(N)-hrbeta[p,][0]*W
    S1=np.linalg.inv(S)
    G=W.dot(S1)
    Gs=G+np.transpose(G)
    Omega=hsig*np.eye(N)+lambda2*S@S.T 
    Omega1=np.linalg.inv(Omega)
    WW=W@S.T+S@W.T
    
    V=S.dot(Y1)-(X.dot(hrbeta[0:p,])).reshape(N,1)
    mu_e=np.mean(np.array(V)**4)-(1-hrbeta[p,][0])**4*mu_ep-6*(1-hrbeta[p,][0])**2*hsig*lambda2
    
    mu_ex=3*lambdax**2
    mu_exb=(mu_ex-lambdax**2)*np.sum(beta2**4)+lambdax**2*np.sum(beta2**2)**2
    
    A1=lambda2*Omega1@W@S.T@Omega1-Omega1@G
    A2=S.T@A1@S
    B1=S.T@Omega1@Omega1@S
    B2=S.T@Omega1@W@S.T@Omega1@S
    B3=Omega1@Omega1
    B4=Omega1@W@S.T@Omega1
    B5=Omega1@G
    B6=S.T@Omega1@W
    
    #calculate \Sigma_2 in Theorem 4
    CO1=1.0/N*X.T@Omega1@X
    Sig2_b1b1=CO1[0:p1,0:p1]
    Sig2_b2b2=CO1[p1:p,p1:p]-lambdax*(Omega1).trace()*np.eye(p2)/N
    Sig2_b1b2=CO1[0:p1,p1:p]
    CO1=np.hstack((np.vstack((Sig2_b1b1,Sig2_b1b2.T)),np.vstack((Sig2_b1b2,Sig2_b2b2))))
    
    
    CO2=lambda2**2/N*(Omega1@WW@Omega1@S@W.T).trace()-2.0*lambda2/N*(WW@Omega1@G).trace()+1.0/N*(hrbeta[0:p,]).T@X.T@G.T@Omega1@G@X@(hrbeta[0:p,])+1.0/N*(G@G).trace()+1.0/N*(Omega1@G@Omega@G.T).trace()
    CO2=CO2-lambdax*bb*(G.T@Omega1@G).trace()/N
    
    CO3=1.0/N*X.T@Omega1@G@X@hrbeta[0:p,]
    CO3=CO3+np.vstack((np.zeros((p1,1)),-lambdax*(Omega1@G).trace()*beta2/N))
    
    CO4=1.0/N*(Omega1@G).trace()-lambda2/(2*N)*(Omega1@WW@Omega1).trace()
    CO4=CO4-lambdax*bb*(G.T@B3).trace()/N
    
    CO=np.vstack((CO1,np.transpose(CO3),np.zeros((1,p))))
    Sigm2=np.hstack((CO,np.vstack((CO3,CO2,CO4)),np.vstack((np.zeros((p,1)),CO4,1.0/(2*N)*(Omega1@Omega1).trace()))))
    #SSS=SSS+Sigm2
    
    #calculate \Delta in Theorem 4
    
     
    Delta_rr=(mu_e-3*hsig**2)/N*sum([A1[i][i]**2 for i in range(len(A1))])+(mu_ep-3*lambda2**2)/N*sum([A2[i][i]**2 for i in range(len(A2))])
    Delta_rr=Delta_rr+(4*lambdax*lambda2**2*bb*(Omega1@W@S.T@Omega1@S@W.T@Omega1).trace()+2*lambda2**2*(lambdax*bb)**2*(B4@B4).trace()+lambdax*bb*(B5.T@B5@Omega).trace()+(mu_exb-3*lambdax**2*bb**2)*sum([B4[i][i]**2 for i in range(len(B4))]))/N+lambdax*bb*hrbeta[0:p,].T@X.T@G.T@B3@G@X@hrbeta[0:p,]/N
    Delta_rr=Delta_rr-lambdax**2*bb**2*(B5.T@B5).trace()/N
    
    Delta_b1b1=lambdax*bb/N*X1.T@B3@X1
    Delta_b2b2=lambdax*bb/N*X2.T@B3@X2+(p+1)/N*lambdax**2*(B3).trace()*beta2@beta2.T+lambdax/N*(Omega1).trace()*np.eye(p2)
    Delta_b2b2=Delta_b2b2-lambdax**2*bb*(B3).trace()*np.eye(p2)/N
    Delta_b1b2=lambdax*bb*X1.T@B3@X2/N
    Delta_bb=np.hstack((np.vstack((Delta_b1b1,Delta_b1b2.T)),np.vstack((Delta_b1b2,Delta_b2b2.T))))
    
    Delta_rs=(mu_ep-3*lambda2**2)/N*(sum([B1[i][i]*B2[i][i] for i in range(len(B2))])+sum([B1[i][i]*B6[i][i] for i in range(len(B1))]))+(mu_e-3*hsig**2)/N*(sum([B3[i][i]*B4[i][i] for i in range(len(B3))])+sum([B3[i][i]*B5[i][i] for i in range(len(B3))]))
    Delta_rs=Delta_rs-2*lambdax*lambda2*bb*(B4@Omega1).trace()/N-0.5*lambdax**2*lambda2*bb**2*((B4+B4.T)@B3).trace()/N
    
    Delta_ss=(mu_e-3*hsig**2)/(4*N)*sum([B3[i][i]**2 for i in range(len(B3))])+(mu_ep-3*lambda2**2)/(4*N)*sum([B1[i][i]**2 for i in range(len(B1))])
    Delta_ss=Delta_ss+0.5/N*lambdax**2*bb**2*(B3@B3).trace()+0.25*(mu_ex-3*lambdax**2)*bb**2*sum([B3[i][i]**2 for i in range(len(B3))])/N+lambdax*bb*(B3@Omega1).trace()/N
    
    Delta_rb1=lambdax*bb*X1.T@Omega1@B5@X@hrbeta[0:p,]/N
    Delta_rb2=lambdax*bb*X2.T@Omega1@B5@X@hrbeta[0:p,]/N+lambda2*lambdax*bb/N*(B4).trace()*beta2+lambdax**2*lambda2*bb*p/N*(B4).trace()*beta2
    Delta_rb2=Delta_rb2-lambdax**2*bb*(B3@G).trace()*beta2/N
    Delta_rb=np.vstack((Delta_rb1,Delta_rb2))
    
    Delta_sb1=np.zeros((p1,1))
    Delta_sb2=-2*lambdax*(B3).trace()/N*beta2-2*lambdax**2*bb*(B3@Omega1).trace()/N*beta2
    Delta_sb=np.vstack((Delta_sb1,Delta_sb2))
    
    #calculate \Sigma_1 in Theorem 4
    Sigm1=np.hstack((np.vstack((CO1+Delta_bb,np.transpose(CO3)+Delta_rb.T,Delta_sb.T)),np.vstack((CO3+Delta_rb,(CO2+Delta_rr),(CO4+Delta_rs))),np.vstack((Delta_sb,(CO4+Delta_rs),(1.0/(2*N)*(Omega1@Omega1).trace()+Delta_ss)))))
    
    
    CO=np.linalg.inv(Sigm2)@Sigm1@np.linalg.inv(Sigm2)
    CO=[CO[i][i] for i in range(p+2)]
    SE1=1.0/N**0.5*np.sqrt((np.array(CO)).reshape((p+2,1)))
    return SE1 
