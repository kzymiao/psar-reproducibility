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
warnings.filterwarnings("ignore")

def f_cls_biascorr(hbeta1,hbeta2,hrho,Y,X,W,lambda2,lambdax):
    """
    calculate the first and second order derivative in CLS method and the estimation of sigma^2.
    :hbeta: the estimate of beta in current iterative step
    :hrho: the estimate of rho in current iterative step
    :W: the network weighting matrix
    :lambda2: the variance of artificially created noise
    :return: the first order derivative in CLS method, the second order derivative in CLS method, the value of objective function, d_\rho, \dot(d_\rho), \ddot(d_\rho) and the estimate of sigma^2.
    """
   
    N=len(Y)
    p1=len(hbeta1)
    p2=len(hbeta2)
    p=p1+p2
    X1=X[:,0:p1]
    X2=X[:,p1:p]
    
    hS= np.eye(N)-hrho*W.toarray()
    
    
    SS=(hS.T)@hS
    
    V=hS.dot(Y)-(X1.dot(hbeta1)+X2.dot(hbeta2)).reshape(N,1)
    
    
    hsigmax=(V.T.dot(V)-lambda2*(SS).trace())/N-lambdax*hbeta2.T@hbeta2
    hsigmax=hsigmax[0,0]
    hsigmax1=1.0/hsigmax
    hsig=hsigmax
    
    s1=1.0/hsig
    
    
    d=((1/np.sum(hS*hS,axis=0))).reshape(N,1)
    D=np.diag(d.flatten())
    dd=d*d
    DD=np.diag(dd.flatten())
    
    W2=(W.T.dot(W)).toarray()
    WW=(np.sum(W.toarray()*W.toarray(),axis=0)).reshape(N,1)
    ldD=-2*hrho*(dd)*(WW)
    dD=np.diag(ldD.flatten())

    lddD=-2*(dd)*(WW)+8*hrho**2*(d*dd*WW*WW)
    ddD=np.diag(lddD.flatten())
    
    WS=W.T@hS
    stv=hS.T.dot(V)
    stwy=WS.T.dot(Y)
    WS=WS+WS.T
    stx=hS.T.dot(X)
    wtv=W.toarray().T.dot(V)
    wtv=W.T.dot(V)
    
    
    
    drho=-2*(stwy.T.dot(DD).dot(stv)+V.T.dot(W.toarray()).dot(DD).dot(stv)-stv.T.dot(dD*D).dot(stv))
    dbeta1=-2*X1.T.dot(hS).dot(DD).dot(stv)
    dbeta2=-2*X2.T.dot(hS).dot(DD).dot(stv)
    
    dsv=d*(stv.A)
    obj=np.mean((dsv)**2)
    
    wsss=WS*SS
    ssss=SS.T*(SS)
    
    deltarho=np.sum(wsss,axis=0).dot(dd)[0]-np.sum(ssss,axis=0).dot(d*ldD)[0]
    deltarho=2*lambda2*deltarho
    drho=drho + deltarho
    
   
    ss=hS*hS
    d2=np.sum(ss,axis=0).dot(d*ldD)[0]
    
    ws=W.toarray()*hS
    d1=np.sum(ws,axis=0).dot(dd)[0]
    drho=drho-2*lambdax*(hbeta2.T.dot(hbeta2))[0,0]*(d2-d1)
    dbeta2=dbeta2-2*lambdax*(SS.dot(DD)).trace()*hbeta2
    
    f=np.vstack((dbeta1,dbeta2,drho))
    
    
    
    drhobeta=X.T.dot(W.toarray()).dot(DD).dot(stv)-2*stx.T.dot(dD*D).dot(stv)+stx.T.dot(DD).dot(wtv)+stx.T.dot(DD).dot(stwy)
    drhobeta=2*drhobeta
    drhobeta1=drhobeta[0:p1]
    
    
    drhobeta2=drhobeta[p1:p]+4*lambdax*d1*hbeta2-4*d2*hbeta2

    d0=np.sum(ss,axis=0).dot(dd)[0]
    
    dbeta11=2*X1.T@hS@DD@hS.T@X1
    dbeta22=2*X2.T@hS@DD@hS.T@X2-2*lambdax*d0
    dbeta12=2*X1.T@hS@DD@hS.T@X2
    
    drho2=2*Y.T.dot(W2).dot(DD).dot(stv)-4*stwy.T.dot(D*dD).dot(stv)+2*stwy.T.dot(DD).dot(wtv)+stwy.T.dot(DD).dot(stwy)-4*wtv.T.dot(D*dD).dot(stv)+wtv.T.dot(DD).dot(wtv)+stv.T.dot(ddD*D).dot(stv)+stv.T.dot(dD*dD).dot(stv)
    

    d0=np.sum(ss,axis=0).dot(d*lddD+ldD*ldD)[0]
    d1=np.sum(W.toarray()*W.toarray(),axis=0).dot(dd)[0]
    d2=np.sum(ws,axis=0).dot(d*ldD)[0]
    drho2=2*drho2+8*lambdax*(hbeta2.T.dot(hbeta2))[0,0]*(d2)-2*lambdax*(hbeta2.T.dot(hbeta2))[0,0]*(d1+d0)

    
    deltarho2=-np.sum(WS*(WS.T),axis=0).dot(dd)[0]-2*np.sum((SS.T)*(W2),axis=0).dot(dd)[0]+4*np.sum(wsss,axis=0).dot(d*ldD)[0]-np.sum(ssss,axis=0).dot(ldD**2+d*lddD)[0]
    deltarho2=lambda2*deltarho2
    drho2=drho2+2*deltarho2
   
    
    gt=np.vstack((dbeta11, np.transpose(dbeta12), np.transpose(drhobeta1)))
    gtt=np.vstack((dbeta12, dbeta22, np.transpose(drhobeta2)))
    g=np.hstack((gt,gtt,np.vstack((drhobeta1, drhobeta2, drho2))))
   
    
    return f,g,obj,D,dD,ddD,hsig

def cls_estimate_x(X,Y1,W,p1,lambda2,lambdax):
    """
    Obtain corrected least squares estimator by Newton-Raphson algorithm.
    :Y1: the observed response
    :W: the network weighting matrix
    :lambda2: the variance of artificially created noise
    :return: the estimate vector of (rho, beta), the estimate of sigma^2.
    """
    
    N=len(Y1)
    p=len(X[1])
    p2=p-p1
    K=50
    e=10**(-4)  
    X1=X[:,0:p1]
    X2=X[:,p1:p]
    hrbeta=(0.01*np.ones(p+1)).reshape((p+1,1))
    
    k=0  
    delta1=1
    obj0=-1
    
    flag=1
    while flag>e and k<=K:
        f,g,obj,D,dD,ddD,hsig=f_cls_biascorr(hrbeta[0:p1,],hrbeta[p1:p,],hrbeta[p,],Y1,X,W,lambda2,lambdax)
        if (np.abs(hrbeta[p,])>1):
            hrbeta[p,]=np.random.rand(1)
            hrbeta[0:p,]=np.random.rand(p).reshape((p,1))
            flag=1
        if (np.abs(hrbeta[p,])>1):
            hrbeta-=0.1*np.dot(np.linalg.inv(g),f)
            flag=np.linalg.norm(0.1*np.dot(np.linalg.inv(g),f), ord=None, axis=None, keepdims=False)
        else:
            hrbeta-=np.dot(np.linalg.inv(g),f)
            flag=np.linalg.norm(np.dot(np.linalg.inv(g),f), ord=None, axis=None, keepdims=False)
        delta1=np.abs(obj-obj0)
        obj0=obj
        k+=1
    
    return hrbeta,hsig,D,dD,ddD

def cls_inference(X,Y1,hrbeta,hsig,lambda2,mu_ep,W,D,dD,ddD,p1,lambdax,mu_ex):
    """
    Obtain estimated standard error SE based on Theorem 6.
    :Y1: the observed response
    :hrbeta: the estimate vector of (rho, beta)
    :hsig: the estimate of sigma^2
    :lambda2: the variance of artificially created noise
    :mu_ep: the fourth moment of epsilon
    :W: the network weighting matrix
    :D: d_\rho
    :dD: \dot(d_\rho)
    :ddD: \ddot(d_\rho)
    :return: the estimated standard error SE.
    """
    p=len(X[1])
    p2=p-p1
    N=len(Y1)
    X1=X[:,0:p1]
    X2=X[:,p1:p]
    beta1=hrbeta[0:p1]
    beta2=hrbeta[p1:p]
    bb=(beta2.T@beta2)[0][0]
    
    S= np.eye(N)-hrbeta[p,][0]*W.toarray()  
    V=S.dot(Y1)-(X.dot(hrbeta[0:p,])).reshape(N,1)
    hsig=(V.T.dot(V)-lambda2*(S@S.T).trace())/N
    hsig=hsig[0,0]
    
    Omega=hsig*np.eye(N)+lambda2*S.dot(S.T)
    S1=np.linalg.inv(S)
    G=W.dot(S1)

    F=D@S.T@X
    J=D@S.T@G@X@hrbeta[0:p,]
    H=D@S.T@Omega@S*D
    M=dD@S.T-D@(W.toarray()).T-D@S.T@G
    M1=S@D@M
    M2=S.T@M1@S
    
    mu_e=np.mean(np.array(V)**4)-(1-hrbeta[p,][0])**4*mu_ep-6*(1-hrbeta[p,][0])**2*hsig*lambda2
    mu_exb=(mu_ex-lambdax**2)*np.sum(beta2**4)+lambdax**2*np.sum(beta2**2)**2
    
    #calculate \Sigma_1 in Theorem 6
    SDS=S@D@D@S.T
    SWSS=S@D@D@W.T-S@D@dD@S.T
    SDW=S@D@D@W.T
    SdS=S@dD@D@S.T
    
    Sig1_rr=4.0/N*(J.T@H@J+(M1@Omega@M1@Omega).trace()+(M1@Omega@M1.T@Omega).trace()+(mu_e-3*hsig**2)*sum([M1[i][i]**2 for i in range(len(M1))])+(mu_ep-3*lambda2**2)*sum([M2[i][i]**2 for i in range(len(M2))]))
    Sig1_rr1=8*lambdax**2*bb**2*(SDW@SDW).trace()+16*lambdax*bb*(SDW@Omega@SDW.T).trace()+4*lambdax*bb*hrbeta[0:p].T@X.T@G.T@SDS@SDS@G@X@hrbeta[0:p]+4*lambdax*bb*(SDS@G@Omega@G.T@SDS).trace()+4*(mu_exb-3*lambdax**2*bb**2)*sum([SDW[i][i]**2 for i in range(len(SDW))])
    Sig1_rr1=Sig1_rr1+16*lambdax*bb*(SdS@Omega@SdS).trace()+8*lambdax**2*bb**2*(SdS@SdS).trace()-32*lambdax*bb*(SDW@Omega@SdS).trace()-8*lambdax**2*bb**2*(SdS@(SDW+SDW.T)).trace()+4*(mu_exb-3*lambdax**2*bb**2)*sum([SdS[i][i]**2 for i in range(len(SdS))])
    Sig1_rr1=Sig1_rr1+8*lambdax*bb*((SDW+SDW.T)@SDS@G@Omega).trace()-16*lambdax*bb*(SdS@SDS@G@Omega).trace()-8*(mu_exb-3*lambdax**2*bb**2)*sum([SDW[i][i]*SdS[i][i] for i in range(len(SDW))])
    Sig1_rr=Sig1_rr+Sig1_rr1/N-4*lambdax**2*bb**2*(G.T@SDS@SDS@G).trace()/N
    
    Sig1_bb=4/N*F.T@H@F
    Sig1_b1b1=Sig1_bb[0:p1,0:p1]+4*lambdax*bb*X1.T@S@D@D@S.T@S@D@D@S.T@X1/N
    Sig1_b2b2=Sig1_bb[p1:p,p1:p]+4*lambdax*bb*X2.T@SDS@SDS@X2/N+8*p2*lambdax**2*(SDS@SDS).trace()/N*beta2@beta2.T+4*lambdax*(SDS@Omega@SDS).trace()/N*np.eye(p2)
    Sig1_b2b2=Sig1_b2b2-4*lambdax**2*bb*(SDS@SDS).trace()*np.eye(p2)/N
    Sig1_b1b2=Sig1_bb[0:p1,p1:p]+4*lambdax*bb*X1.T@SDS@SDS@X2/N
    Sig1_bb=np.hstack((np.vstack((Sig1_b1b1,Sig1_b1b2.T)),np.vstack((Sig1_b1b2,Sig1_b2b2))))
    
    Sig1_br=4/N*F.T@H@J
    Sig1_b1r=4*lambdax*bb/N*X1.T@SDS@SDS@G@X@hrbeta[0:p,]
    Sig1_b2r=lambdax*bb*X2.T@SDS@SDS@G@X@hrbeta[0:p,]-2*lambdax*(SWSS@Omega@SDS).trace()*beta2-lambdax**2*bb*((SWSS+SWSS.T)@SDS).trace()*beta2
    Sig1_b2r=4*Sig1_b2r/N-4*lambdax**2*bb*(SDS@SDS@G).trace()/N*beta2
    Sig1_br=Sig1_br+np.vstack((Sig1_b1r,Sig1_b2r))
    
    Sig1=np.hstack((np.vstack((Sig1_bb,Sig1_br.T)),np.vstack((Sig1_br,Sig1_rr))))
    
    #calculate \Sigma_2 in Theorem 6
    Sig2_rr=2/N*(hsig*(M.T@M).trace()+J.T@J+2*hrbeta[p][0]*(W@D@D@W.T).trace()+hrbeta[p][0]*(S@ddD@D@S.T).trace()-4*hrbeta[p][0]*(dD@D@S.T@W).trace())
    Sig2_rr=Sig2_rr-2*lambdax*bb*(G.T@SDS@G).trace()/N
    
    Sig2_bb=2/N*F.T@F
    Sig2_b1b1=Sig2_bb[0:p1,0:p1]
    Sig2_b2b2=Sig2_bb[p1:p,p1:p]-2*lambdax*(SDS).trace()*np.eye(p2)/N
    Sig2_b1b2=Sig2_bb[0:p1,p1:p]
    Sig2_bb=np.hstack((np.vstack((Sig2_b1b1,Sig2_b1b2.T)),np.vstack((Sig2_b1b2,Sig2_b2b2))))
    
    Sig2_br=2/N*F.T@J
    Sig2_br=Sig2_br-np.vstack((np.zeros((p1,1)),2*lambdax*(SDS@G).trace()*beta2/N))
    
    Sig2=np.hstack((np.vstack((Sig2_bb,Sig2_br.T)),np.vstack((Sig2_br,Sig2_rr))))
    
    CO=np.linalg.inv(Sig2)@Sig1@np.linalg.inv(Sig2)
    CO=[CO[i][i] for i in range(p+1)]
    SE3=1.0/N**0.5*np.sqrt((np.array(CO)).reshape((p+1,1)))
    
    return SE3
