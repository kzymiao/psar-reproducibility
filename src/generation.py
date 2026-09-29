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
import itertools
import math
import warnings
warnings.filterwarnings("ignore")


def generate_covariates(N, p):
    """Generate N x p standard-normal covariates truncated to [-2, 2].

    The paper bounds the infinity norm of each covariate vector by 2.
    Values outside [-2, 2] are resampled until all coordinates satisfy
    the bound.
    """
    X = np.random.normal(0, 1, N * p).reshape(N, p)
    mask = np.abs(X) > 2
    while np.any(mask):
        X[mask] = np.random.normal(0, 1, np.sum(mask))
        mask = np.abs(X) > 2
    return X


def generate_PowerLaw(N):  
    
    df=np.ceil(powerlaw.Power_Law(xmin=3,parameters=[3.5]).generate_random(N))

    df[df>N]=N
    R=np.zeros((int(sum(df)),1))
    s1,s2=0,0
    for i in range(N):
        s1=s2
        s2=s2+int(df[i])
        R[s1:s2]=i
    C=np.zeros(())
    for i in range(N):
        pos=np.random.permutation(N).reshape(N,1)
        C=np.vstack((C,pos[0:int(df[i])]))
    C=C[1:]
    rr=R[R!=C]
    cc=C[R!=C]
    l=np.ones((rr.shape))
    A=sps.coo_matrix((l,(rr,cc)),shape=(N,N))
    Asum = np.array(A.sum(1))
    r_inv = np.power(Asum.astype(float), -1).flatten()
    r_mat_inv = sps.diags(r_inv)
    W = r_mat_inv.dot(A)
    
    return W

def generate_PowerLaw1(N):  
    
    s=nx.utils.powerlaw_sequence(N, 4)
    G = nx.expected_degree_graph(s)
    G=nx.Graph(G)
    A=np.array(nx.adjacency_matrix(G).todense())
    #A=nx.to_numpy_matrix(G)
    A[np.diag_indices_from(A)]=0
    Asum = np.array(A.sum(1))
    A=coo_matrix(A)
    r_inv = np.power(Asum.astype(float), -1).flatten()
    r_mat_inv = sps.diags(r_inv)
    W = r_mat_inv.dot(A)
    return W

def generate_Dyad(N):  
    B=np.ones((N,N))
    A=np.zeros((N,N))
    ind0=np.where(np.triu(B,1))
    ind=(np.array([ind0[0], ind0[1]])).T
    delta=1.2
    N1=10
    s1 = [randint(0,(len(ind)-1)) for _ in range(len(ind))]
    indM = ind[sample(s1, N1),]### sample N*N1 as mutual pairs in the upper triangular matrix
    for i in range(len(indM)):
        A[indM[i][0],indM[i][1]]=1
        A[indM[i][1],indM[i][0]]=1
    ind1 = np.where(A==0)
    ind2=(np.array([ind1[0], ind1[1]])).T
    s2 = [randint(0,(len(ind2)-1)) for _ in range(len(ind2))]
    indS = ind2[sample(s2, int(N**delta)),] 
    s3 = [randint(0,(len(indS)-1)) for _ in range(len(indS))]
    tmp = sample(s3, int(N**delta/2))
    for i in range(len(indS)):
        A[indS[i][1],indS[i][0]]=1
    A[np.diag_indices_from(A)]=0
    Asum = np.array(A.sum(1))
    A=coo_matrix(A)
    r_inv = np.power(Asum.astype(float), -1).flatten()
    r_mat_inv = sps.diags(r_inv)
    W = r_mat_inv.dot(A)
    return W


def _get_num_pos_edges(c1_size, c2_size, same_cluster, self_loops, directed):
    """
    Compute the number of possible edges between two clusters.
    :param c1_size: The size of the first cluster
    :param c2_size: The size of the second cluster
    :param same_cluster: Whether these are the same cluster
    :param self_loops: Whether we will generate self loops
    :param directed: Whether we are generating a directed graph
    :return: the number of possible edges between these clusters
    """
    if not same_cluster:
        # The number is simply the product of the number of vertices
        return c1_size * c2_size
    else:
        # The base number is n choose 2
        possible_edges_between_clusters = int((c1_size * (c1_size - 1)) / 2)

        # If we are allowed self-loops, then add them on
        if self_loops:
            possible_edges_between_clusters += c1_size

        # The number is normally the same for undirected and directed graphs, unless the clusters are the same, in which
        # case the number for the directed graph is double since we need to consider both directions of each edge.
        if directed:
            possible_edges_between_clusters *= 2

        # But if we are allowed self-loops, then we shouldn't double them since there is only one 'direction'.
        if directed and self_loops:
            possible_edges_between_clusters -= c1_size

        return possible_edges_between_clusters


def _get_number_of_edges(c1_size, c2_size, prob, same_cluster, self_loops, directed):
    """
    Compute the number of edges there will be between two clusters.
    :param c1_size: The size of the first cluster
    :param c2_size: The size of the second cluster
    :param prob: The probability of an edge between the clusters
    :param same_cluster: Whether these are the same cluster
    :param self_loops: Whether we will generate self loops
    :param directed: Whether we are generating a directed graph
    :return: the number of edges to generate between these clusters
    """
    # We need to compute the number of possible edges
    possible_edges_between_clusters = _get_num_pos_edges(c1_size, c2_size, same_cluster, self_loops, directed)

    # Sample the number of edges from the binomial distribution
    return np.random.binomial(possible_edges_between_clusters, prob)


def _generate_sbm_edges(cluster_sizes, prob_mat_q, directed=False):
    """
    Given a list of cluster sizes, and a square matrix Q, generates edges for a graph in the following way.
    For two vertices u and v where u is in cluster i and v is in cluster j, there is an edge between u and v with
    probability Q_{i, j}.
    For the undirected case, we assume that the matrix Q is symmetric (and in practice look only at the upper triangle).
    For the directed case, we generate edges (u, v) and (v, u) with probabilities Q_{i, j} and Q_{j, i} respectively.
    May return self-loops. The calling code can decide what to do with them.
    Returns edges as pairs (u, v) where u and v are integers giving the index of the respective vertices.
    :param cluster_sizes: a list giving the number of vertices in each cluster
    :param prob_mat_q: A square matrix where Q_{i, j} is the probability of each edge between clusters i and j. Should
                       be symmetric in the undirected case.
    :param directed: Whether to generate a directed graph (default is false).
    :return: Edges (u, v).
    """
    # We will iterate over the clusters. This variable keeps track of the index of the first vertex in the current
    # cluster_1.
    c1_base_index = 0

    for cluster_1 in range(len(cluster_sizes)):
        # Keep track of the index of the first vertex in the current cluster_2
        c2_base_index = c1_base_index

        # If we are constructing a directed graph, we need to consider all values of cluster_2.
        # Otherwise, we will consider only the clusters with an index >= cluster_1.
        if directed:
            second_clusters = range(len(cluster_sizes))
            c2_base_index = 0
        else:
            second_clusters = range(cluster_1, len(cluster_sizes))

        for cluster_2 in second_clusters:
            # Compute the number of edges between these two clusters
            num_edges = _get_number_of_edges(cluster_sizes[cluster_1],
                                             cluster_sizes[cluster_2],
                                             prob_mat_q[cluster_1][cluster_2],
                                             cluster_1 == cluster_2,
                                             True,
                                             directed)

            # Sample this number of edges. TODO: correct for possible double-sampling of edges
            num_possible_edges = (cluster_sizes[cluster_1] * cluster_sizes[cluster_2]) - 1
            for i in range(num_edges):
                edge_idx = random.randint(0, num_possible_edges)
                u = c1_base_index + int(edge_idx / cluster_sizes[cluster_1])
                v = c2_base_index + (edge_idx % cluster_sizes[cluster_1])
                yield u, v

            # Update the base index for the second cluster
            c2_base_index += cluster_sizes[cluster_2]

        # Update the base index of this cluster
        c1_base_index += cluster_sizes[cluster_1]


def sbm_adjmat(cluster_sizes, prob_mat_q, directed=False, self_loops=False):
    """
    Generate a graph from the stochastic block model.
    The list cluster_sizes gives the number of vertices inside each cluster and the matrix Q gives the probability of
    each edge between pairs of clusters.
    For two vertices u and v where u is in cluster i and v is in cluster j, there is an edge between u and v with
    probability Q_{i, j}.
    For the undirected case, we assume that the matrix Q is symmetric (and in practice look only at the upper triangle).
    For the directed case, we generate edges (u, v) and (v, u) with probabilities Q_{i, j} and Q_{j, i} respectively.
    Returns the adjacency matrix of the graph as a sparse scipy matrix in the CSR format.
    :param cluster_sizes: The number of vertices in each cluster.
    :param prob_mat_q: A square matrix where Q_{i, j} is the probability of each edge between clusters i and j. Should
                       be symmetric in the undirected case.
    :param directed: Whether to generate a directed graph (default is false).
    :param self_loops: Whether to generate self-loops (default is false).
    :return: The sparse adjacency matrix of the graph.
    """
    # Initialize the adjacency matrix
    adj_mat = scipy.sparse.lil_matrix((sum(cluster_sizes), sum(cluster_sizes)))

    # Generate the edges in the graph
    for (u, v) in _generate_sbm_edges(cluster_sizes, prob_mat_q, directed=directed):
        if u != v or self_loops:
            # Add this edge to the adjacency matrix.
            adj_mat[u, v] = 1

            if not directed:
                adj_mat[v, u] = 1

    # Reformat the output matrix to the CSR format
    return adj_mat.tocsr()


def ssbm_adjmat(n, k, p, q, directed=False):
    """
    Generate a graph from the symmetric stochastic block model.
    Generates a graph with n vertices and k clusters. Every cluster will have floor(n/k) vertices. The probability of
    each edge inside a cluster is given by p. The probability of an edge between two different clusters is q.
    :param n: The number of vertices in the graph.
    :param k: The number of clusters.
    :param p: The probability of an edge inside a cluster.
    :param q: The probability of an edge between clusters.
    :param directed: Whether to generate a directed graph.
    :return: The sparse adjacency matrix of the graph.
    """
    # Every cluster has the same size.
    cluster_sizes = [int(n/k)] * k

    # Construct the k*k probability matrix Q. The off-diagonal entries are all q and the diagonal entries are all p.
    prob_mat_q = []
    for row_num in range(k):
        new_row = [q] * k
        new_row[row_num] = p
        prob_mat_q.append(new_row)

    # Call the general sbm method.
    return sbm_adjmat(cluster_sizes, prob_mat_q, directed=directed)

def generate_Block(N):
    Nblock=20
    A = ssbm_adjmat(N, Nblock, 10/N, 2/N, directed=False)
    Asum = np.array(A.sum(1))
    r_inv = np.power(Asum.astype(float), -1).flatten()
    r_mat_inv = sps.diags(r_inv)
    W = r_mat_inv.dot(A)
    
    return W
    


def generate(N,p,beta,rho,type_w,distri_e,distri_ep,lambda2):
    
    #np.random.seed(123)
    if type_w == 'PowerLaw':
        W=generate_PowerLaw1(N)
    elif type_w == 'Dyad':
        W=generate_Dyad(N)
    elif type_w == 'SBM':
        W=generate_Block(N)
    
    #sigma2=1
    
    S = np.eye(N)-rho*W
    S1=np.linalg.inv(S)
    X=generate_covariates(N, p)
    #cov=[[1. , 0.2 ],[0.2 , 1]]
    #X= np.random.multivariate_normal([0]*p, cov, size=(N))

    
    if distri_e == 'N':
        sigma2=1
        E=np.random.normal(0,sigma2**0.5,N).reshape(N,1)
        
    elif distri_e == 't':
        prng = np.random.RandomState(123456789)
        E=prng.standard_t(6, size=N).reshape(N,1)
        sigma2=1.5
        
    if distri_ep == 'N':
        epsilon=np.random.normal(0,lambda2**0.5,N).reshape(N,1)
        mu_ep=3*lambda2*lambda2
    elif distri_ep == 't':
        prng = np.random.RandomState(12345678)
        epsilon=prng.standard_t(6, size=N).reshape(N,1)/np.sqrt(3)
        mu_ep=6*1.5*1.5/9
    
    Y=S1.dot(X.dot(beta)+E)
    Y1=Y+epsilon
    return X,Y1,W,mu_ep