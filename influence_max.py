# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:39:51 2019

note that nodes of G need to be integers 0,1,...,n for code to work currently

@author: aklamun
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def f_LT(S, G):
    '''Linear threshold influence propagation
    G = weighted directed graph, wts \leq 1
    S = influence set of vertices in G'''
    A = nx.adjacency_matrix(G)
    y = np.array([1 if v in S else 0 for v in G.nodes()])
    z = A*y
    return z

def w_ew(S):
    return len(S)

###############################################################################
'''Integral Influence Maximization'''

def CalcIntCascade(S,f,theta,G):
    '''Calculate influence cascade given integral intervention'''
    S_0 = []
    S_1 = list(set(S))
    while len(S_1) != len(S_0):
        S_0 = S_1[:]
        fSi = f(S_0, G)
        S_1 = [v for v in G.nodes() if fSi[v] >= theta[v] or v in  S_0]
    return S_1

def SigmaInt(S,f,w,G, theta_exp, theta_err, k=10000):
    '''estimate sigma(S) = expected activations
    theta_exp = expected threshold values
    theta_err = theta error range around theta_exp (uniform distr)'''
    sigma = 0
    for i in range(k):
        theta = np.random.uniform(theta_exp-theta_err, theta_exp+theta_err, G.number_of_nodes())
        T = CalcIntCascade(S,f,theta,G)
        sigma += w(T)
    return float(sigma)/k

def GreedyIntInfMax(f,w,b,G,theta_exp=0.5, theta_err=0.5, k=10000):
    '''Greedy algorithm for integral influence maximization'''
    (S_0,S_1) = ([],[])
    (sigma_0,sigma_1) = (0,0)
    while len(S_1) < b and len(S_1) < G.number_of_nodes():
        q = {}
        for v in set(G.nodes())-set(S_1):
            q[v] = SigmaInt(S_1 + [v], f,w,G,theta_exp,theta_err,k)
        S_0 = S_1[:]
        u = max(q, key=q.get)
        S_1 = list(set(S_1 + [u]))
        (sigma_0,sigma_1) = (sigma_1, q[u])
    if len(S_1) <= b:
        return S_1, sigma_1
    else:
        return S_0, sigma_0


###############################################################################
'''Fractional Influence Maximization'''

def CalcFracCascade(x,f,theta,G):
    '''Calculate influence cascade given fractional intervention
    x = vector of fractional influence on nodes'''
    S_0 = []
    S_1 = [v for v in G.nodes() if x[v] >= theta[v]]
    while len(S_1) != len(S_0):
        S_0 = S_1[:]
        fSi = f(S_0, G)
        S_1 = [v for v in G.nodes() if v in S_0 or fSi[v] + x[v] >= theta[v]]
    return S_1

def SigmaFrac(x,f,w,G, theta_exp, theta_err, k=10000):
    '''estimate sigma(S) = expected activations
    theta_exp = expected threshold values
    theta_err = theta error range around theta_exp (uniform distr)'''
    sigma = 0
    for i in range(k):
        theta = np.random.uniform(theta_exp-theta_err, theta_exp+theta_err, G.number_of_nodes())
        T = CalcFracCascade(x,f,theta,G)
        sigma += w(T)
    return float(sigma)/k

def GreedyFracInfMax(f,w,b,G,theta_exp=0.5, theta_err=0.5, k=10000):
    '''Greedy algorithm for fractional influence maximization'''
    (x_0,x_1) = (np.zeros(G.number_of_nodes()),np.zeros(G.number_of_nodes()))
    (sigma_0,sigma_1) = (0,0)
    S = []
    while np.sum(x_1) < b and len(S) < G.number_of_nodes():
        print(np.sum(x_1),len(S))
        q = {}
        fS = f(S,G)
        A = list(set(G.nodes())-set(S))
        for v in A:
            x_v = np.array(x_1, copy=True)
            x_v[v] = theta_exp + theta_err - fS[v]
            q[v] = SigmaFrac(x_v, f,w,G,theta_exp,theta_err,k)
        x_0 = np.array(x_1, copy=True)
        u = max(q, key=q.get)
        u_frac_inf = max(theta_exp + theta_err - fS[u],0)
        if u_frac_inf == 0:
            #in this case, don't consider other nodes that are already passed threshold
            for vv in A:
                if fS[vv] > theta_exp + theta_err:
                    S.append(vv)
            S = list(set(S))
        else:
            x_1[u] = u_frac_inf
            S.append(u)
        (sigma_0,sigma_1) = (sigma_1, q[u])
    if np.sum(x_1) <= b:
        return x_1, sigma_1
    else:
        return x_0, sigma_0


###############################################################################
'''Heuristic for Fractional Influence Maximization'''

def Gamma_neg(v,A,f,G):
    '''\Gamma^-(v,A) = total sum of weight of edges from node v to set A'''
    fv = f([v],G)
    y = np.array([1 if u in A else 0 for u in G.nodes()])
    return np.sum(np.multiply(y,fv))

def Gamma_neg_LT(S,G):
    '''\Gamma^-(v,A) optimized for linear threshold model'''
    y = np.array([1 if u in S else 0 for u in G.nodes()])
    A = nx.adjacency_matrix(G)
    z = A.transpose()*y
    q = {}
    for v in S:
        q[v] = z[v]
    return q

def DiscountFrac(f,b,G,theta_exp=0.5, theta_err=0.5):
    '''heuristic algorithm for fractional influence maximization'''
    (x_0,x_1) = (np.zeros(G.number_of_nodes()),np.zeros(G.number_of_nodes()))
    S = []
    while np.sum(x_1) < b and len(S) < G.number_of_nodes():
        #print(np.sum(x_1),len(S))
        q = {}
        A = list(set(G.nodes())-set(S))
        #for v in A:            #note: this code is used for general f
        #    q[v] = Gamma_neg(v, A,f,G)
        q = Gamma_neg_LT(A,G)   #note: this code is used for linear threshold model only
        x_0 = np.array(x_1, copy=True)
        u = max(q, key=q.get)
        fS = f(S,G)
        u_frac_inf = max(theta_exp + theta_err - fS[u],0)
        if u_frac_inf == 0:
            #in this case, don't consider other nodes that are already passed threshold
            for vv in A:
                if fS[vv] > theta_exp + theta_err:
                    S.append(vv)
            S = list(set(S))
        else:
            x_1[u] = u_frac_inf
            S.append(u)
    if np.sum(x_1) <= b:
        return x_1
    else:
        return x_0
    

