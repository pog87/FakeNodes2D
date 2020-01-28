# =============================================================================
# Funzioni varie su Lagrange
# =============================================================================

import numpy as np
from numpy import matlib

# For compunting the Lebesgue functions
def l_j(j, x, nodes):
    N = np.shape(nodes)[0]
    product = 1.0
    for i in range(0, N):
        if not i==j:
            product *= (x - nodes[i]) / (nodes[j] - nodes[i])
    return product

def lS_j(j, x, nodes,S):
    N = np.shape(nodes)[0]
    product = 1.0
    for i in range(0, N):
        if not i==j:
            product *= (S(x) - S(nodes[i])) / (S(nodes[j]) - S(nodes[i]))
    return product

#def y(x,nodes):
#    N = np.shape(nodes)[0]
#    csum = 0
#    for j in range(0, N):
#        csum += np.abs(l_j(j, x, nodes))
#    return csum

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

#def d(nodes,interval):
#    greatest = None
#    for x in interval:
#        current = y(x,nodes)
#        if greatest is None or current > greatest:
#            greatest = current
#    return greatest

def lebesgue(x,xx):
    N = np.shape(x)[0]
    lsum = np.zeros(np.shape(xx)[0])
    for i in range(0, N):
        lsum += np.abs(l_j(i,xx, x))
    return lsum

def lagrange_interp(xx,x,y):
    n=len(x); m=len(xx)
    L=np.zeros((m,n))
    for k in range(n):
        x_k = np.concatenate((x[:k],x[k+1:]))
        L[:,k] = np.prod(np.matlib.repmat(xx.reshape(m,1),1,n-1) - np.matlib.repmat(x_k,m,1) , axis=1) / np.prod(x[k]-x_k)
    return np.dot(L,y)

def ChebyL(n):
    #compute the n Chebyshev-Lobatto points in [-1,1]
    return -np.cos(np.pi*np.arange(0.,n)/(n-1))

def Cheby(n):
    #compute the n Chebyshev-Lobatto points in [-1,1]
    return -np.cos(np.pi*(2*np.arange(0.,n)-1)/(2*n))