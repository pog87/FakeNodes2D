import numpy as np
from lagrange import lagrange_interp

def sliced_interp2D(X,Y,F,XX,YY, mapping = None):
    if mapping is None:
        mapping = lambda x: x
    m,n = X.shape; M,N = XX.shape
    x_slices = np.zeros((m,N))
    interp_values = np.zeros((M,N))
    for i in range(0,m):
        x_slices[i,:] = lagrange_interp( mapping(XX[0,:]), mapping(X[0,:]), F[i,:] )
    for i in range(0,N):
        interp_values[:,i] = lagrange_interp( mapping(YY[:,0]), mapping(Y[:,0]), x_slices[:,i] )
    return interp_values


def sliced_interp3D(X,Y,Z,F,XX,YY,ZZ, mapping = None):
    #untested!!
    if mapping is None:
        mapping = lambda x: x
    m,n,p = X.shape; M,N,P = XX.shape
    x_slices = np.zeros((m,n,P))
    y_slices = np.zeros((m,N,P))
    interp_values = np.zeros((M,N,P))
    for i in range(0,m):
        x_slices[i,:,:] = lagrange_interp( mapping(XX[0,:,:]), mapping(X[0,:,:]), F[i,:,:] )
    for i in range(0,N):
        y_slices[:,i,:] = lagrange_interp( mapping(YY[:,0,:]), mapping(Y[:,0,:,]), x_slices[:,i,:] )
    for i in range(0,P):
        interp_values[:,:,i] = lagrange_interp( mapping(ZZ[:,:,0]), mapping(Z[:,:,0]), y_slices[:,:,i] )
    return interp_values
