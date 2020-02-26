
# DISCLAIMER:
#
# All the functions in this file (except for `equisp_unisolv`) are translated in python from MATLAB.
#   The original MATLAB files can be found in this webpages
#           https://www.math.unipd.it/~marcov/CAAsoft.html
#   and have been released with GNU/General Public License v2 by the CAA group
#           https://www.math.unipd.it/~marcov/CAA.html
#
# For reasons academic good practice, we ask the users to cite the following papers in every scientific product using this script:
#
#   1. M. Caliari, S. De Marchi, A. Sommariva and M. Vianello
#           Padua2DM: fast interpolation and cubature at the Padua points in Matlab/Octave - Numer. Algorithms 56 (2011)
#   2. Bos, L., De Marchi, S., Sommariva, A., & Vianello, M. (2011)
#           Weakly Admissible Meshes and Discrete Extremal Sets. Numerical Mathematics: Theory, Methods and Applications, 4(1), 1-12. #           doi:10.1017/S1004897900000507

import numpy as np

def wamfit(deg, wam, pts, fval):
    # Compute the coefficients for polynomial approximation
    both = np.vstack((wam, pts))
    rect = [np.min(both[:,0]), np.max(both[:,0]), np.min(both[:,1]), np.max(both[:,1])]
    Q, R1, R2 = wamdop(deg, wam, rect)
    DOP = wamdopeval(deg, R1, R2, pts, rect)
    cfs = np.matmul(Q.T, fval)
    lsp = np.matmul(DOP, cfs)
    return np.array(cfs), np.array(lsp)


def wamdopeval(deg,R1,R2,pts,rect):
    # Evaluate the approximant
    W = chebvand(deg, pts, rect)
    TT = np.linalg.solve(R1.T, W.T).T
    return np.linalg.solve(R2.T, TT.T).T


def wamdop(deg,wam,rect):
    # Factorize the Vandermonde matrix
    V = chebvand(deg,wam,rect)
    Q1, R1 = np.linalg.qr(V)
    TT = np.linalg.solve(R1.T,V.T).T
    Q, R2 = np.array(np.linalg.qr(TT))
    return Q, R1, R2


def chebvand(deg,wam,rect):
    # Construct the Vandermonde matrix
    j = np.linspace(0, deg, deg+1)
    j1, j2 = np.meshgrid(j, j)
    j11 = j1.T.flatten()
    j22 = j2.T.flatten()

    good = np.argwhere(j11+j22 < deg+1)
    couples = np.matrix(np.vstack((j11[good].T, j22[good].T)).T)
    a, b, c, d = rect
    mappa1 = (2.* wam[:, 0] - b - a) / (b - a)
    mappa2 = (2.* wam[:, 1] - d - c) / (d - c)
    mappa = np.vstack((mappa1.T, mappa2.T)).T
    V1 = np.cos(np.multiply(couples[:,0], np.arccos(mappa[:,0].T)))
    V2 = np.cos(np.multiply(couples[:,1], np.arccos(mappa[:,1].T)))
    V = np.multiply(V1, V2).T
    return V


def pdpts(n):
    # Compute the Padua points of total degree n in [-1,1]^2
    xyrange = np.array([-1,1,-1,1])
    zn = (xyrange[0]+xyrange[1]+(xyrange[1]-xyrange[0])*
                        np.cos(np.linspace(0,1,n+1)*np.pi))/2
    zn1 = (xyrange[2]+xyrange[3]+(xyrange[3]-xyrange[2])*
                        np.cos(np.linspace(0,1,n+2)*np.pi))/2

    Pad1, Pad2 = np.meshgrid(zn,zn1)

    f1 = np.linspace(0,n,n+1)
    f2 = np.linspace(0,n+1,n+2)

    M1, M2 = np.meshgrid(f1,f2)
    h = np.array(np.mod(M1+M2,2))
    g = np.array(np.concatenate(h.T))

    findM = np.argwhere(g)

    Pad_x = np.concatenate(Pad1.T)[findM]
    Pad_y = np.concatenate(Pad2.T)[findM]
    return Pad_x.reshape(-1), Pad_y.reshape(-1)


def equisp_unisolv(n):
    # Compute an unisolvent subset of equispaced grid of total degree n in [-1,1]^2
    xyrange = np.array([-1,1,-1,1])
    zn  = np.linspace(xyrange[0],xyrange[1],n+1)[::-1]
    zn1 = np.linspace(xyrange[2],xyrange[3],n+2)[::-1]

    X, Y = np.meshgrid(zn,zn1)

    f1 = np.linspace(0,n,n+1)
    f2 = np.linspace(0,n+1,n+2)

    M1, M2 = np.meshgrid(f1,f2)
    h = np.array(np.mod(M1+M2,2))
    g = np.array(np.concatenate(h.T))

    findM = np.argwhere(g)

    x = np.concatenate(X.T)[findM]
    y = np.concatenate(Y.T)[findM]
    return x.reshape(-1), y.reshape(-1)
