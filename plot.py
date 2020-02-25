from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def formCoordinateVectors(L, d):
    converted = csr_matrix((L.data, L.colInd, L.rowPtr), (L.rows, L.columns))
    converted = converted.asfptype()

    eigs, vecs = eigsh(converted, d, which='SM', tol=1e-3)
    return vecs

def getColors(X, P):
    colors = [0 for _ in range(len(X))]

    for i in range(P.rows):
        for k in range(P.rowPtr[i], P.rowPtr[i + 1]):
            j = P.colInd[k]
            colors[i] = j
    
    #colorMap = cm.get_cmap('Dark2')
    return colors

def visualize(X, P, d):
    if d > 3: return

    xCoords = [x[0] for x in X]
    yCoords = [x[1] for x in X]
    if d == 3:
        zCoords = [x[2] for x in X]

    colors = getColors(X, P)

    size, alpha = (75, 0.4) if len(xCoords) < 1250 else (50, 0.2)

    if d == 2:
        plt.scatter(xCoords, yCoords, c=colors, s=size, alpha=alpha, marker="o")
        plt.show()
    elif d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xCoords, yCoords, zCoords, c=colors, s=size, alpha=alpha)
        plt.show()