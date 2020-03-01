from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx

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

def visualizeGraph(E, P, w = None):
    G = nx.Graph()
    G.add_nodes_from([i for i in range(E.columns)])

    if not w:
        w = [1 for _ in range(E.rows)]

    for i in range(E.rows):
        edge = E.colInd[E.rowPtr[i]:E.rowPtr[i + 1]]
        G.add_edge(edge[0], edge[1], weight=w[i])

    colors = [0 for _ in range(P.rows)]
    for i in range(P.rows):
        for k in range(P.rowPtr[i], P.rowPtr[i + 1]):
            colors[i] = P.colInd[k]

    nx.draw_networkx(G, with_labels = False, alpha = 0.6, node_color=colors)
    plt.show()