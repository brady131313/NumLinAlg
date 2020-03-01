import os
import time

import util
from matrix import Sparse
from graph import Graph, formVertexToK1Aggregate, QModularity, recursiveLouvains
from graph.graph import *
import plot

def main():
    with open(util.getMatrixFile("50.mtx")) as file:
        g = Graph.fromFile(file)
    
    As, Ps = recursiveLouvains(g)

    Pk = formVertexToK1Aggregate(Ps, len(Ps))
    plot.visualizeGraph(g.edgeVertex, Pk)
    
    #L = g.getLaplacian()
    #X = plot.formCoordinateVectors(L, 3)
    #plot.visualize(X, Pk, 3)

if __name__ == '__main__':
    main()
