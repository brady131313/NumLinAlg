import os
import time

import util
from matrix import Sparse
from graph import Graph, lubys, recursiveLubys
from graph.graph import *
import plot

def main():
    with open(util.getMatrixFile("schedule00.dat")) as file:
        g = Graph.fromFile(file)

    As, Ps = recursiveLubys(g, 4)
    print(len(As), len(Ps))

    P3 = formVertexToK1Aggregate(Ps, len(Ps))
    print(P3.columns)
    
    L = g.getLaplacian()
    X = plot.formCoordinateVectors(L, 3)
    plot.visualize(X, P3, 3)
    
    

if __name__ == '__main__':
    main()
