import os
import time

import util
from matrix import Sparse
from graph import Graph, formVertexToK1Aggregate, QModularity, recursiveLouvains
from graph.graph import *
import plot

def main():
    with open(util.getMatrixFile("500.mtx")) as file:
        g = Graph.fromFile(file)
    
    As, Ps = recursiveLouvains(g)
    Pk = formVertexToK1Aggregate(Ps, len(Ps))
    Pk.visualizeShape()
    print(Pk.columns)
    
    plot.visualizeGraph(g.edgeVertex, Pk)
    

if __name__ == '__main__':
    main()
