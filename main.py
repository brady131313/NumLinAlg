import os
import time

import matrix
import graph
import decomp
import solvers
import util

def main():
    start = time.time()
    with open(util.getMatrixFile("25.mtx")) as file:
        g = graph.Graph.fromFile(file)
    end = time.time()

    #TODO for some reason EtE != D + A
    #Perhaps I'm finding either the edge_vertex matrix wrong
    #Maybe degree matrix is found wrong?

    print(g.adjacency)
    print(g.getDegree())
    print(g.getVertexEdge().multMat(g.edgeVertex))


    print(f"Total operation time: {end - start}")



main()
