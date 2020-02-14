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


    #TODO should adjacency matrix have entries along diagonal?

    print(g.adjacency)
    g.adjacency.visualizeShape()

    #print(g.getDegree())
    #print(g.getVertexEdge().multMat(g.edgeVertex))


    print(f"Total operation time: {end - start}")



main()
