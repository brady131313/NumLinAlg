import os
import time

import matrix
import graph
import decomp
import solvers
import util
import cluster

def main():
    start = time.time()
    with open(util.getMatrixFile("test.mtx")) as file:
        g = graph.Graph.fromFile(file)
    end = time.time()

    w = g.getRandomWeights()

    clusters = cluster.lubys(g.edgeVertex, w)
    print(clusters)
    
    P = graph.getVertexAggregate2(g.adjacency.rows, clusters)
    coarse = graph.formCoarse(P, g.adjacency)

    P.visualizeShape()
    print(coarse)


    print(f"Total operation time: {end - start}")



main()
