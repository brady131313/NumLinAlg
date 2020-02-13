import os
import time

import matrix
import graph
import decomp
import solvers
import util

def main():
    with open(util.getMatrixFile("test.mtx")) as file:
        A = graph.Graph.fromFileToEdge(file)


    #TODO continue testing to make sure reading file to edge work
    A.visualizeShape()

    start = time.time()
    end = time.time()


    print(f"Total operation time: {end - start}")



main()
