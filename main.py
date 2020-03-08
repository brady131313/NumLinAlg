import os
import time

import util
from matrix import Sparse
from graph import Graph, formVertexToK1Aggregate, QModularity, recursiveLouvains
from graph.graph import *
import plot

def main():
    with open(util.getMatrixFile("25.mtx")) as file:
        A = Sparse.fromFile(file)
    
    A.visualizeShape()
    

if __name__ == '__main__':
    main()
