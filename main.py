import os
import time

import util
from matrix import Sparse

def main():
    with open(util.getMatrixFile("25.mtx")) as file:
        A = Sparse.fromFile(file)

    A.visualizeShape()
    
    

    

if __name__ == '__main__':
    main()
