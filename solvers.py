from enum import Enum

import matrix
import decomp
import util

def forward(A, b):
    if not isinstance(A, matrix.Dense):
        raise Exception("Matrix A must be dense")
    if A.rows != b.dim:
        raise Exception(f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

    data = [0] * A.rows
    for i in range(A.rows):
        data[i] = b.data[i]

        for j in range(i):
            data[i] = data[i] - (A.data[i][j] * data[j])
        
        data[i] = data[i] / A.data[i][i]
    
    return matrix.Vector(A.rows, data)

def forwardSparse(A, b):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != b.dim:
        raise Exception(f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

    data = [0] * A.rows
    for i in range(A.rows):
        data[i] = b.data[i]
        diagonal = 1

        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]
            if i == j:
                diagonal = A.data[k]
            elif i > j:
                data[i] = data[i] - (A.data[k] * data[j])

        data[i] = data[i] / diagonal

    return matrix.Vector(A.rows, data) 

def backward(A, b):
    if not isinstance(A, matrix.Dense):
        raise Exception("Matrix A must be dense")
    if A.rows != b.dim:
        raise Exception(f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

    data = [0] * A.rows
    for i in range(A.rows - 1, -1, -1):
        data[i] = b.data[i]

        for j in range(i + 1, A.rows):
            data[i] = data[i] - (A.data[i][j] * data[j])

        data[i] = data[i] / A.data[i][i]

    return matrix.Vector(A.rows, data)

def backwardSparse(A, b):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != b.dim:
        raise Exception(f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")
    
    data = [0] * A.rows
    for i in range(A.rows - 1, -1, -1):
        data[i] = b.data[i]
        diagonal = 1

        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]
            if i == j:
                diagonal = A.data[k]
            elif i < j:
                data[i] = data[i] - (A.data[k] * data[j])

        data[i] = data[i] / diagonal
    
    return matrix.Vector(A.rows, data)

def vectorSolver(a, b):
    if a.dim != b.dim:
        raise Exception(f"Dimension mismatch, {a.dim} != {b.dim}")

    data = [0] * a.dim
    for i in range(a.dim):
        if a.data[i] == 0 or b.data[i] == 0:
            data[i] = 0
        else:
            data[i] = b.data[i] / a.data[i]

    return matrix.Vector(a.dim, data)

class IterMatrix(Enum):
    l1Smoother = 1
    forwardGaussSeidel = 2
    backwardGaussSeidel = 3
    symmetricGaussSeidel = 4

def stationaryIterative(A, b, xInit, maxIter, tolerance, iterMatrix, displayResidual):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")

    x = xInit
    deltaInit = 0
    delta = 9999

    if iterMatrix == IterMatrix.l1Smoother:
        D = decomp.l1Smoother(A)
    elif iterMatrix == IterMatrix.symmetricGaussSeidel:
        D = decomp.diagonal(A)
    

    for k in range(maxIter):
        r = b - A.multVec(x)
        if k == 0:
            deltaInit = r.norm()
        
        delta = r.norm()
        if displayResidual:
            print(delta)

        if iterMatrix == IterMatrix.l1Smoother:
            z = vectorSolver(D, r)
        elif iterMatrix == IterMatrix.forwardGaussSeidel:
            z = forwardSparse(A, r)
        elif iterMatrix == IterMatrix.backwardGaussSeidel:
            z = backwardSparse(A, r)
        elif iterMatrix == IterMatrix.symmetricGaussSeidel:
            y = forwardSparse(A, r)
            z = backwardSparse(A, D.elementWiseMult(y))

        x = x + z

        if delta < (tolerance * deltaInit):
            print("Convergence")
            print(f"Accuracy: {delta / deltaInit}")
            return [x, k, delta]

    print("Max iter reached")
    print(f"Accuracy: {delta / deltaInit}")

    return [x, maxIter, delta]
