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

    #TODO fix how diagonal entry is grabbed
    diagonal = A.data[0]
    data = [0] * A.rows
    for i in range(A.rows):
        data[i] = b.data[i]

        for k in range(A.rowPtr[i], A.rowPtr[i + 1] - 1):
            j = A.colInd[k]
            if i == j + 1:
                diagonal = A.data[k + 1]

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
    
    #TODO Fix how diagonal entry is retrieved
    diagonal = A.data[len(A.data) - 1]
    data = [0] * A.rows
    for i in range(A.rows - 1, -1, -1):
        data[i] = b.data[i]

        for k in range(A.rowPtr[i] + 1, A.rowPtr[i + 1]):
            j = A.colInd[k]
            if i == j - 1:
                diagonal = A.data[k - 1]

            data[i] = data[i] - (A.data[k] * data[j])

        data[i] = data[i] / diagonal
        diagonal = 1
    
    return matrix.Vector(A.rows, data)

def stationaryIterative(A, b, xInit, maxIter, tolerance):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")

    x = xInit
    B = decomp.l1Smoother(A)
    deltaInit = 0
    delta = 9999

    for k in range(maxIter):
        r = b - A.multVec(x)
        if k == 0:
            deltaInit = r.norm()
            print(deltaInit)
        
        delta = r.norm()
        print(delta)

        z = backwardSparse(B, r)
        x = x + z

        if delta < (tolerance * deltaInit):
            print("Convergence")
            return x

    print("Max iter reached")
    return x
