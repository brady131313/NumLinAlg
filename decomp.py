import copy

import matrix

def QR(A):
    if not isinstance(A, matrix.Dense):
        raise Exception("Matrix A must be dense")

    Q = matrix.Dense(A.rows, A.columns)
    R = matrix.Dense(A.columns, A.columns)

    vecs = []
    for i in range(A.columns):
        vecs.append(A.getVector(i))

    for i in range(A.columns):
        R.data[i][i] = vecs[i].norm()
        Q.setVector(i, vecs[i].scale(1 / R.data[i][i]))

        for j in range(i + 1, A.columns):
            R.data[i][j] = Q.getVector(i).dot(vecs[j])
            vecs[j] = vecs[j] - Q.getVector(i).scale(R.data[i][j])
    
    return [Q, R]

def LDL(A):
    if not isinstance(A, matrix.Dense):
        raise Exception("Matrix A must be dense")

    A = copy.deepcopy(A)
    L = matrix.Dense(A.rows, A.columns)
    D = matrix.Vector(A.rows)

    for i in range(A.rows):
        L.data[i][i] = 1
        a = A.data[i][i]

        for j in range(i):
            a = a - D.data[j] * L.data[i][j] * L.data[i][j]
        D.data[i] = a

        if D.data[i] == 0:
            raise Exception("A is no LDL decomposition")

        for j in range(i + 1, A.rows):
            a = A.data[j][i]

            for k in range(j):
                a = a - D.data[k] * L.data[j][k] * L.data[i][k]
            L.data[j][i] = a / D.data[i]

    return [L, D]

def forwardGaussSeidel(A):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns:
        raise Exception("Matrix A must be square and SPD")

    B = matrix.Sparse(A.rows, A.columns)
    nnz = 0

    for i in range(A.rows):
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]

            if i >= j:
                B.data.append(A.data[k])
                B.colInd.append(j)
                nnz += 1
        
        B.rowPtr.append(nnz)

    return B

def backwardGaussSeidel(A):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns:
        raise Exception("Matrix A must be square and SPD")

    B = matrix.Sparse(A.rows, A.columns)
    nnz = 0

    for i in range(A.rows):
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]

            if i <= j:
                B.data.append(A.data[k])
                B.colInd.append(j)
                nnz += 1

        B.rowPtr.append(nnz)

    return B

def symmetricGaussSeidel(A):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns:
        raise Exception("Matrix A must be square and SPD")

    L = matrix.Sparse(A.rows, A.columns)
    U = matrix.Sparse(A.rows, A.columns)
    D = matrix.Sparse(A.rows, A.columns)
    nnz = [0] * 3 # nnz[0]: L, nnz[1]: R, nnz[2]: Dinv

    for i in range(A.rows):
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]

            if i >= j:
                L.data.append(A.data[k])
                L.colInd.append(j)
                nnz[0] += 1
            if i <= j:
                U.data.append(A.data[k])
                U.colInd.append(j)
                nnz[1] += 1
            if i == j:
                D.data.append(A.data[k])
                D.colInd.append(j)
                nnz[2] += 1

        L.rowPtr.append(nnz[0])
        U.rowPtr.append(nnz[1])
        D.rowPtr.append(nnz[2])
    
    return [L, D, U]

def l1Smoother(A):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns:
        raise Exception("Matrix A must be square and SPD")

    B = matrix.Sparse(A.rows, A.columns)
    nnz = 0

    for i in range(A.rows):
        sum = 0
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            sum += abs(A.data[k])
        
        B.data.append(sum)
        B.colInd.append(i)
        nnz += 1

        B.rowPtr.append(nnz)

    return B
