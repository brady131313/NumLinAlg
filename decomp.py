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

def diagonal(A):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns:
        raise Exception("Matrix A must be square and SPD")

    D = matrix.Vector(A.rows)

    for i in range(A.rows):
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            j = A.colInd[k]

            if i == j:
                D.data[i] = A.data[k]
    
    return D

def l1Smoother(A):
    if not isinstance(A, matrix.Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns:
        raise Exception("Matrix A must be square and SPD")

    b = matrix.Vector(A.rows)

    for i in range(A.rows):
        sum = 0
        for k in range(A.rowPtr[i], A.rowPtr[i + 1]):
            sum += abs(A.data[k])

        b.data[i] = sum

    return b
