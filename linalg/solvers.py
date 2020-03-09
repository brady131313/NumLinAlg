from enum import Enum
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
import numpy as np

from matrix import Dense, Sparse, Vector
import graph
from linalg.decomp import l1Smoother, diagonal
import util


def composite(A, b, x, components, residual):
    if not isinstance(A, Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns:
        raise Exception(f"A must be SPD")
    if A.columns != b.dim:
        raise Exception(
            f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector b is {b.dim}x1")

    r = b - A.multVec(x)
    delta = r.norm()

    for component in components:
        y = component(r)
        x = x + y
        r = r - A.multVec(y)
        if residual:
            delta = r.norm()
            print(delta)

    for component in reversed(components):
        y = component(r)
        x = x + y
        r = r - A.multVec(y)
        if residual:
            delta = r.norm()
            print(delta)

    return [x, delta]


def pcg(A, b, x, maxIter, tolerance, preconditioner, displayResidual):
    if not isinstance(A, Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != A.columns or A.columns != b.dim:
        raise Exception(f"A must be SPD")
    if A.columns != b.dim:
        raise Exception(
            f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector b is {b.dim}x1")

    r = b - A.multVec(x)
    pr = preconditioner(r)

    deltaInit = r.dot(pr)
    delta = deltaInit

    p = pr

    for i in range(maxIter):
        g = A.multVec(p)
        alpha = delta / (p.dot(g))

        x = x + p.scale(alpha)
        r = r - g.scale(alpha)
        pr = preconditioner(r)

        deltaOld = delta
        delta = r.dot(pr)

        if displayResidual:
            print(delta)

        if delta < (tolerance ** 2) * deltaInit:
            break

        beta = delta / deltaOld
        p = pr + p.scale(beta)

    return [x, i, delta]


def stationaryIterative(A, b, xInit, maxIter, tolerance, iterSolver, displayResidual):
    if not isinstance(A, Sparse):
        raise Exception("Matrix A must be sparse")

    x = xInit
    deltaInit = 0
    delta = 9999

    for k in range(maxIter):
        r = b - A.multVec(x)
        if k == 0:
            deltaInit = r.norm()

        delta = r.norm()
        if displayResidual:
            print(delta)

        z = iterSolver(r)
        x = x + z

        if delta < (tolerance * deltaInit):
            break

    accuracy = delta / deltaInit
    return [x, k, delta, accuracy]


def forward(A, b):
    if not isinstance(A, Dense):
        raise Exception("Matrix A must be dense")
    if A.rows != b.dim:
        raise Exception(
            f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

    data = [0] * A.rows
    for i in range(A.rows):
        data[i] = b.data[i]

        for j in range(i):
            data[i] = data[i] - (A.data[i][j] * data[j])

        data[i] = data[i] / A.data[i][i]

    return Vector(A.rows, data)


def forwardSparse(A, b):
    if not isinstance(A, Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != b.dim:
        raise Exception(
            f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

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

    return Vector(A.rows, data)


def backward(A, b):
    if not isinstance(A, Dense):
        raise Exception("Matrix A must be dense")
    if A.rows != b.dim:
        raise Exception(
            f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

    data = [0] * A.rows
    for i in range(A.rows - 1, -1, -1):
        data[i] = b.data[i]

        for j in range(i + 1, A.rows):
            data[i] = data[i] - (A.data[i][j] * data[j])

        data[i] = data[i] / A.data[i][i]

    return Vector(A.rows, data)


def backwardSparse(A, b):
    if not isinstance(A, Sparse):
        raise Exception("Matrix A must be sparse")
    if A.rows != b.dim:
        raise Exception(
            f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

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

    return Vector(A.rows, data)


def vectorSolver(a, b):
    if a.dim != b.dim:
        raise Exception(f"Dimension mismatch, {a.dim} != {b.dim}")

    data = [0] * a.dim
    for i in range(a.dim):
        if a.data[i] == 0 or b.data[i] == 0:
            data[i] = 0
        else:
            data[i] = b.data[i] / a.data[i]

    return Vector(a.dim, data)


def l1Solver(A):
    D = l1Smoother(A)

    def solver(r):
        return vectorSolver(D, r)
    return solver


def fgsSolver(A):
    def solver(r):
        return forwardSparse(A, r)
    return solver


def bgsSolver(A):
    def solver(r):
        return backwardSparse(A, r)
    return solver


def sgsSolver(A):
    D = diagonal(A)

    def solver(r):
        y = forwardSparse(A, r)
        return backwardSparse(A, D.elementWiseMult(y))
    return solver


def diagonalPreconditioner(A):
    D = diagonal(A)

    def preconditioner(r):
        return vectorSolver(D, r)
    return preconditioner


def sgsPreconditioner(A):
    D = diagonal(A)

    def preconditioner(r):
        y = forwardSparse(A, r)
        return backwardSparse(A, D.elementWiseMult(y))
    return preconditioner


def twolevelPreconditioner(A, solver):
    E, w = graph.fromAdjacencyToEdge(A, True)

    P = graph.lubys(E, w)
    coarse = graph.formCoarse(P, A)

    def preconditioner(r):
        y = solver(r)

        rc = P.transpose().multVec(r - A.multVec(y))
        yc = _solveCoarse(coarse, rc)

        y = y + P.multVec(yc)
        z = backwardSparse(A, r - A.multVec(y))
        return y + z
    return preconditioner


def _solveCoarse(A, r):
    converted = csr_matrix((A.data, A.colInd, A.rowPtr),
                           (A.rows, A.columns)).asfptype()

    B = splu(converted.tocsc())
    y = B.solve(np.array(r.data))

    y = Vector(y.shape[0], list(y))
    return y
