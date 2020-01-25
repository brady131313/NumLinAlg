import matrix

def forward(A, b):
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
    if A.rows != b.dim:
        raise Exception(f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")

    diagonal = A.data[0]
    data = [0] * A.rows
    for i in range(A.rows):
        data[i] = b.data[i]

        for k in range(A.rowPtr[i], A.rowPtr[i + 1] - 1):
            j = A.colInd[k]
            diagonal = A.data[k + 1] if i == j + 1 else 1

            data[i] = data[i] - (A.data[k] * data[j])

        data[i] = data[i] / diagonal

    return matrix.Vector(A.rows, data) 

def backward(A, b):
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
    if A.rows != b.dim:
        raise Exception(f"Dimension mismatch, matrix is {A.rows}x{A.columns}, vector is {b.dim}x1")
    
    diagonal = A.data[len(A.data) - 1]
    data = [0] * A.rows
    for i in range(A.rows - 1, -1, -1):
        data[i] = b.data[i]

        for k in range(A.rowPtr[i] + 1, A.rowPtr[i + 1]):
            j = A.colInd[k]
            diagonal = A.data[k - 1] if i == j - 1 else 1
        
            data[i] = data[i] - (A.data[k] * data[j])

        data[i] = data[i] / diagonal

    return matrix.Vector(A.rows, data)