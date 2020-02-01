## Homework 1
*Objective*: Given a nxn SPD matrix, first factor it
into A=LDLt where L is a unit lower triangular matrix. U=Lt is a upper triangular matrix. Write an algorithm for forward elimitation (Ly = b), and backward elimination (Ux = y), where L and U are in CSR format.

*Usage*: python hw1.py matrix_file.mtx
matrix files must be placed in the samples directory. An optional -d flag will display a comparison between the found solution vector, and the origion vector.

### Forward Sparse Elimination
*Memory Cost*: 2nnz + 3n + 1, where nnz is number of non zero entries of a CSR matrix, and n is the number of rows and columns of the matrix.

*Operation Cost*: 2nnz - 2n

### Backward Sparse Elimination
*Memory Cost*: 2nnz + 3n + 2, where nnz is number of non zero entries of a CSR matrix, and n is the number of rows and columns of the matrix.

*Operation Cost*: 2nnz - n

## Homework 2
*Objective*: Given a nxn SPD matrix in CSR format, write a stationary iterative method to solve Ax=b.
*Input*: nxn SPD matrix A, rhs vector b, initial iterate x0, max iterations, and tolerance.
*Output*: number of iterations, current iter, residual norm, and accuracy achieved.

*Usage*: python hw2.py [-i MAXITER] [-t TOLERANCE] [-b ITERMATRIX] [-d] [-r] filename
The parameter -b can be l1 for l1Smoother, fgs for forward Gauss Seidel, bgs for backward Gauss Seidel, or sgs for symmetric Gauss Seidel. Additonally -d will display the resulting iter at the end of the algorithm. -r will display the residual at each iteration of the algorithm.
