import matrix

data1 = [[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]]

data2 = [[2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]

d1 = matrix.Dense(4, 3, data1)
d2 = matrix.Dense(3, 4, data2)

print(d1)
print(d2)
print(d1.multMat(d2))
