import os
import time
import functools

def getMatrixFile(name):
    scriptDir = os.path.dirname(__file__)
    relDir = "samples/" + name
    filePath = os.path.join(scriptDir, relDir)
    return filePath

def compareVectors(v1, v2):
    for i in range(v1.dim):
        if i >= 25:
            return
        print(f"{round(v1.data[i], 3)} : {round(v2.data[i], 3)}")

def checkLowerTriangular(A):
    for i in range(A.rows):
        for j in range(A.columns):
            if i < j and A.getValue(i, j) != 0:
                print("Not lower triangular")
                return
    
    print("Lower triangular")

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start

        print(f"Finished {func.__name__!r} in {elapsed:.4f} seconds")
        return value
    return wrapper_timer