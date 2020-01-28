import os

def getMatrixFile(name):
    scriptDir = os.path.dirname(__file__)
    relDir = "samples/" + name
    filePath = os.path.join(scriptDir, relDir)
    return filePath

def compareVectors(v1, v2):
    for i in range(v1.dim):
        print(f"{round(v1.data[i], 3)} : {round(v2.data[i], 3)}")