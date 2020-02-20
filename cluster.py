import random
import numpy as np

import matrix

def kMeans(X, K, d, maxIter, tolerance):
    centers = random.choices(X, k=K)
    delta = 0
    delatPrev = 0
    
    for i in range(maxIter):
        aggregates = [[] for _ in range(K)]
        
        for x in X:
            closest = _findClosestCenter(x, centers)
            aggregates[closest].append(x)

        centers = _updateCenters(aggregates, d)

        delatPrev = delta
        delta = _computeDelta(X, aggregates, centers)

        if i > 1 and abs(delta - delatPrev) <= tolerance * delatPrev:
            print("Convergence")
            return (aggregates, i, delta)
    
    print("Max iter")
    return (aggregates, maxIter, delta)


def _computeDelta(X, aggregates, centers):
    delta = 0

    for r in range(len(centers)):
        for i in range(len(aggregates[r])):
            delta += (np.linalg.norm(np.subtract(aggregates[r][i], centers[r]))) ** 2

    return delta
        

def _updateCenters(aggregates, d):
    centers = [None for _ in range(len(aggregates))]

    for r in range(len(aggregates)):
        newCenter = np.array([0] * d)
        
        for x in aggregates[r]:
            newCenter = np.add(newCenter, x)
        
        if len(aggregates[r]) == 0:
            raise Exception("Empty Cluster")
        
        newCenter = newCenter * (1 / len(aggregates[r]))
        centers[r] = newCenter
    
    return centers


def _findClosestCenter(x, centers):
    closest = (np.linalg.norm(np.subtract(x, centers[0])), 0)

    for i in range(1, len(centers)):
        update = (np.linalg.norm(np.subtract(x, centers[i])), i)
        if update[0] < closest[0]:
            closest = update
        
    return closest[1]

        