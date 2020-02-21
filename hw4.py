import argparse

import matrix
import graph
import cluster
import util

def hw4(filename, r, tau, p):
    with open(util.getMatrixFile(filename)) as file:
        g = graph.Graph.fromFile(file)

    w = g.getRandomWeights()

    print("-- SINGLE --")
    clusters = cluster.lubys(g.edgeVertex, w)

    P = graph.getVertexAggregate2(g.adjacency.rows, clusters)
    coarse = graph.formCoarse(P, g.adjacency)

    efficiency = (g.adjacency.rows / 2) / len(clusters)

    if p:
        P.visualizeShape()
        print(coarse)
    print(f"{len(clusters)} clusters, {efficiency} efficiency")

    if not r: return

    print("\n-- RECURSIVE --")
    
    As, Ps = cluster.recursiveLubys(g, tau)
    print(len(As), len(Ps))

    k = len(Ps)
    Pk = graph.formVertexToK1Aggregate(Ps, k)

    (Pk.transpose().multMat(As[0]).multMat(Pk)).visualizeShape()
    As[k].visualizeShape()



parser = argparse.ArgumentParser()
parser.add_argument("filename", help="graph to be clustered", type=str)
parser.add_argument("-r", dest="r", action='store_true', help="Recursive Lubys")
parser.add_argument("-t", dest="tau", default=8, action='store', type=int, help="Tau shrinkige")
parser.add_argument("-p", dest="p", action='store_true', help="Display matrix")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("graph must be supplied")
else:
    hw4(args.filename, args.r, args.tau, args.p)