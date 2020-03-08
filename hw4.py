import argparse

from graph.graph import *
from graph import lubys, recursiveLubys, QModularity, modularityMatrix
import util
import plot

def single(filename, O, modularity, printResult, visualize):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file, O)

    if modularity:
        w = modularityWeights(g.adjacency, g.edgeVertex)
    else:
        w = randomWeights(g.edgeVertex)

    P = lubys(g.edgeVertex, w)
    coarse = formCoarse(P, g.adjacency)
    Q = QModularity(g.adjacency, P)

    if printResult:
        P.visualizeShape()
        print(coarse)

    print(f"\nClusters = {P.columns}")
    print(f"Q        = {Q}\n")

    if visualize:
        plot.visualizeGraph(g.edgeVertex, P, w.data)

def recursive(filename, O, tau, modularity, printResult, visualize):
    with open(util.getMatrixFile(filename)) as file:
        g = Graph.fromFile(file, O)

    As, Ps = recursiveLubys(g, tau, modularity)

    Pk = formVertexToK1Aggregate(Ps, len(Ps))
    Q = QModularity(As[0], Pk)


    if printResult:
        Pk.visualizeShape()
        print(As[-1])

    print(f"\nClusters = {Pk.columns}")
    print(f"Levels   = {len(As)}")
    print(f"Q        = {Q}\n")

    if visualize:
        plot.visualizeGraph(g.edgeVertex, Pk)


parser = argparse.ArgumentParser()
parser.add_argument("filename", help="graph to be clustered", type=str)
parser.add_argument("-r", dest="r", action='store_true', help="Recursive Lubys")
parser.add_argument("-t", dest="tau", default=8, action='store', type=int, help="Tau shrinkige")
parser.add_argument("-m", dest="m", action='store_true', help="Use modularity matrix weights")
parser.add_argument("-p", dest="p", action='store_true', help="Display matrix")
parser.add_argument("-v", dest="v", action='store_true', help="Visualize clusters")
parser.add_argument("-O", dest="O", default=0, action='store', type=int, help="Offset for file")
args = parser.parse_args()

if not args.filename or len(args.filename) == 0:
    print("graph must be supplied")
else:
    if args.r:
        recursive(args.filename, args.O, args.tau, args.m, args.p, args.v)
    else:
        single(args.filename, args.O, args.m, args.p, args.v)
