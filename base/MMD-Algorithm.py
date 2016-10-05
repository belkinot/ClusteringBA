import networkx as nx
import numpy as np
from base.helperfunctions import *


def noniterative_clustering(data):

    #matrix n*n wobei distanzen zu allen punkten enthalten sind --> "noise" wird die distanz auf beliebig großen wert gesetzt
    # matrix(i,j) = distanz zwischen i und j
    mindist = []
    idxset = {i for i in range(len(data))}
    noiseset = set()
    #for i in combinations(range(len(data)), 2):
    #    mindist += [mydist(data[i[0]], data[i[1]])]
    for i in range(len(data)):
        temp = []
        for j in range(len(data)):
            temp += [mydist(data[i], data[j])]
        mindist += [temp]

    MMD = noisedetect(mindist, idxset, noiseset)
    print(MMD)
    adjacency_matrix = []
    for idx, distance in enumerate(mindist):
            adjacency_matrix_help = []
            for dist in distance:
                if dist <= (2* MMD):
                    adjacency_matrix_help.append(1)
                else:
                    adjacency_matrix_help.append(0)
            adjacency_matrix.append(adjacency_matrix_help)
    #adjacency_matrix_test = np.array(adjacency_matrix)
    adjacency_matrix_np = np.matrix(adjacency_matrix)
    print(adjacency_matrix_np)
    graph = nx.from_numpy_matrix(adjacency_matrix_np)
    return adjacency_matrix_np
    #graph plotten richtung datensatz
    clusteringlabels = list(nx.connected_components(graph))
    clustering, noiseclusters = gen_results_from_labels(data,clusteringlabels)
    print('labellength', len(clusteringlabels))
    print("LABELS: ", clusteringlabels)
    showres(clustering, noiseclusters)
    with open('test.out3', "a") as f:
        print(clusteringlabels, file=f)
    #v-measure

def noisedetect(distmatrix, index_set, noise_set):
    prevlen = len(index_set)+1

    while len(index_set) < prevlen:
        MMD = 0
        for idx, example in enumerate(distmatrix):
            if idx in index_set:
                MMD = MMD + min(i for i in example if i > 0)
        MMD = MMD/len(index_set)
        for idx, example in enumerate(distmatrix):
            if min(i for i in example if i > 0) > 2* MMD and idx in index_set:
                index_set.remove(idx)
                noise_set.add(idx)
        prevlen = len(index_set)
    return MMD