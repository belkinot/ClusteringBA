from base.helperfunctions import *


def noniterative_clustering(data, MMD_parameter=2):

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

    MMD = noisedetect(mindist, idxset, noiseset, MMD_parameter)
    #print(MMD)
    adjacency_matrix = []
    for idx, distance in enumerate(mindist):
            adjacency_matrix_help = []
            for dist in distance:
                if dist <= (MMD_parameter* MMD): #MMD-Parameter
                    adjacency_matrix_help.append(1)
                else:
                    adjacency_matrix_help.append(0)
            adjacency_matrix.append(adjacency_matrix_help)
    adjacency_matrix_np = np.matrix(adjacency_matrix)
    #print(adjacency_matrix_np)
    graph = nx.from_numpy_matrix(adjacency_matrix_np)
    #graph plotten richtung datensatz
    clusteringlabels = list(nx.connected_components(graph))
    clustering, noiseclusters = gen_results_from_labels(data,clusteringlabels)
    #print('labellength', len(clusteringlabels))
    #print("LABELS: ", clusteringlabels)
    #print('CLUSTERInG' ,clustering)
    #return graph
    #return clusteringlabels


    #showres(clustering, noiseclusters)
    return clusteringlabels
    """
    with open('test2.out', "a") as f:
        print(clusteringlabels, file=f)
    #v-measure
    with open('test2_matrix.out', "a") as f:
        print(mindist, file=f)
    degree_connectedness(graph)
    """

def noisedetect(distmatrix, index_set, noise_set, MMD_parameter):
    """Erkennt Noise und entfernt diese aus dem Datensatz"""
    counter = 0
    while True:
        MMD = 0
        for idx, example in enumerate(distmatrix):
            if idx in index_set and example:
                MMD = MMD + min(i for i in example if i > 0)
        MMD = MMD/len(index_set)
        for idx, example in enumerate(distmatrix):
            if min(i for i in example if i > 0) > MMD_parameter* MMD and idx in index_set: #MMD-Parameter
                index_set.remove(idx)
                noise_set.add(idx)
                counter += 1
        if counter == 0:
            break
        #print(counter, "COUNTER")
        counter = 0

    return MMD