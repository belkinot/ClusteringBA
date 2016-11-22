"""MMD-Algorithmus"""
from base.helperfunctions import mydist, np, nx, gen_results_from_labels, showres

def noniterative_clustering(data, mmd_parameter=2):
    """MMD-Algorithmus"""
    mindist = [] # Distanzmatrix
    idxset = {i for i in range(len(data))}
    noiseset = set()
    for i in range(len(data)):
        temp = [] # Zeile der Distanzmatrix
        for j in range(len(data)):
            temp += [mydist(data[i], data[j])]
        mindist += [temp] #Zusammenbau der Distanzmatrix

    mmd = noisedetect(mindist, idxset, noiseset, mmd_parameter) #Berechne MMD, entferne Noise
    adjacency_matrix = []
    for distance in mindist:
        adjacency_matrix_help = []
        for dist in distance:
            if dist <= (mmd_parameter * mmd): # MMD-Parameter
                adjacency_matrix_help += [1]
            else:
                adjacency_matrix_help += [0]
        adjacency_matrix += [adjacency_matrix_help]
    adjacency_matrix_np = np.matrix(adjacency_matrix)
    graph = nx.from_numpy_matrix(adjacency_matrix_np)
    clusteringlabels = list(nx.connected_components(graph))
    #clustering, noiseclusters = gen_results_from_labels(data,clusteringlabels)
    #print('labellength', len(clusteringlabels))
    #print("LABELS: ", clusteringlabels)
    #print('CLUSTERING' ,clustering)
    #showres(clustering, noiseclusters)
    return clusteringlabels

def noisedetect(distmatrix, index_set, noise_set, mmd_parameter):
    """Erkennt Noise und entfernt diese aus dem Datensatz"""
    counter = 0
    while True:
        mmd = 0
        for idx, example in enumerate(distmatrix):
            if idx in index_set:
                mmd = mmd + min(i for i in example if i > 0)
        mmd = mmd/len(index_set)
        for idx, example in enumerate(distmatrix):
            if min(i for i in example if i > 0) > mmd_parameter* mmd and idx in index_set:
                index_set.remove(idx)   # remove point from Set
                noise_set.add(idx)  # add point to noise cluster
                counter += 1
        if counter == 0:
            break   # Abbruchbedingung, wenn kein Punkt mehr als Noise identifiziert wird
        #print(counter, "COUNTER")
        counter = 0

    return mmd
