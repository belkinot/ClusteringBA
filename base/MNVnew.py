from base.helperfunctions import *
import networkx as nx
import time

np.set_printoptions(threshold=np.inf)

def MNV_algorithm(dataset, nearest=5):
    start_time = time.time()

    #neighbourhoodmatrix = []
    nb = dict()
    for i, pts in enumerate(dataset):
        neighbour = [[mydist(pts,pts2), idx] for idx, pts2 in enumerate(dataset)]
        neighbour.sort()
        #print(neighbour)
        nb[i] = [x[1] for x in neighbour[:11]]
        #neighbourhoodmatrix += [nb]
    #for i in range(dataset):
     #   for j in range(i+1, dataset):
      #      pts1 = dataset[i]
       #     pts2 = dataset[j]
        #    nb = [[]]

    #matrix mnv values
    #print(neighbourhoodmatrix[0])
    #print(nb)
    m2 = [[0 for _ in range(len(dataset))] for _ in range(len(dataset))]
    erg = []

    for index in range(len(dataset)):
        for mnv1 in nb[index]:
            index1 = nb[index].index(mnv1)
            if index in nb[mnv1]:
                wo = nb[mnv1].index(index)
                mnv = wo + index1
                if 2 <= mnv <= 10:
                    erg += [(index, mnv1)]

    #for line in m2:
    #    print(line)
    labels = gen_labels(len(dataset),erg)
    #admatrix = np.matrix(m2)
    #print(admatrix)
    #g = nx.from_numpy_matrix(admatrix)
    #labels = list(nx.connected_components(g))
    #print('LABELS', labels)
    #return labels
    #results, _ = gen_results_from_labels(dataset, labels)
    #print('ZEIT' , time.time() - start_time)
    #showres(results)
    return labels