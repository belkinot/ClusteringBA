"""MNV-Algorithmus"""
from base.helperfunctions import mydist, gen_labels, gen_results_from_labels, showres
#import time



def mnv_algorithm(dataset, nearest=5):
    """MNV-Algorithm von Gowda/Krishna"""

    #start_time = time.time()

    nb = dict()
    for i, pts in enumerate(dataset):
        neighbour = [[mydist(pts, pts2), idx] for idx, pts2 in enumerate(dataset)]
        neighbour.sort() # sortierte Nachbarn
        nb[i] = [x[1] for x in neighbour[:(2*nearest)+1]] # k-Nachbarn zu Punkt i
    erg = []

    for index in range(len(dataset)):
        for mnv1 in nb[index]:
            index1 = nb[index].index(mnv1)
            if index in nb[mnv1]:
                wo = nb[mnv1].index(index)
                mnv = wo + index1
                if 0 <= mnv <= 2*nearest:
                    erg += [(index, mnv1)] # Tupel aus Punkten die unter der Bedingung dass ihr mnv <= 2* nearest ist

    labels = gen_labels(len(dataset), erg) # generiere Cluster
    #print('LABELS', labels)
    #results, _ = gen_results_from_labels(dataset, labels)
    #print('ZEIT' , time.time() - start_time)
    #showres(results)
    return labels
