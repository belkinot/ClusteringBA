import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import networkx as nx
from scipy.stats import multivariate_normal
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs, make_moons
from scipy.spatial import Delaunay

from base.helperfunctions import *
from base.MNVnew import *
from base.MMD import *
from base.MNV import *
from base.kmeans import *
from base.Delaunay_mimic import *
from base.Delaunay_reduce import *
from itertools import combinations

import time

#spatial separation abstand kürzester aus sunterschiedlichen clustern
#connectedness lokale nachbarschaft maximaler abstand dbscan minpts --> connectmaß
#compactness sse dbsscan- kmeans
#uniform density - radius um jeden punkt in einem cluster, anzahl der punkt in dem radius sollte einigermaßen gleich sein

if __name__ == '__main__':
    #dataset = np.random.uniform(0, 10, size=(40, 2))
    start_time = time.time()
    #DATASET = load_mydataset()
 #   dataset2 = load_mydataset(True)
    """
    params = []
    for i in range(1000):
        samples = random.randint(100,1000)
        std = random.randint(1,10)
        rd_state = random.randint(1,10000)
        DATASET, truelabels = make_blobs(n_samples=samples, n_features=2, centers=2, cluster_std=std, random_state=rd_state)
        labels = Delaunay_reduce(DATASET)
        spatial_sep = spatial_separation(DATASET,labels)
        params += [samples, std, rd_state, spatial_sep]
    with open('testen.out', 'a') as f:
            print(params, file=f)

    """
    #DATASET, truelabels = make_moons(1000, noise=0.06)
    DATASET, truelabels = make_blobs(1000,2,5)
    #kmeans(DATASET, 5)

    label_liste = Delaunay_reduce(DATASET)
    print(get_rand_index_format(DATASET, label_liste))
    #print(DATASET)
    #print(truelabels)
    y_pred = DBSCAN().fit_predict(X=DATASET)
    #label_liste = kmeans(DATASET, 2)
    #labels = Delaunay_reduce(DATASET)
    #labels = MNV_algorithm(DATASET)
    #labels = noniterative_clustering(DATASET)
    #print(label_liste)
    print(y_pred)
    #label_list = [{} for x in range(max(y_pred)+1)]
    #labellist = []
    #for i in range(len(DATASET)):
    #    label_list[y_pred[i+1]] |= i


    #res = gen_results_from_labels(DATASET, y_pred)
    #showres(res)



    #Delaunay_mimic(DATASET, 2)
    #tri = Triangulation(dataset[:,0], dataset[:,1])
    #tri = Delaunay(dataset)
    #print(tri)
    #MNV_algorithm(dataset)
    #agglomutualnearestneighbour(dataset)
    #kmeans(dataset,2)
    #agglomutualnearestneighbour(dataset)
    #noniterative_clustering(dataset, 2)
    #spatial_separation(dataset)
    #0.0128479387995
    #uniform_density(dataset)
    """
    test = reader('test.out')
    test2 = reader('test2.out')
    ls = [-1 for i in range(len(dataset))]
    ls2 = [-1 for i in range(len(dataset))]
    for idx, i in enumerate(test):
        for j , _ in enumerate(i):
            ls[j] = idx
    for idx, i in enumerate(test2):
        for j, _ in enumerate(i):
            ls2[j] = idx

    print(ls, "LS")
    print("zwote", ls2)

    print(adjusted_rand_score(ls,ls2))
    """
    #dataset = load_dataset_with_labels()
   # plot_true_labels(dataset)
    #codeStandard(dataset)
    #codeSpecial(dataset)
    #agglomutualnearestneighbour(dataset)
    #y_pred = KMeans(2).fit_predict(dataset)
    #y_pred = MeanShift().fit_predict(dataset)
    #sklearn_kmeans_sse(y_pred, dataset)
    #res = reader('k_means.out')
    #showres(res)
    #print(y_pred)
    #noniterative_clustering(dataset)
    #myplots(dataset)
    #sse_calculate()

    print("My Programm took", time.time()-start_time, "to run")
