from base.boxplot import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import networkx as nx
from scipy.stats import multivariate_normal
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs, make_moons

from base.helperfunctions import *
from base.MNVnew import *
from base.MMD import *
from base.MNV import *
from base.kmeans import *
from base.Delaunay_mimic import *
from base.Delaunay_reduce import *
from base.DBScan import DBSCANlabels_to_Labels
from itertools import combinations

import time


def cluster_analysis(features, n_clust):
    sfeature = str(features)
    scluster = str(n_clust)
    # parameter = reader('testen8dim2cluster.out')
    starter = time.time()
    # print(parameter[0][0])
    params = []
    result_mimic = []
    result_reduce2 = []
    result_mmd = []
    result_mnv = []
    result_dbscan = []
    result_kmeans = []

    for i in range(1000):
        # samples = parameter[i][0]
        samples = random.randint(100, 1000)
        # std = parameter[i][1]
        std = random.randint(1, 10)
        # rd_state = parameter[i][2]
        rd_state = random.randint(1, 100000)
        DATASET, truelabels = make_blobs(n_samples=samples, n_features=features, centers=n_clust, cluster_std=std,
                                         random_state=rd_state)
        #delaunays algorithm
        triangulat = Delaunay(DATASET)
        labels = Delaunay_mimic(DATASET, 1.65, triangulat)
        labels2 = Delaunay_reduce(DATASET, triangulat)
        spatial_sep = spatial_separation(DATASET, labels)
        spatial_sep2 = spatial_separation(DATASET, labels2)
        sse = sse_per_label(DATASET, labels)
        sse2 = sse_per_label(DATASET, labels2)
        rand_index = adjusted_rand_score(truelabels, get_rand_index_format(DATASET, labels))
        rand_index2 = adjusted_rand_score(truelabels, get_rand_index_format(DATASET, labels2))
        connectedness = create_mst(DATASET, labels)
        connectedness2 = create_mst(DATASET, labels2)

        result_mimic += [(spatial_sep, sse, rand_index, connectedness)]
        result_reduce2 += [(spatial_sep2, sse2, rand_index2, connectedness2)]

        # MMD - Algorithm
        labels_mmd = noniterative_clustering(DATASET)
        spatial_sep_mmd = spatial_separation(DATASET, labels_mmd)
        sse_mmd = sse_per_label(DATASET, labels_mmd)
        rand_index_mmd = adjusted_rand_score(truelabels, get_rand_index_format(DATASET, labels_mmd))
        connectedness_mmd = create_mst(DATASET, labels_mmd)

        result_mmd += [(spatial_sep_mmd, sse_mmd, rand_index_mmd, connectedness_mmd)]

        # MNV-Algorithm
        labels_mnv = MNV_algorithm(DATASET)
        spatial_sep_mnv = spatial_separation(DATASET, labels_mnv)
        sse_mnv = sse_per_label(DATASET, labels_mnv)
        rand_index_mnv = adjusted_rand_score(truelabels, get_rand_index_format(DATASET, labels_mnv))
        connectedness_mnv = create_mst(DATASET, labels_mnv)

        result_mnv += [(spatial_sep_mnv, sse_mnv, rand_index_mnv, connectedness_mnv)]

        # DBSCAN
        labels_dbscan = DBSCANlabels_to_Labels(DATASET,DBSCAN().fit_predict(X=DATASET))
        spatial_sep_dbscan = spatial_separation(DATASET, labels_dbscan)
        sse_dbscan = sse_per_label(DATASET, labels_dbscan)
        rand_index_dbscan = adjusted_rand_score(truelabels, get_rand_index_format(DATASET,labels_dbscan))
        connectedness_dbscan = create_mst(DATASET, labels_dbscan)

        result_dbscan += [(spatial_sep_dbscan, sse_dbscan, rand_index_dbscan, connectedness_dbscan)]

        # KMEANS- selfimplemented
        labels_kmeans = kmeans(DATASET, n_clust)
        spatial_sep_kmeans = spatial_separation(DATASET, labels_kmeans)
        sse_kmeans = sse_per_label(DATASET, labels_kmeans)
        rand_index_kmeans = adjusted_rand_score(truelabels, get_rand_index_format(DATASET,labels_kmeans))
        connectedness_kmeans = create_mst(DATASET, labels_kmeans)

        result_kmeans += [(spatial_sep_kmeans, sse_kmeans, rand_index_kmeans, connectedness_kmeans)]

        params += [(samples, std, rd_state)]

        print('Datensatz', i)
        print('Zeit' , time.time()-starter)
    with open('dataset_f' + sfeature + 'c' + scluster + '.out', 'a') as f:
           print(params, file=f)
    with open('result_dmimic_f'+ sfeature + 'c' + scluster + 'out', 'a') as ff:
        print(result_mimic, file=ff)
    with open('result__dreduc_f' + sfeature + 'c' + scluster + '.out', 'a') as fff:
        print(result_reduce2, file=fff)
    with open('result_mmd_f' + sfeature + 'c' + scluster + '.out' , 'a') as ffff:
        print(result_mmd, file=ffff)
    with open('result_mnv_f' + sfeature + 'c' + scluster + '.out', 'a') as fffff:
        print(result_mnv, file=fffff)
    with open('result_dbscan_f' + sfeature + 'c' + scluster + '.out','a') as dbs:
        print(result_dbscan, file=dbs)
    with open('result_kmeans_f' + sfeature + 'c' + scluster + '.out' , 'a') as km:
        print(result_kmeans, file=km)

        # boxplots median obere/untere hälfte
        # pseudo code -> nachimplementieren möglich , gemäß gleichung 7 etc




#spatial separation abstand kürzester aus sunterschiedlichen clustern
#connectedness lokale nachbarschaft maximaler abstand dbscan minpts --> connectmaß


 # längste kante im Graph / MST


#compactness sse dbsscan- kmeans
#uniform density - radius um jeden punkt in einem cluster, anzahl der punkt in dem radius sollte einigermaßen gleich sein

if __name__ == '__main__':
    #dataset = np.random.uniform(0, 10, size=(40, 2))
    start_time = time.time()

    #cluster_analysis(2,2)
    cluster_analysis(2,3)
    cluster_analysis(2,4)
    cluster_analysis(2,5)
    cluster_analysis(2,6)
    cluster_analysis(2,7)

    #cluster_analysis(3,2)
    #cluster_analysis(3,3)
    #cluster_analysis(3,4)
    #cluster_analysis(3,5)
    #cluster_analysis(3,6)
    #cluster_analysis(3,7)
    #DATASET = load_mydataset()
 #   dataset2 = load_mydataset(True)
    #DATASET, truelabels = make_moons(1000, noise=0.06)
    #DATASET, truelabels = make_blobs(1000,8,5)
    #mylist, mymst=  create_mst(DATASET, kmeans(DATASET,2))
    #mylist, mymst = create_mst(DATASET, MNV_algorithm(DATASET))
    #longest = []
    #for i in mymst:
     #   graph = nx.from_scipy_sparse_matrix(i)
     #   print(degree_connectedness(graph))
    #    longest += [graph_longest_edge(graph)]
    #print(longest)
    #print(mylist)
    #print(degree_connectedness(graph))
    #print(degree_connectedness(noniterative_clustering(DATASET, 2)))
    #boxplot()
    #Delaunay_mimic(DATASET, 2)
    #Delaunay_reduce(DATASET)

    #MNV_algorithm(DATASET)
    #noniterative_clustering(DATASET,1.5)


    #label_liste = Delaunay_reduce(DATASET)
    #print(get_rand_index_format(DATASET, label_liste))
    #print(DATASET)
    #print(truelabels)
    #y_pred = DBSCAN().fit_predict(X=DATASET)
    #print(y_pred)
    #print('0', DBSCANlabels_to_Labels(DATASET, y_pred)[0])
    #print(y_pred[0])
    #print(DBSCANlabels_to_Labels(DATASET, y_pred))
    #label_liste = kmeans(DATASET, 2)
    #labels = Delaunay_reduce(DATASET)
    #labels = MNV_algorithm(DATASET)
    #labels = noniterative_clustering(DATASET)
    #print(label_liste)
    #print(y_pred)
    #print(type(y_pred))
    #print(type(label_liste))
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
