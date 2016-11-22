import numpy as np
from scipy.spatial import Delaunay
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import make_blobs, make_moons, make_circles

from base.helperfunctions import *
from base.MNVnew import mnv_algorithm
from base.MMD import noniterative_clustering
from base.Delaunay_mimic import Delaunay_mimic
from base.Delaunay_reduce import Delaunay_reduce
from base.DBScan import DBSCANlabels_to_Labels, KMEANSlabels_to_Labels
import random

import time

def cluster_analysis_fixed_dataset(features, n_clust):
    """Cluster-analyse auf einem bereits festgelegten Datensatz"""
    start = time.time()
    sfeature = str(features)
    scluster = str(n_clust)
    params = reader('Results/'+ sfeature + ' Dim ' + scluster + ' Clust/dataset_f' + sfeature + 'c' + scluster + '.out')
    result_kmeans = []
    for i in range(1000):
        samples = params[i][0]
        std = params[i][1]
        rd_state = params[i][2]
        DATASET, truelabels = make_blobs(n_samples=samples, n_features=features, centers=n_clust, cluster_std=std,
                                         random_state=rd_state)
        y_pred = KMeans(n_clust).fit_predict(DATASET)
        labels_kmeans = KMEANSlabels_to_Labels(y_pred)
        spatial_sep_kmeans = spatial_separation(DATASET, labels_kmeans)
        sse_kmeans = sse_per_label(DATASET, labels_kmeans)
        rand_index_kmeans = adjusted_rand_score(truelabels, y_pred)


        result_kmeans += [(labels_kmeans, spatial_sep_kmeans, sse_kmeans,
                           rand_index_kmeans)]
        print(i)
        print('Zeit', time.time() - start)
    with open('result_kmeanslabels_f' + sfeature + 'c' + scluster + '.out', 'a') as km:
        print(result_kmeans, file=km)



def cluster_analysis(features, n_clust):
    """cluster-analyse mit 5 algorithmen"""
    sfeature = str(features)
    scluster = str(n_clust)
    starter = time.time()
    for i in range(10):
        params = []
        result_mimic = []
        result_reduce2 = []
        result_mmd = []
        result_mnv = []
        result_dbscan = []
        #result_kmeans = []

        for j in range(100):
            samples = random.randint(100, 1000)
            std = random.uniform(0.5, 2)
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
            #connectedness_mimic = connectedness(create_mst_weighted(DATASET, labels))
            #connectedness2 = connectedness(create_mst_weighted(DATASET,labels))

            result_mimic += [(labels, spatial_sep, sse, rand_index)]
            result_reduce2 += [(labels2, spatial_sep2, sse2, rand_index2)]

            # MMD - Algorithm
            labels_mmd = noniterative_clustering(DATASET)
            spatial_sep_mmd = spatial_separation(DATASET, labels_mmd)
            sse_mmd = sse_per_label(DATASET, labels_mmd)
            rand_index_mmd = adjusted_rand_score(truelabels, get_rand_index_format(DATASET, labels_mmd))
            #connectedness_mmd = connectedness(create_mst_weighted(DATASET, labels_mmd))

            result_mmd += [(labels_mmd, spatial_sep_mmd, sse_mmd, rand_index_mmd)]

            # MNV-Algorithm
            labels_mnv = mnv_algorithm(DATASET)
            spatial_sep_mnv = spatial_separation(DATASET, labels_mnv)
            sse_mnv = sse_per_label(DATASET, labels_mnv)
            rand_index_mnv = adjusted_rand_score(truelabels, get_rand_index_format(DATASET, labels_mnv))
            #connectedness_mnv = connectedness(create_mst_weighted(DATASET, labels_mnv))

            result_mnv += [(labels_mnv, spatial_sep_mnv, sse_mnv, rand_index_mnv)]

            # DBSCAN
            labels_dbscan = DBSCANlabels_to_Labels(DBSCAN().fit_predict(X=DATASET))
            spatial_sep_dbscan = spatial_separation(DATASET, labels_dbscan)
            sse_dbscan = sse_per_label(DATASET, labels_dbscan)
            rand_index_dbscan = adjusted_rand_score(truelabels, get_rand_index_format(DATASET, labels_dbscan))
            #connectedness_dbscan, mst_dbscan = create_mst(DATASET, labels_dbscan)

            result_dbscan += [(labels_dbscan, spatial_sep_dbscan, sse_dbscan, rand_index_dbscan)]
            """
            # KMEANS- selfimplemented
            labels_kmeans = kmeans(DATASET, n_clust)
            spatial_sep_kmeans = spatial_separation(DATASET, labels_kmeans)
            sse_kmeans = sse_per_label(DATASET, labels_kmeans)
            rand_index_kmeans = adjusted_rand_score(truelabels, get_rand_index_format(DATASET,labels_kmeans))
            #connectedness_kmeans, mst_kmeans = create_mst(DATASET, labels_kmeans)

            result_kmeans += [(labels_kmeans,spatial_sep_kmeans, sse_kmeans, rand_index_kmeans)]
            """
            params += [(samples, std, rd_state)]

            print('Datensatz', j, 'reihe', i)
            print('Zeit', time.time()-starter)
        with open('dataset_f' + sfeature + 'c' + scluster + '.out', mode='a') as f:
            print(params, file=f)
        with open('result_dmimiclabels_f'+ sfeature + 'c' + scluster + '.out', 'a') as ff:
            print(result_mimic, file=ff)
        with open('result_dreduclabels_f' + sfeature + 'c' + scluster + '.out', 'a') as fff:
            print(result_reduce2, file=fff)
        with open('result_mmdlabels_f' + sfeature + 'c' + scluster + '.out', 'a') as ffff:
            print(result_mmd, file=ffff)
        with open('result_mnvlabels_f' + sfeature + 'c' + scluster + '.out', 'a') as fffff:
            print(result_mnv, file=fffff)
        with open('result_dbscanlabels_f' + sfeature + 'c' + scluster + '.out', 'a') as dbs:
            print(result_dbscan, file=dbs)
        #with open('result_kmeanslabels_f' + sfeature + 'c' + scluster + '.out' , 'a') as km:
        #    print(result_kmeans, file=km)

        # boxplots median obere/untere hälfte
        # pseudo code -> nachimplementieren möglich , gemäß gleichung 7 etc


def boxplot_result(feature, clust):
    sfeature = str(feature)
    sclust = str(clust)
    res_kmean = reader('Results/' +sfeature + ' Dim ' + sclust + ' Clust/result_kmeanslabels_f'+sfeature+'c'+sclust+'.out')
    res_dbscan = reader('Results/' +sfeature + ' Dim ' +sclust + ' Clust/result_dbscanlabels_f'+sfeature+'c'+sclust+'.out')
    res_mmd = reader('Results/' +sfeature + ' Dim ' +sclust + ' Clust/result_mmdlabels_f'+sfeature+'c'+sclust+'.out')
    res_mnv = reader('Results/' +sfeature + ' Dim ' +sclust + ' Clust/result_mnvlabels_f'+sfeature+'c'+sclust+'.out')
    res_dmimic = reader('Results/' +sfeature + ' Dim ' +sclust + ' Clust/result_dmimiclabels_f'+sfeature+'c'+sclust+'.out')
    res_dreduce = reader('Results/' +sfeature + ' Dim ' +sclust + ' Clust/result_dreduclabels_f'+sfeature+'c'+sclust+'.out')

    res_data_kmean = evaluate(res_kmean, 1)
    res_data_dbscan = evaluate(res_dbscan, 1)
    res_data_mmd = evaluate(res_mmd, 1)
    res_data_mnv = evaluate(res_mnv, 1)
    res_data_dmimic = evaluate(res_dmimic, 1)
    res_data_dreduce = evaluate(res_dreduce, 1)

    plotdata = [res_data_kmean, res_data_dbscan, res_data_mmd, res_data_mnv, res_data_dmimic, res_data_dreduce]
    plt.figure()
    plt.boxplot(plotdata)
    plt.show()
#spatial separation abstand kürzester aus sunterschiedlichen clustern
#connectedness lokale nachbarschaft maximaler abstand dbscan minpts --> connectmaß

def evaluate(data, idx):
    """gebe Ergebnisse für ein bestimmtes Maß aus den Textdateien aus"""
    res_data = []
    for i in range(len(data)):
        res_data += [data[i][idx]] # welches Kriterium soll evaluiert werden (0 - labels, 1 spatial_sep, 2 sse, 3 randscore, 4 wenn vorhanden connectedness
    return res_data


def connectivity_calc(features, n_clust):
    sfeature = str(features)
    scluster = str(n_clust)
    data = reader('Results/'+sfeature+ ' Dim '+ scluster + ' Clust/dataset_f' +sfeature +'c'+ scluster+ '.out')
    data_mmd = reader('Results/' +sfeature+ ' Dim '+ scluster + ' Clust/result_mmdlabels_f' +sfeature +'c'+ scluster+ '.out')
    data_mnv = reader('Results/'+sfeature+ ' Dim '+ scluster + ' Clust/result_mnvlabels_f' +sfeature +'c'+ scluster+ '.out')
    data_dmimic = reader('Results/'+sfeature+ ' Dim '+ scluster + ' Clust/result_dmimiclabels_f' +sfeature +'c'+ scluster+ '.out')
    data_dreduc = reader('Results/'+sfeature+ ' Dim '+ scluster + ' Clust/result_dreduclabels_f' +sfeature +'c'+ scluster+ '.out')
    data_dbscan = reader('Results/'+sfeature+ ' Dim '+ scluster + ' Clust/result_dbscanlabels_f' +sfeature +'c'+ scluster+ '.out')
    data_kmeans = reader('Results/'+sfeature+ ' Dim '+ scluster + ' Clust/result_kmeanslabels_f'+sfeature+'c'+scluster+'.out')
    connectres = []
    for i in range(len(data)):
        DATASET, truelabels = make_blobs(n_samples=data[i][0], n_features=features, centers=n_clust, cluster_std=data[i][1], random_state=data[i][2])
        labels_mmd = data_mmd[i][0]
        labels_mnv = data_mnv[i][0]
        labels_dmimic = data_dmimic[i][0]
        labels_dreduc = data_dreduc[i][0]
        labels_dbscan = data_dbscan[i][0]
        labels_kmeans = data_kmeans[i][0]
        labels_true = KMEANSlabels_to_Labels(truelabels)

        connect_true = connectedness(create_mst_weighted(DATASET, labels_true))
        connect_mmd = connectedness(create_mst_weighted(DATASET, labels_mmd))
        connect_mnv = connectedness(create_mst_weighted(DATASET, labels_mnv))
        connect_dmimic = connectedness(create_mst_weighted(DATASET, labels_dmimic))
        connect_dreduc = connectedness(create_mst_weighted(DATASET, labels_dreduc))
        connect_dbscan = connectedness(create_mst_weighted(DATASET, labels_dbscan))
        connect_kmeans = connectedness(create_mst_weighted(DATASET, labels_kmeans))
        connectres += [(connect_true, connect_mmd, connect_mnv, connect_dmimic, connect_dreduc, connect_dbscan, connect_kmeans)]
        print(i)
    with open('connectedness_f'+sfeature+'clust'+scluster+'.out', 'a') as f:
        print(connectres, file=f)
#compactness sse dbsscan- kmeans
#uniform density - radius um jeden punkt in einem cluster, anzahl der punkt in dem radius sollte einigermaßen gleich sein

if __name__ == '__main__':
    #dataset = np.random.uniform(0, 10, size=(40, 2))
    start_time = time.time()
    boxplot_result(2,3)

    #cluster_analysis_fixed_dataset(4,2)
    #cluster_analysis_fixed_dataset(4,3)
    #cluster_analysis_fixed_dataset(4,4)
    #cluster_analysis_fixed_dataset(4,5)
    #cluster_analysis_fixed_dataset(4,6)
    #cluster_analysis_fixed_dataset(4,7)
    #connectivity_calc(3,2)
    #connectivity_calc(3,3)
    #connectivity_calc(3,4)
    #connectivity_calc(3,5)
    #connectivity_calc(3,6)
    #cluster_analysis(2,2)
    """
    res = reader('Results/2 Dim 3 Clust/dataset_f2c3.out')
    print(len(res))
    res2 = reader('Results/2 Dim 3 Clust/result_mmd_f2c3.out')
    data2 = []
    for i in range(len(res2)):
        data2 += [res2[i][1]]

    data = []
    for i in range(len(res)):
        data += [res[i][1]]
    sum = 0
    for i in data:
        sum += i
    sum = sum/len(data)
    print(sum)
    print(max(data))
    print(min(data))
    plotdate = [data,data2]
    plt.figure()
    plt.boxplot(data)
    plt.show()

    """
    #cluster_analysis(3,7)


    #DATASET = load_mydataset()

    #labels = kmeans(DATASET,2)
    #connectedness, mst = create_mst(DATASET, labels)
    #mst2 = create_mst_weighted(DATASET, labels)

    #print(connectedness)
    #print(type(mst[0]))

    #longest = 0
    #for i in mst2:
    #    if graph_longest_edge(i) > longest:
    #        longest = graph_longest_edge(i)
    #print(longest)

 #   dataset2 = load_mydataset(True)
    #DATASET, truelabels = make_circles(1000,noise=0.06, factor= 0.5)
    #DATASET, truelabels = make_moons(500, noise=0.06)
    #DATASET, truelabels = make_blobs(1000, 2, 7, cluster_std=0.8, random_state=5)
    #mnv_algorithm(DATASET)
    #noniterative_clustering(DATASET)
    #tri = Delaunay(DATASET)
    #Delaunay_mimic(DATASET, 1.65, tri)
    #labels = DBSCAN().fit_predict(DATASET)
    #print(adjusted_rand_score(truelabels, labels))
    #labels = kmeans(DATASET, 5)
    #y_pred = KMeans(5).fit_predict(DATASET)
    #print(y_pred)
    #print(KMEANSlabels_to_Labels(DATASET,y_pred))

    #mylist, mymst=  create_mst(DATASET, kmeans(DATASET,2))
    #mylist, mymst = create_mst(DATASET, MNV_algorithm(DATASET))
    #longest = []
    #for i in mst:
    #    graph = nx.from_scipy_sparse_matrix(i)
        #print(degree_connectedness(graph))
    #    longest += [graph_longest_edge(graph)]
    #print(longest)
    #print(mylist)
    #print(degree_connectedness(graph))
    #print(degree_connectedness(noniterative_clustering(DATASET, 2)))




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
