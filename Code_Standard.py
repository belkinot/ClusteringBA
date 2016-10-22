import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import networkx as nx
from scipy.stats import multivariate_normal
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

from base.helperfunctions import *
from base.MMD import *
from base.MNV import *
from base.kmeans import *
from itertools import combinations

import time




#spatial separation abstand kürzester aus sunterschiedlichen clustern
#connectedness lokale nachbarschaft maximaler abstand dbscan minpts --> connectmaß
#compactness sse dbsscan- kmeans
#uniform density - radius um jeden punkt in einem cluster, anzahl der punkt in dem radius sollte einigermaßen gleich sein




#plot
def myplots(points):
    x, y =  zip(*points)
    plt.scatter(x, y)
    plt.show()

def plotdemo(points, contour):
    x,y = (zip(*points))
    plt.scatter (x,y, s= 10, c = 'blue', alpha = 1)
    x2,y2 = (zip(*contour))
    plt.scatter(x2, y2, s=10, c='red', alpha=0.7)
    plt.show()

def plotdemo2(data,cluster,noncluster):
    x,y = (zip(*data))
    plt.scatter (x,y , s=10, c = 'blue', alpha = 1)
    x2,y2 = (zip(*cluster))
    plt.scatter(x2,y2, s= 10, c = 'red', alpha = 0.7)
    x3,y3 = (zip(*noncluster))
    plt.scatter(x3,y3, s= 10, c = 'green', alpha = 0.5)
    plt.show()



def codeSpecial(data):
    mindist = []
    vals = []
    gridhelp = []
    contour = []
    for i, pts in enumerate(data):
        temp = []
        j = i + 1
        for pts2 in data:
            if not (pts == pts2).all():
                temp += [mydist(pts, pts2)]
            if ( j < len(data)):
                gridhelp += [np.add(data[i],1/2 * np.subtract(data[j],data[i]))]
                j = j + 1

        mindist += [min(temp)]
        vals += [multivariate_normal(mean=pts, cov=[[mindist[i], 0], [0, mindist[i]]], allow_singular=True)]
    print('vals finished')
    print(len(gridhelp), len(data))

    for points in gridhelp:
        result = 0
        for j in vals:
            result = j.pdf(points) + result
            if result <= 1:  # less calcualting, threshold erreicht genügt
                break
        if result > 1:
            contour += [points]
    print ('contour len:',len(contour))
    clusterpts = []
    for pts in data:
        result = 0
        for j in vals:
            result = j.pdf(pts) + result
            if result >= 1:
                clusterpts += [pts]
                break
    print('clusterpts', len(clusterpts))
    if (len(contour) == 0):
        plotdemo(data, clusterpts)
    else:
        plotdemo2(data, clusterpts, contour)

def codeStandard(data):
    mindist = []#minimum distance for each points
    #mindistindex = [] #index of point with minimum distance
    tempcovariance = []
    mintotaldistance = 0  # gitternetzlinienabstand
    minx = data[0][0]
    miny = data[0][1]
    maxx = data[0][0]
    maxy = data[0][1]
    vals = []
    for i,pts in enumerate(data):
        temp = []
        for pts2 in data:
            if not (pts==pts2).all():
                temp += [mydist(pts, pts2)]
        mindist += [min(temp)]
        #gitternetz hilfskonstrukt
        if pts[0] < minx:
            minx = pts[0]
        if pts[0] > maxx:
            maxx = pts[0]
        if pts[1] < miny:
            miny = pts[1]
        if pts[1] > maxy:
            maxy = pts[1]
         #mindistindex += [temp.index(min(temp))] index of minimum distance point
        #temp covariance values
        #tempcovariance += [mydistance(pts, data[mindistindex[i]])] # kleinster index abstand komponentenweise
        #vals = multivariate_normal(mean=pts, cov=[[tempcovariance[i][0],0],[0,tempcovariance[i][1]]],allow_singular=True) #covariance komponentenweise
        vals  += [multivariate_normal(mean=pts, cov=[[mindist[i],0],[0, mindist[i]]], allow_singular = True)]

    #get grid
    mintotaldistance = min(mindist)
    minx = minx-mintotaldistance/2
    miny = miny-mintotaldistance/2
    maxx = maxx+mintotaldistance
    maxy = maxy+mintotaldistance
    print('minx',minx,'maxx',maxx, 'miny',miny,maxy, mintotaldistance)

    #numpy magic http://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy FANCY creates grid
    gridhelp = np.mgrid[minx:maxx:mintotaldistance, miny:maxy:mintotaldistance].reshape(2,-1).T
    print('normal grid', len(gridhelp))
    # print(gridhelp[0])
    myttime = time.time()
    clusterres = []
    # #without GRID
    # for res in data:
    #     result = []
    #     for i in vals:
    #         result += [i.pdf(res)]
    #     clusterres += [np.sum(result)]
    # clusteredpoints = 0
    # contour = []
    # for i,value in enumerate(data):
    #     if clusterres[i] > 1:
    #         print(clusterres[i])
    #         contour += [value.tolist()]
    #         clusteredpoints += 1


    clusteredpoints = 0
    contour = []
    contourzero = []
    for idx, pointsgrid in enumerate(gridhelp):
        result = 0
        for j in vals:
            result = j.pdf(pointsgrid) + result
            if result >= 1: # less calcualting, threshold erreicht genügt
               clusteredpoints += 1
               contour += [pointsgrid]
               break
        #clusterres += [result]
        #if 0.5 <= result < 1:
         #   contourzero += [pointsgrid]
    print("zeit für epic schleife", time.time()-myttime)
    print(' ', clusterres)

    #print(contour)
    print(clusteredpoints, "Anzahl punkte in clustern")
    #myplots(contour)
    if len(contourzero)> 0:
        plotdemo(data, contourzero)
    else:
        print("no zeroes")
    plotdemo(data,contour)



def dendrogrammofclustering(result):
    plt.figure(figsize=(25,10))
    plt.title('Hierarchical Clustering')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(result, leaf_rotation=90, leaf_font_size = 8)
    plt.show()




#0,0159077092538
if __name__ == '__main__':
    #dataset = np.random.uniform(0, 10, size=(40, 2))
    start_time = time.time()
    dataset = load_mydataset()
 #   dataset2 = load_mydataset(True)

    dataset, truelabels = list(make_blobs(1000,2,2))
    print(dataset)
    #tri = Triangulation(dataset[:,0], dataset[:,1])

    kmeans(dataset,2)
    agglomutualnearestneighbour(dataset)
    noniterative_clustering(dataset, 2)
    #spatial_separation(dataset)
    #0.0128479387995
    #uniform_density(dataset)

    """test = reader('test.out')
    test2 = reader('test2.out')
    ls = [-1 for i in range(len(dataset))]
    ls2 = [-1 for i in range(len(dataset))]
    for idx, i in enumerate(test):
        for j in i:
            ls[j] = idx
    for idx, i in enumerate(test2):
        for j in i:
            ls2[j] = idx

    print(ls, "LS")
    print("zwote", ls2)

    print(adjusted_rand_score(ls,ls2[::-1]))

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
