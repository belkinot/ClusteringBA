import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import multivariate_normal
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score, v_measure_score

from base.helperfunctions import *
from base.MMD-Algorithm import *
from itertools import combinations

import time
from random import shuffle



#spatial separation abstand kürzester aus sunterschiedlichen clustern
#connectedness lokale nachbarschaft maximaler abstand dbscan minpts --> connectmaß
#compactness sse dbsscan- kmeans
#uniform density - radius um jeden punkt in einem cluster, anzahl der punkt in dem radius sollte einigermaßen gleich sein



def get_colors():
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    """for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    """
    #hexadecimal for scatter (problem interpretation of the rgb vs colorcodecatalogue
    res = ["#{:02X}{:02X}{:02X}".format(*triple) for triple in tableau20]
    shuffle(res)
    return res

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

def showres(result, noise=None):
    colors = get_colors()
    for i, point_list in enumerate(result):
        x, y = zip(*point_list)
        plt.scatter(x, y, c=colors[i % len(colors)])

        if noise:
            x, y = zip(*noise)
            plt.scatter(x, y, c='b', marker='o')
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

def get_matrix(size, connected):
    m = np.zeros((size, size))

    for t in connected:
        x, y = min(t), max(t)
        m[(x,y)] = 1

    return m

def gen_labels(size, connected):
    m = get_matrix(size, connected)
    g = nx.from_numpy_matrix(m)
    return list(nx.connected_components(g))


def gen_results_from_labels(points, labels):
    result = []
    noise = []
    for i in range(len(labels)):
        if len(labels[i]) > 1:
            result += [[tuple(points[comp]) for comp in list(labels[i])]]
        else:
            noise += [tuple(points[comp]) for comp in list(labels[i])]

    print(result)
    return result, noise

def dendrogrammofclustering(result):
    plt.figure(figsize=(25,10))
    plt.title('Hierarchical Clustering')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(result, leaf_rotation=90, leaf_font_size = 8)
    plt.show()

def agglomutualnearestneighbour(dataset):
    #matrix nearest neighbours of each point
    #matrix m1 and D, where neighbourmatrix[0] -> distance, neighbourhoodmatrix[1] --> index of point
    neighbourhoodmatrix = []
    for pts in dataset:
        neighbour = []
        for idx,pts2 in enumerate(dataset):
            neighbour += [[mydist(pts,pts2),idx]]
        neighbour.sort()
        neighbourhoodmatrix += [neighbour]

    #matrix m2
    m2_len = len(dataset)
    m2 = [[0 for x in range(m2_len)] for y in range(m2_len)]
    for idx, mnv in enumerate(m2):
        for i in range(m2_len):
            temp = neighbourhoodmatrix[idx][i][1]
            #print('temp',temp)
            item = idx
            #print('item',item)
            for jdx, pts in enumerate(neighbourhoodmatrix[temp]):
                if item  == pts[1]:
                    if jdx + i > 10:
                        m2[idx][i] = 2000
                    else:
                        m2[idx][i] = jdx + i #mutual neighbourhood value
                    #print('tempmnv', tempmnv)
    counter = 0
    tempcluster = []
    for mnvvalue in range (2,11):
        for idx in range(len(m2)):

            for i in range(0,5):# only 5 nearest neighbours - k -->
                if m2[idx][i] == mnvvalue:
                    tempdist = neighbourhoodmatrix[idx][i][0] #punkte zu cluster: idx, neighbourhoodmatrix[idx][i][1]
                    tempcluster += [[[tempdist],[idx], [neighbourhoodmatrix[idx][i][1]]]]
                if m2[idx][i] <= 10:
                    counter += 1
    #print(len(tempcluster),tempcluster)
    l = [(x[1][0], x[2][0]) for x in tempcluster]
    #print(l)
    labels = gen_labels(len(dataset), l)
    print('LABELS' , labels, len(labels))
    results, _ = gen_results_from_labels(dataset, labels)
    #print(results)
    showres(results)
    #np.savetxt('test.out', list(zip(*results)), delimiter=',')  # X is an array
    with open('test.out', "a") as f:
        print(results, file=f)
    #print(neighbourhoodmatrix[11])
    #print(m2[0])
    #print(counter)




#0,0159077092538
if __name__ == '__main__':
    #dataset = np.random.uniform(0, 10, size=(40, 2))
    start_time = time.time()
    dataset = load_mydataset()
    dataset2 = load_mydataset(True)
    graph1 = noniterative_clustering(dataset)
    graph2 = noniterative_clustering(dataset2)
    print(graph1, graph2)
    print(graph1 == graph2)
    counter = 0
    for idx, i in enumerate(graph1):
        if not i.all() == graph2[idx].all():
            counter += 1
    print(counter)

    #adjacencymatrix sind gleich!
    test = reader('test.out')
    test2 = reader('test.out2')
    ls = [-1 for i in range(len(dataset))]
    ls2 = [-1 for i in range(len(dataset))]
    for idx, i in enumerate(test):
        for j in i:
            ls[j] = idx
    for idx, i in enumerate(test2):
        for j in i:
            ls2[j] = idx
    #print(v_measure_score())
    #dataset = load_dataset_with_labels()
   # plot_true_labels(dataset)
    #codeStandard(dataset)
    #codeSpecial(dataset)
    #agglomutualnearestneighbour(dataset)
    #y_pred = KMeans(6).fit_predict(dataset)
    #y_pred = MeanShift().fit_predict(dataset)
    #helperfunctions.sklearn_kmeans_sse(y_pred, dataset)
    #res = reader('k_means.out')
    #showres(res)
    #print(y_pred)
    #noniterative_clustering(dataset)
    #myplots(dataset)
    #sse_calculate()

    print("My Programm took", time.time()-start_time, "to run")
