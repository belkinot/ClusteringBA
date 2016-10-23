from scipy.spatial import Delaunay
from base.helperfunctions import *
import networkx as nx
import math
import time
from itertools import combinations

#delaunay triangulation
def Delaunay_reduce(dataset):
    start_time = time.time()
    tri = Delaunay(dataset)
    #print(tri.simplices)
    dist = gen_all_connected(dataset, tri)
    dist.sort()
    sum = 0
    helpmean = [[] for _ in range(len(dataset))] # variance of all local_means
    local_mean = []
    for i in range(0, len(dist), 2):
        value = dist[i][0]
        sum += value
        helpmean[dist[i][1]] += [value]
        helpmean[dist[i][2]] += [value]
    #convert the list of distances into local mean value
    for list in helpmean:
        val = 0
        for i in list:
            val += i
        val = val/len(list)
        local_mean += [val]
    global_mean = sum/(len(dist)/2)
    global_variation = 0
    alpha = []
    for i in range(0, len(dist), 2): #"remove" duplicates
        global_variation += (dist[i][0] - global_mean)*(dist[i][0]- global_mean) #dist[i][0] is the distance
    global_variation = global_variation/(len(dist)/2)
    global_distance_constraint = []
    for i in range(len(dataset)):
        alpha += [global_mean/local_mean[i]]
        global_distance_constraint += [global_mean + alpha[i] * global_variation]
    #print(global_distance_constraint)
    tri_global_reduced = []
    tri_global_reduced2 = []
    for i in dist:
        if global_distance_constraint[i[1]] >= i[0]:
            tri_global_reduced += [i]
            tri_global_reduced2 += [(i[1], i[2])]

    local_distance_constraint = [[] for _ in range(len(dataset))]
    m = get_matrix(len(dataset), tri_global_reduced2)
    graph = nx.from_numpy_matrix(m)
    nb = [] #direct neighbours
    for i in range(len(dataset)):
        nb += [graph.neighbors(i)]
    nbb = [set() for _ in range(len(dataset))]#neighbours of neighbours
    for idx, second in enumerate(nb):
        nbb[idx] = nbb[idx] | set(second) # füge DIREKTE Nachbarn ein
        for i in second:
            nbb[idx] = nbb[idx] | set(nb[i])        #nbb - set of 2nd neighbours
    #need kantenlängen der neighboursets
    #kanten sind alle combinaationen aus dem Set nbb (als tupel)
    edgelist = [] # edgelist for 2-order mean
    for nbbset in nbb:
        #print(nbbset)
        edges = [x for x in combinations(nbbset, 2)]
        edgelist += [edges]
    #edgelist is a result of all edges in 2-neighbourhood of edelist[i] which is dataset[i]

    two_order_mean = [] # 2 order mean list
    mean_variation = []
    for idx, list in enumerate(edgelist):
        #calculate two-order mean
        #two_order_mean_help = 0
        distance = []
        denominator = 0
        denom_help = 0
        mean_variation_help = 0
        for j in range(len(list)):
            p1 = list[j][0]
            p2 = list[j][1]
            distance += [mydist(dataset[p1], dataset[p2])]
            #two_order_mean_help += mydist(dataset[p1], dataset[p2])
            denominator += 1
        #two_order_mean_help = two_order_mean_help/denominator
        #two_order_mean += [two_order_mean_help]
        two_order_mean += [np.sum(distance)/len(list)]
        #print(two_order_mean)
        #print(np.sum(distance)/denominator) weniger nachkommastellen!!! 10 statt 16
        for example in distance:
            mean_variation_help += (two_order_mean[idx] - example)*(two_order_mean[idx] - example)
        mean_variation += [mean_variation_help/len(list)]
    for i in range(len(dataset)):
        local_distance_constraint[i] = two_order_mean[i] - mean_variation[i]
    tri_local_reduced = []
    tri_local_reduced_tuples = []
    for i in tri_global_reduced:
        if local_distance_constraint[i[1]] >= i[0]:
            tri_local_reduced += [i]
            tri_local_reduced_tuples += [(i[1], i[2])]
    labels = gen_labels(len(dataset), tri_local_reduced_tuples)
    print('LABELS', labels)
    result, noise = gen_results_from_labels(dataset, labels)
    print('ZEIT' , time.time()-start_time)
    showres(result, noise)


def find_neighbors(pindex, tri):
    return tri.vertex_neighbor_vertices[1][
           tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex + 1]]

def gen_all_connected(dataset,tri):
    dist = []
    for i, p in enumerate(dataset):
        neighbors = find_neighbors(i,tri)
        dist += [(mydist(p, dataset[n]), i, n) for n in neighbors]
    return dist