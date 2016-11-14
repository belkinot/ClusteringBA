from scipy.spatial import Delaunay
from base.helperfunctions import *
import queue
import networkx as nx
import math
import time
from itertools import combinations

#delaunay triangulation
def Delaunay_reduce(dataset, triangulation):
    """ ganzer algorithmus """
    start_time = time.time()
    #tri = Delaunay(dataset)
    tri = triangulation
    #print(tri.simplices)
    dist = gen_all_connected(dataset, tri)
    dist.sort()
    sum = 0
    helpmean = [[] for _ in range(len(dataset))] # Kantenlänge JEDER benachbarten Kante von Punkt an der Stelle i
    local_mean = []
    for i in range(0, len(dist), 2):
        value = dist[i][0]
        sum += value
        helpmean[dist[i][1]] += [value]
        helpmean[dist[i][2]] += [value]
    # convert the list of distances into local mean value
    for liste in helpmean:
        val = 0
        for i in liste:
            val += i
        val = val/len(liste)
        local_mean += [val] # Local_Mean des Punktes i
    global_mean = sum/(len(dist)/2)
    global_variation = 0
    for i in range(0, len(dist), 2): # "remove" duplicates
        global_variation += (dist[i][0] - global_mean)**2 #dist[i][0] is the distance
    global_variation = math.sqrt(global_variation/(len(dist)/2)) # global "variation" laut paper
    global_distance_constraint = [] # global distance constraint
    for i in range(len(dataset)):
        alpha = global_mean/local_mean[i]
        global_distance_constraint += [global_mean + alpha * global_variation]
    #print(global_distance_constraint)
    tri_global_reduced = [] #global reduzierte DT
    tri_global_reduced2 = [] # Tupelliste der global reduzierten DT
    for i in dist:
        if global_distance_constraint[i[1]] >= i[0]:
            tri_global_reduced += [i]
            tri_global_reduced2 += [(i[1], i[2])]

    local_distance_constraint = []
    m = get_matrix(len(dataset), tri_global_reduced2)
    graph = nx.from_numpy_matrix(m)
    nb = [] # Nachbarn des Punkte i
    for i in range(len(dataset)):
        nb += [graph.neighbors(i)]
    #print(nb)
    res_dict = dict()
    two_order_nb = [] # 2er Nachbarschaften
    # baue Distanzdictionary
    for i, bla in enumerate(nb):
        two_order_nb_list = []
        for j in bla:
            res_dict[(i, j)] = mydist(dataset[i], dataset[j]) # distances of all edges in triangulation
            two_order_nb_list += [(i,j)]
            for example in nb[j]:
                two_order_nb_list += [(j,example)]
        two_order_nb += [two_order_nb_list]
    #print(two_order_nb)
    two_order_neighborhood = [] # alle Tupel der 2-er Nachbarschaften, keine doppelten Kanten!
    for i in two_order_nb:
        two_order_neighborhood += [set(tuple(sorted(t)) for t in i)]

    #print(two_order_neighborhood[0])
    two_order_mean_list = [] # Mittelwert der Kantenlänge der 2-er Nachbarschaft des Punkte i
    for i in two_order_neighborhood:
        two_order_mean = 0
        for j in i:
            two_order_mean += res_dict[j]
        if len(i) != 0: # falls keine Nachbarn mehr vorhanden!
            two_order_mean_list += [two_order_mean/len(i)]
        else:
            two_order_mean_list += [0]
    #print(len(nb))
    #print(len(two_order_neighborhood))
    #print(len(two_order_mean_list))
    #print(len(dataset))

    local_variance_list = [] # local variance list des Punkte an der Stelle i(1er nachbarschaft)
    for i in range(len(dataset)):
        local_variance_help = 0
        for x in helpmean[i]:
            local_variance_help += (x-local_mean[i])**2
        local_variance_help = math.sqrt(local_variance_help/len(helpmean[i]))
        local_variance_list += [local_variance_help]

    mean_variation = [] # mean-variation der 2er Nachbarschaft des Punktes an der Stelle i
    for neighbor in nb:
        mean_variation_help = 0 # standardabweichung der 1er nachbarn
        two_path_length_neighbor = []
        for i in neighbor:
            mean_variation_help += local_variance_list[i]
            two_path_length_neighbor += nb[i]
        if (len(neighbor) != 0): # falls kein nachbar vorhanden
            mean_variation_help = mean_variation_help/len(neighbor)
        else:
            mean_variation_help = 0
        mean_variation_help2 = 0 # standardabweichung der 2er Nachbarn
        for x in two_path_length_neighbor:
            mean_variation_help2 += local_variance_list[x]
        if (len(two_path_length_neighbor) != 0): # falls kein nachbar vorhanden
            mean_variation_help2 = mean_variation_help2/len(two_path_length_neighbor)
        else:
            mean_variation_help2 = 0
        mean_variation_help += mean_variation_help2
        mean_variation_help = mean_variation_help/2
        mean_variation += [mean_variation_help] # mean variation aus dem paper
    for i in range(len(dataset)):
        local_distance_constraint += [two_order_mean_list[i] + mean_variation[i]]




    """
    local_variance_list = []
    for idx, i in enumerate(two_order_neighborhood):
        local_variance = 0
        for j in i:
            local_variance += (res_dict[j] - two_order_mean_list[idx])**2
        if len(i) != 0:
            local_variance_list += [local_variance/len(i)]
        else:
            local_variance_list += [0]
    for i in range(len(dataset)):
        local_distance_constraint += [two_order_mean_list[i] + local_variance_list[i]]
    """

    """
    for i, blub in enumerate(nb):
        test = [x for x in combinations(blub, 2)]
        for t in test:
            l = min(t[0], t[1])
            r = max(t[0], t[1])
            left = (l, i)
            right = (i, r)

            end_res += [((left + (right[1],)), left, res_dict[left], right, res_dict[right], res_dict[left] + res_dict[right])]
    end_res.sort()
    end_res_dict = dict()
    for i, bla in enumerate(end_res):
        end_res_dict[left + (right[1],)] = res_dict[left] + res_dict[right]
    two_order_mean = []
    local_meanvariation = []
    for idx in range(len(dataset)):
        mean = 0
        mean2 = []
        denom = 0
        for val in end_res:
            if val[0][2] == idx or val[0][0] == idx:
                mean += val[2] + val[4]
                denom += 1
        if denom != 0:
            two_order_mean += [mean/denom]
        else:
            two_order_mean += [0]
    print(two_order_mean)

    """


    tri_local_reduced = [] # lokal reduzierte DT
    tri_local_reduced_tuples = [] # Tupelliste der lokal reduzierten DT
    for i in tri_global_reduced:
        if local_distance_constraint[i[1]] >= i[0]:
            tri_local_reduced += [i]
            tri_local_reduced_tuples += [(i[1], i[2])]

    """
    density_indicator_list = [] # density indicator aus dem paper
    nsdr_list = [] # nsdr aus paper
    m2= get_matrix(len(dataset), tri_local_reduced_tuples)
    g2 = nx.from_numpy_matrix(m2)
    density_nb = [] # alle nachbarn des Punktes an der Stelle i
    for i in range(len(dataset)):
        density_nb += [g2.neighbors(i)]
    nearest_neighbor = []
    threshold = []
    directly_reachable = [[] for x in range(len(dataset))]
    for idx, neighbor in enumerate(density_nb): #alle nachbarn
        nsdr = 0
        nearest_neighbor_dist_help = []
        for j in neighbor:
            nearest_neighbor_dist_help += [res_dict[idx,j]]
        if len(nearest_neighbor_dist_help) != 0:
            nearest_neighbor += [min(nearest_neighbor_dist_help)] # nächster nachbar
        else:
            nearest_neighbor += [0]
        std = 0
        for i in nearest_neighbor_dist_help:
            std += math.sqrt((nearest_neighbor[idx] - i)**2)
        if len(nearest_neighbor_dist_help) != 0:
            std = std/len(nearest_neighbor_dist_help)
        else:
            std = 0
        threshold += [nearest_neighbor[idx] + 3* std]

        for j in neighbor:
            if res_dict[idx, j] <= threshold[idx]: # überprüfen ob der Punkt spatial directly reachable ist
                directly_reachable[idx] += [j]
                nsdr += 1
        nsdr_list += [nsdr]
        if len(density_nb[idx]) != 0:
            density_indicator_list += [(nsdr_list[idx] + nsdr_list[idx] / len(density_nb[idx]), idx)]
        else:
            density_indicator_list += [(0, idx)]  # no neighbours

    #firstindex = density_indicator_list.index(max(density_indicator_list)) # höchster density_indicator
    maxindex = [i for i,x in enumerate(density_indicator_list) if x == max(density_indicator_list)]
    print(maxindex)

    mintreshold = threshold[maxindex[0]]
    firstindex = maxindex[0]
    for i in maxindex:
        if threshold[i] < mintreshold:
            mintreshold = threshold[i]
            firstindex = i
    cluster = [firstindex]
    q = queue.Queue()
    cluster = density_clusters(dataset, density_indicator_list,density_nb, directly_reachable, threshold, firstindex, cluster,q)
    print('CLUSTER', cluster)
    #expandingneighbors = density_nb[firstindex]
    #print(expandingneighbors)
    """
    labels = gen_labels(len(dataset), tri_local_reduced_tuples)
    #print('LABELS', labels)
    result, noise = gen_results_from_labels(dataset, labels)
    #print('ZEIT' , time.time()-start_time)
    return labels
    showres(result, noise)

    """temp_density_indicator_list = []

    for i in expandingneighbors:
        temp_density_indicator_list += [density_indicator_list[i]]
    temp_density_indicator_list.sort(reverse=True)

    print(temp_density_indicator_list)

    cluster = [firstindex, temp_density_indicator_list[0][1]]
    #clusterset += [set(firstindex, temp_density_indicator_list[0][1])]

    for i in range(1,len(temp_density_indicator_list)):
        if temp_density_indicator_list[i][1] in directly_reachable[firstindex]:
            for i in cluster:
                clusteravg += dataset[i]
            clusteravg = clusteravg/len(cluster)
            if mydist(dataset[temp_density_indicator_list[i][1]],clusteravg) < threshold[firstindex]:
                cluster += [i]
"""

def density_clusters(dataset,density_indicators_list, density_neighbor,directly_reachables,threshold, index, cluster, queues):
    expandingneighbors = density_neighbor[index]
    temp_density_indicator_list = []
    for i in expandingneighbors:
        temp_density_indicator_list += [density_indicators_list[i]]
    temp_density_indicator_list.sort(reverse=True)
    for i in temp_density_indicator_list:
        queues.put(i[1])
    nextindex = queues.get()
    density_reachable_check(dataset, density_indicators_list, density_neighbor, directly_reachables, threshold, index, nextindex, cluster, queues)
    return cluster

def density_reachable_check(dataset, density_indicators_list,density_neighbor,directly_reachables, threshold, index, nextindex,cluster, queues):
    if nextindex in directly_reachables[index] and nextindex not in cluster:
        clusteravg = 0
        for i in cluster:
            clusteravg += dataset[i]
        clusteravg = clusteravg/len(cluster)
        if mydist(dataset[nextindex],clusteravg) < threshold[index]:
            cluster += [nextindex]
            density_clusters(dataset,density_indicators_list, density_neighbor, directly_reachables, threshold, nextindex, cluster, queues)
    if not queues.empty():
        nextindex = queues.get()
        density_reachable_check(dataset, density_indicators_list, density_neighbor, directly_reachables, threshold, nextindex, index, cluster, queues)



    # alle nachbarn von diesem index (density_nb[firstindex]
    # sortiere diese (aber spatially directly reachable, d.h. res_dict[firstindex, diesen nachbarn] < t1 UNd spatially reachable ??
    #






def find_neighbors(pindex, tri):
    return tri.vertex_neighbor_vertices[1][
           tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex + 1]]

def gen_all_connected(dataset,tri):
    dist = []
    for i, p in enumerate(dataset):
        neighbors = find_neighbors(i,tri)
        dist += [(mydist(p, dataset[n]), i, n) for n in neighbors]
    return dist
