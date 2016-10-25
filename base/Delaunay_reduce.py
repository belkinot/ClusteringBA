from scipy.spatial import Delaunay
from base.helperfunctions import *
import networkx as nx
import math
import time
from itertools import combinations

#delaunay triangulation
def Delaunay_reduce(dataset):
    """ ganzer algorithmus """
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
    # convert the list of distances into local mean value
    for liste in helpmean:
        val = 0
        for i in liste:
            val += i
        val = val/len(liste)
        local_mean += [val]
    global_mean = sum/(len(dist)/2)
    global_variation = 0
    for i in range(0, len(dist), 2): # "remove" duplicates
        global_variation += (dist[i][0] - global_mean)**2 #dist[i][0] is the distance
    global_variation = global_variation/(len(dist)/2)
    global_distance_constraint = []
    for i in range(len(dataset)):
        alpha = global_mean/local_mean[i]
        global_distance_constraint += [global_mean + alpha * global_variation]
    #print(global_distance_constraint)
    tri_global_reduced = []
    tri_global_reduced2 = []
    for i in dist:
        if global_distance_constraint[i[1]] >= i[0]:
            tri_global_reduced += [i]
            tri_global_reduced2 += [(i[1], i[2])]

    local_distance_constraint = [] # [[] for _ in range(len(dataset))]
    m = get_matrix(len(dataset), tri_global_reduced2)
    graph = nx.from_numpy_matrix(m)
    nb = [] # direct neighbours
    for i in range(len(dataset)):
        nb += [graph.neighbors(i)]
    print(nb)
    res_dict = dict()
    two_order_nb = []
    for i, bla in enumerate(nb):
        two_order_nb_list = []
        for j in bla:
            #if i < j:
            res_dict[(i, j)] = mydist(dataset[i], dataset[j])
            two_order_nb_list += [(i,j)]
            for example in nb[j]:
                two_order_nb_list += [(j,example)]
        two_order_nb += [two_order_nb_list]
    print(two_order_nb)
    two_order_neighborhood = []
    for i in two_order_nb:
        two_order_neighborhood += [set(tuple(sorted(t)) for t in i)]

    print(two_order_neighborhood[0])
    two_order_mean_list = []
    for i in two_order_neighborhood:
        two_order_mean = 0
        for j in i:
            two_order_mean += res_dict[j]
        if len(i) != 0:
            two_order_mean_list += [two_order_mean/len(i)]
        else:
            two_order_mean_list += [0]
    print(len(nb))
    print(len(two_order_neighborhood))
    print(len(two_order_mean_list))
    print(len(dataset))
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
    """
    res_list = []
    local_mean_variation = []
    local_mean_constraint = []
    for idx in range(len(dataset)):
        set_of_dings = set()
        for val in end_res:
            if val[0][2] == idx or val[0][0] == idx:
                set_of_dings |= {val[1]}
                set_of_dings |= {val[3]}
        res_list += [set_of_dings]
    mean_variation = 0
    for idx in range(len(dataset)):
        for mean_value in res_list:
            for mean_value_v in mean_value:
                mean_variation += (res_dict[tuple(mean_value_v)]-two_order_mean[idx])**2
            local_mean_variation += [mean_variation/len(mean_value)]
            local_distance_constraint += [two_order_mean[idx]+local_mean_variation[idx]]
#
    nbb = [set() for _ in range(len(dataset))] # neighbours of neighbours
    for idx, second in enumerate(nb):
        nbb[idx] = nbb[idx] | set(second) # füge DIREKTE Nachbarn ein
        for i in second:
            nbb[idx] = nbb[idx] | set(nb[i]) # nbb - set of 2nd neighbours
    edgelist = [] # edgelist for 2-order mean
    print(len(nbb[99]))
    print(nbb[99])
    return
    for nbbset in nbb:
        #print(nbbset)
        edges = [x for x in combinations(nbbset, 2)]
        print(edges[0])
        edgelist += [edges]
    # edgelist is a result of all edges in 2-neighbourhood of edelist[i] which is dataset[i]
    two_order_mean = [[] for _ in range(len(dataset))] # 2 order mean list
    mean_variation = []
    test_time = time.time()
    for idx, liste in enumerate(edgelist):
        #calculate two-order mean
        #two_order_mean_help = 0
        distance = []
        mean_variation_help = 0
        for j in range(len(list)):
            p1 = liste[j][0]
            p2 = liste[j][1]
            distance += [mydist(dataset[p1], dataset[p2])]
            print(type(distance[0]))
            #print(len(liste))
            #two_order_mean_help += mydist(dataset[p1], dataset[p2])
        #two_order_mean_help = two_order_mean_help/denominator
        #two_order_mean += [two_order_mean_help]
        two_order_mean += [np.sum(distance)/len(liste)]
        #print(two_order_mean)
        #print(np.sum(distance)/denominator) weniger nachkommastellen!!! 10 statt 16
        for example in distance:
            mean_variation_help += (two_order_mean[idx] - example)*(two_order_mean[idx] - example)

        mean_variation += [mean_variation_help/len(liste)]

    print(time.time()-test_time)

    for i in range(len(dataset)):
        local_distance_constraint[i] = two_order_mean[i] - mean_variation[i]
    """
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
