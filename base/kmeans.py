from base.helperfunctions import *
import random
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import csgraph_from_dense
import networkx as nx
from networkx import union
from base.Delaunay_reduce import gen_all_connected

def kmeans(dataset, k):
    """ K-Means-Algorithm"""
    centre = [dataset[i] for i in range(k)]
    # nutze zufallswerte
    #centre = [dataset[i] for i in random.sample(range(len(dataset)), k)]
    #solange clusterzentrne nicht gleich oder solange k-1 clusters nicht gleich oder n-iterationen
    n = 0
    clusters, centre, labels = kmeansdistance(dataset, centre)
    test_centre = tuple()
    print(centre)
    while n < 300 and not test_centre == centre:
        test_centre = centre[:]
        clusters, centre, labels = kmeansdistance(dataset, centre)
        n += 1
    print(clusters)
    print(n)
    label_set = []
    for jdx, i in enumerate(labels):
        label_set += [set(i)]
    graph = create_mst(dataset, label_set) # build a graph out of k-means clustering
    #degree_connectedness(graph)
    return (label_set)


    #cluster_labels = [set(x) for x in clusters]
    #clustering, noise = gen_results_from_labels(dataset, cluster_labels)
    showres(clusters)


#create graph out of clusters:
def create_mst(dataset, labels):
    mst = []
    temp_dataset = []
    graph = nx.Graph()
    for i in labels:
        for j in i:
            temp_dataset += [dataset[j]]
        delaunay_graph = Delaunay(temp_dataset)
        delaunay_graph = gen_all_connected(temp_dataset, delaunay_graph)
        delaunay = []
        for graphes in delaunay_graph:
            delaunay += [(graphes[1], graphes[2])]
        m = get_matrix(len(temp_dataset), delaunay)
        graph2 = nx.from_scipy_sparse_matrix(csgraph_from_dense(m))
        mst += [nx.minimum_spanning_tree(graph2)]# minimum spanning tree out of graph
    for subgraph in mst:
        graph = nx.disjoint_union(graph,subgraph)
        #graph.add_nodes_from(subgraph.nodes)
        #graph.add_edges_from(subgraph.edges)
        #graph = nx.union(graph, subgraph)
    return graph

def kmeansdistance(dataset, centre):
    clusters = [[] for x in range(len(centre))]
    labellist = [[] for x in range(len(centre))]
    for idx, example in enumerate(dataset):
        exampledist = []
        for i in range(0, len(centre)):
            exampledist += [mydist(example, centre[i])]
        clusters[exampledist.index(min(exampledist))] += [example]
        labellist[exampledist.index(min(exampledist))] += [idx]
    #print(clusters)
    for idx, example in enumerate(clusters):
        #x, y = np.sum(example, axis=0)
        #newcentre = (x/len(example), y/len(example))
        newcentre = tuple(x/len(example) for x in np.sum(example, axis=0))
        centre[idx] = newcentre

    return clusters, centre, labellist
