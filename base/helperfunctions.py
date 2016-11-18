import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from random import shuffle

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

def mydistance (p1,p2):
    return 1/2*np.absolute(p1-p2)
    #componentwise distance


def mydist (p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def load_mydataset(reverse=False):
    name = 'compound'
    data = np.genfromtxt(name + '.csv', delimiter=',')
    dataset = data[:,:4]
    #dataset2 = np.concatenate((dataset[::2],dataset[1::2]), axis=0)
    if reverse:
        dataset = dataset[::-1]
    return dataset

def load_dataset_with_labels():
    name = 'pathbased'
    data = np.genfromtxt(name + '.csv', delimiter=',')
    return data

def showres(result, noise=None):
    colors = get_colors()
    for i, point_list in enumerate(result):
        coord = list(zip(*point_list))
        plt.scatter(coord[0], coord[1], c=colors[i % len(colors)])

        if noise:
            coord = list(zip(*noise))
            plt.scatter(coord[0], coord[1], c='b', marker='o')
    plt.show()

def get_matrix(size, connected):
    m = np.zeros((size, size))

    for t in connected:
        x, y = min(t), max(t)
        m[(x,y)] = 1

    return m

def get_weighted_matrix(size, connected, weights):
    m = np.zeros((size,size))
    for t in connected:
        x,y = min(t), max(t)
        m[(x,y)] = weights[x][y]
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

    #print(result)
    return result, noise


def plot_true_labels(dataset):
    data1 = []
    data2 = []
    data3 = []
    for data in dataset:
        if data[2] == 1:
            data1 += [(data[0], data[1])]
        elif data[2] == 2:
            data2 += [(data[0], data[1])]
        elif data[2] == 3:
            data3 += [(data[0], data[1])]
    x, y = zip(*data1)
    plt.scatter(x,y, c = 'red')
    x,y = zip(*data2)
    plt.scatter(x,y, c = 'blue')
    x,y = zip(*data3)
    plt.scatter(x,y, c = 'green')
    plt.show()


def sklearn_kmeans_sse(labels, dataset):
    res_list = [list([]) for _ in range(max(labels)+1)]
    for i, label in enumerate(labels):
        res_list[label] += [tuple(dataset[i])]
    print(res_list)
    with open('k_means.out', 'w') as f:
        print(res_list, file=f)


def reader(test,):
    x = []
    with open(test, "r") as f:
        for line in f.readlines():
            x += eval(line)
    return x


def sse_per_label(DATASET, labels):
    sse = 0
    for vals in labels:
        centre = 0
        sse_label = 0
        for i in vals:
            centre += DATASET[i]
        centre = centre/len(vals)
        for j in vals:
            sse_label += mydist(centre, DATASET[j])**2
        sse += sse_label/len(vals)
    return sse

def sse_calculate():
    x = reader('k_means.out')
    l = 0
    r = 0
    for i, vals in enumerate(x):
        for j, points in enumerate(vals):
            l += points[0]
            r += points[1]
        l = l/len(vals)
        r = r/len(vals)
        clustercenter = (l,r)
        sse = 0
        for j, points in enumerate(vals):
            sse += mydist(clustercenter, points)
        print(sse)


def spatial_separation(dataset, labels):
    #labels = reader('test2.out')
    res = mydist(dataset[0], dataset[1])
    for enum, i in enumerate(labels):
        for j in labels[enum+1:]:
            for example in i:
                for example2 in j:
                    minseparation = mydist(dataset[example], dataset[example2])
                    if res > minseparation:
                        res = minseparation
    #print(res)
    return res

#def myradius(centre, radius, dataset):


def uniform_density(dataset):
    mydataset = []
    edgelist = []
    labels = reader('test2.out')
    test = labels[0]
    for i in test:
        mydataset += [dataset[i]]
    for idx in range(len(mydataset)):
        for idx2 in range(idx+1, len(mydataset)):
            edgelist += [({idx, idx2}, mydist(mydataset[idx], mydataset[idx2]))]
    edgelist.sort(key=lambda x: x[1])

    res_dict = dict()
    for edge in edgelist:
        idx = edge[0]
        print(idx)
        for i in idx:
            if i in res_dict.keys():
                res_dict[i] += [1]
            else:
                res_dict[i] = [1]
    #print(res_dict)
    print(edgelist[0], edgelist[1])



def graph_longest_edge(graph):
    l_edge = 0
    for i in graph:
        dicto = graph[i]
        for word in dicto.keys():
            #print(word)
            #print(dicto[word])
            #print(dicto[word]['weight'])
            if dicto[word]['weight'] > l_edge:
                l_edge = dicto[word]['weight']
    return l_edge



def degree_connectedness(graph):

    degreesum = 0
    mindegree = len(graph)
    maxdegree = 0
    for i in range(0,len(graph)):
        degreesum += graph.degree(i)
        if graph.degree(i) > maxdegree:
            maxdegree = graph.degree(i)
        if graph.degree(i) < mindegree:
            mindegree = graph.degree(i)
    degreesum = degreesum/len(graph)
    return degreesum, mindegree, maxdegree
    print(degreesum)
    print('min' , mindegree , 'max' , maxdegree)
    print('graphlength' ,len(graph))


def get_rand_index_format(dataset, labels):
    result = [-1 for x in range(len(dataset))]
    for idx, liste in enumerate(labels):
        for i in liste:
            if len(liste) > 1:
                result[i] = idx
    return result

# create graph out of clusters:
def create_mst_weighted(dataset, labels):
    mst = []
    temp_dataset = []
    for i in labels:
        if len(i) > 2:
            adjacency_matrix = np.zeros((len(dataset), len(dataset)))
            mytemplist = []
            for j in i:
                mytemplist += [j]
            for pts, idx in enumerate(mytemplist):
                for pts2 in range(idx, len(mytemplist)):
                    adjacency_matrix[pts][pts2] += mydist(dataset[pts], dataset[pts2])
            graph = nx.from_numpy_matrix(adjacency_matrix)
            mst += [nx.minimum_spanning_tree(graph)]
    return mst
        #graph2 = nx.from_scipy_sparse_matrix(csgraph_from_dense(m))
        #mst += [nx.minimum_spanning_tree(graph2)]# minimum spanning tree out of graph
    #for subgraph in mst:
        #graph = nx.disjoint_union(graph,subgraph)
        #graph.add_nodes_from(subgraph.nodes)
        #graph.add_edges_from(subgraph.edges)
        #graph = nx.union(graph, subgraph

def connectedness(minimumspanning):
    for graph in minimumspanning:
        longest = 0
        temp = graph_longest_edge(graph)
        if temp > longest:
            longest = temp
    return longest



