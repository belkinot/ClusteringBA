from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import numpy as np




def load_mydataset():
    name = 'aggregation'
    data = np.genfromtxt(name + '.csv', delimiter=',')
    dataset = data[:, :2]
    return dataset

def hierarchical(dataset):

    Z = linkage(dataset,'ward')
    c, coph_dists = cophenet(Z,pdist(dataset))
    c


    plt.figure(figsize=(25,10))
    plt.title('Hierarchical Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z, leaf_rotation=90.,
               leaf_font_size=8., truncate_mode='lastp' , p = 12, show_leaf_counts=False, show_contracted=True)
    plt.show()

    clusters = fcluster(Z, 6, criterion='maxclust')
    print(clusters)
    plt.figure(figsize=(10,8))
    plt.scatter(dataset[:,0], dataset[:,1], c=clusters, cmap='prism')
    plt.show()

if __name__ == '__main__':
    dataset = load_mydataset()
    hierarchical(dataset)