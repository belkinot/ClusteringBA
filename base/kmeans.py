from base.helperfunctions import *
import random

def kmeans(dataset, k):
    centre = [dataset[i] for i in range(k)]
    # nutze zufallswerte :P
    #centre = [dataset[i] for i in random.sample(range(len(dataset)), k)]
    #solange clusterzentrne nicht gleich oder solange k-1 clusters nicht gleich oder n-iterationen
    n = 0
    clusters, centre = kmeansdistance(dataset, centre)
    test_centre = tuple()
    print(centre)
    while n < 300 and not test_centre == centre:
        test_centre = centre[:]
        clusters, centre = kmeansdistance(dataset, centre)
        n += 1
    print(clusters)
    print(n)
    #cluster_labels = [set(x) for x in clusters]
    #clustering, noise = gen_results_from_labels(dataset, cluster_labels)
    showres(clusters)


def kmeansdistance(dataset, centre):
    clusters = [[] for x in range(len(centre))]
    for example in dataset:
        exampledist = []
        for i in range(0, len(centre)):
            exampledist += [mydist(example, centre[i])]
        clusters[exampledist.index(min(exampledist))] += [example]
    print(clusters)
    for idx, example in enumerate(clusters):
        #x, y = np.sum(example, axis=0)
        #newcentre = (x/len(example), y/len(example))
        newcentre = tuple(x/len(example) for x in np.sum(example, axis=0))
        centre[idx] = newcentre

    return clusters, centre
