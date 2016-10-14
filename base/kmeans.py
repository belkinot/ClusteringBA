from base.helperfunctions import *



def kmeans(dataset, k):
    centre = []
    for i in range(0,k):
        centre += [dataset[i]]

    #solange clusterzentrne nicht gleich oder solange k-1 clusters nicht gleich oder n-iterationen
    n = 0
    clusters, centre = kmeansdistance(dataset, centre)
    oldclusters = 0
    while n < 300 and not (oldclusters == clusters):
        oldclusters = clusters
        clusters, centre = kmeansdistance(dataset, centre)
        n += 1
    print(clusters)
    print(n)
    clustering = []
    for example in clusters:
        helpcluster = [] #needs to be tuple???
        for iexample in example:
            helpcluster += dataset[iexample]
        clustering += [helpcluster]

    print(clustering)
    #showres(clustering)


def kmeansdistance(dataset, centre):
    exampleincluster = []
    for example in dataset:
        exampledist = []
        for i in range(0, len(centre)):
            exampledist += [mydist(example, centre[i])]
        exampleincluster += [exampledist.index(min(exampledist))]
        # clusterzuordnung
    clusters = [[] for x in range(0, len(centre))]
    for idx, i in enumerate(exampleincluster):
        clusters[i] += [[idx]]
    #zentrumsberechnung
    newcentre = 0
    for idx, _ in enumerate(clusters):
        for i in range(0, len(clusters[idx])):
            newcentre += dataset[clusters[idx][i]]
        newcentre = newcentre / len(clusters[idx])
        centre[idx] = newcentre
        newcentre = 0

    return clusters, centre
