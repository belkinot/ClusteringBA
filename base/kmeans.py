from base.helperfunctions import *



def kmeans(dataset, k):
    centre = []
    for i in range(0,k):
        centre += [dataset[i]]

    #solange clusterzentrne nicht gleich oder solange k-1 clusters nicht gleich

    clusters, centre = kmeansdistance(dataset, centre)
    clusters2 = kmeansdistance(dataset, centre)
    print(clusters2)


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
