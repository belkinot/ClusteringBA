from base.helperfunctions import *

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

