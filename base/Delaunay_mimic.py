from scipy.spatial import Delaunay
from base.helperfunctions import *
import math
import time


#delaunay triangulation
def Delaunay_mimic(dataset, threshold, triangulation):
    """Delaunay mimics human clustering"""
    start_time = time.time()
    #tri = Delaunay(dataset)
    tri = triangulation
    #print(tri.simplices)
    dist = gen_all_connected(dataset, tri)
    dist.sort()


    helpdist = [None for i in range(len(dataset))]
    lauf = len(dataset)
    # kantenlängen
    for example in dist:
        if not helpdist[example[1]]:
            helpdist[example[1]] = example[0]
            lauf -= 1
        if not helpdist[example[2]]:
            helpdist[example[2]] = example[0]
            lauf -= 1
        if lauf <= 0:
            break

    # R_1 und R_2 Faktoren aus dem Paper
    reduced_tri = []
    for example in dist:
        r_1 = example[0]/helpdist[example[1]]
        r_2 = example[0]/helpdist[example[2]]

        r_erg = math.sqrt(r_1*r_2) # geometrisches Mittel
        if r_erg < threshold:
            reduced_tri += [(example[1], example[2])]
    labels = gen_labels(len(dataset), reduced_tri)
    #print(labels)
    #res, noise = gen_results_from_labels(dataset, labels)
    #print('Zeit', time.time()-start_time)

    #showres(res, noise)
    return labels




def find_neighbors(pindex, tri):
    return tri.vertex_neighbor_vertices[1][
           tri.vertex_neighbor_vertices[0][pindex]:tri.vertex_neighbor_vertices[0][pindex + 1]]

def gen_all_connected(dataset,tri):
    dist = []
    for i, p in enumerate(dataset):
        neighbors = find_neighbors(i,tri)
        dist += [(mydist(p, dataset[n]), i, n) for n in neighbors]
    return dist
#r - factor (parameter in algorithm)

