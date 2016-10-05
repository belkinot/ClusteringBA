import numpy as np
import matplotlib.pyplot as plt


def mydistance (p1,p2):
    return 1/2*np.absolute(p1-p2)
    #componentwise distance


def mydist (p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def load_mydataset(reverse=False):
    name = 'moons'
    data = np.genfromtxt(name + '.csv', delimiter=',')
    dataset = data[:, :2]
    #dataset2 = np.concatenate((dataset[::2],dataset[1::2]), axis=0)
    if reverse:
        dataset = dataset[::-1]
    return dataset

def load_dataset_with_labels():
    name = 'pathbased'
    data = np.genfromtxt(name + '.csv', delimiter=',')
    return data


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



def reader(test):
    with open(test, "r") as f:
        for line in f.readlines():
            x = eval(line)
    return x



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
