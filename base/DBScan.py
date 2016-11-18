def DBSCANlabels_to_Labels(dataset, labels):
    labellist = [set() for _ in range(max(labels)+2)]
    for idx,i in enumerate(labels):
        #print(type(labellist[i]))
        mySet = {idx}
        labellist[i+1] |= mySet
    return labellist


def KMEANSlabels_to_Labels(labels):
    labellist = [set() for _ in range(max(labels)+1)]
    for idx, i in enumerate(labels):
        mySet = {idx}
        labellist[i] |= mySet
    return labellist
