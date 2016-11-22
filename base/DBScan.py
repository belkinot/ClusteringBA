def DBSCANlabels_to_Labels(labels):
    """transformiert Sklearn labels in anderes labelformat"""
    if min(labels) == -1: # wenn noise (-1 label) vorhanden
        labellist = [set() for _ in range(max(labels)+2)]
        for idx,i in enumerate(labels):
        #print(type(labellist[i]))
            mySet = {idx}
            labellist[i+1] |= mySet
    else:
        labellist = KMEANSlabels_to_Labels(labels)
    return labellist


def KMEANSlabels_to_Labels(labels):
    """transformiert SKlearn Labels in anderes labelformat"""
    labellist = [set() for _ in range(max(labels)+1)]
    for idx, i in enumerate(labels):
        mySet = {idx}
        labellist[i] |= mySet
    return labellist
