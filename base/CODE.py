#plot
def myplots(points):
    x, y =  zip(*points)
    plt.scatter(x, y)
    plt.show()


def plotdemo(points, contour):
    x,y = (zip(*points))
    plt.scatter (x,y, s= 10, c = 'blue', alpha = 1)
    x2,y2 = (zip(*contour))
    plt.scatter(x2, y2, s=10, c='red', alpha=0.7)
    plt.show()


def plotdemo2(data,cluster,noncluster):
    x,y = (zip(*data))
    plt.scatter (x,y , s=10, c = 'blue', alpha = 1)
    x2,y2 = (zip(*cluster))
    plt.scatter(x2,y2, s= 10, c = 'red', alpha = 0.7)
    x3,y3 = (zip(*noncluster))
    plt.scatter(x3,y3, s= 10, c = 'green', alpha = 0.5)
    plt.show()


def codeSpecial(data):
    mindist = []
    vals = []
    gridhelp = []
    contour = []
    for i, pts in enumerate(data):
        temp = []
        j = i + 1
        for pts2 in data:
            if not (pts == pts2).all():
                temp += [mydist(pts, pts2)]
            if ( j < len(data)):
                gridhelp += [np.add(data[i],1/2 * np.subtract(data[j],data[i]))]
                j = j + 1

        mindist += [min(temp)]
        vals += [multivariate_normal(mean=pts, cov=[[mindist[i], 0], [0, mindist[i]]], allow_singular=True)]
    print('vals finished')
    print(len(gridhelp), len(data))

    for points in gridhelp:
        result = 0
        for j in vals:
            result = j.pdf(points) + result
            if result <= 1:  # less calcualting, threshold erreicht genügt
                break
        if result > 1:
            contour += [points]
    print ('contour len:',len(contour))
    clusterpts = []
    for pts in data:
        result = 0
        for j in vals:
            result = j.pdf(pts) + result
            if result >= 1:
                clusterpts += [pts]
                break
    print('clusterpts', len(clusterpts))
    if (len(contour) == 0):
        plotdemo(data, clusterpts)
    else:
        plotdemo2(data, clusterpts, contour)


def codeStandard(data):
    mindist = []#minimum distance for each points
    #mindistindex = [] #index of point with minimum distance
    tempcovariance = []
    mintotaldistance = 0  # gitternetzlinienabstand
    minx = data[0][0]
    miny = data[0][1]
    maxx = data[0][0]
    maxy = data[0][1]
    vals = []
    for i,pts in enumerate(data):
        temp = []
        for pts2 in data:
            if not (pts==pts2).all():
                temp += [mydist(pts, pts2)]
        mindist += [min(temp)]
        #gitternetz hilfskonstrukt
        if pts[0] < minx:
            minx = pts[0]
        if pts[0] > maxx:
            maxx = pts[0]
        if pts[1] < miny:
            miny = pts[1]
        if pts[1] > maxy:
            maxy = pts[1]
         #mindistindex += [temp.index(min(temp))] index of minimum distance point
        #temp covariance values
        #tempcovariance += [mydistance(pts, data[mindistindex[i]])] # kleinster index abstand komponentenweise
        #vals = multivariate_normal(mean=pts, cov=[[tempcovariance[i][0],0],[0,tempcovariance[i][1]]],allow_singular=True) #covariance komponentenweise
        vals  += [multivariate_normal(mean=pts, cov=[[mindist[i],0],[0, mindist[i]]], allow_singular = True)]

    #get grid
    mintotaldistance = min(mindist)
    minx = minx-mintotaldistance/2
    miny = miny-mintotaldistance/2
    maxx = maxx+mintotaldistance
    maxy = maxy+mintotaldistance
    print('minx',minx,'maxx',maxx, 'miny',miny,maxy, mintotaldistance)

    #numpy magic http://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy FANCY creates grid
    gridhelp = np.mgrid[minx:maxx:mintotaldistance, miny:maxy:mintotaldistance].reshape(2,-1).T
    print('normal grid', len(gridhelp))
    # print(gridhelp[0])
    myttime = time.time()
    clusterres = []
    # #without GRID
    # for res in data:
    #     result = []
    #     for i in vals:
    #         result += [i.pdf(res)]
    #     clusterres += [np.sum(result)]
    # clusteredpoints = 0
    # contour = []
    # for i,value in enumerate(data):
    #     if clusterres[i] > 1:
    #         print(clusterres[i])
    #         contour += [value.tolist()]
    #         clusteredpoints += 1


    clusteredpoints = 0
    contour = []
    contourzero = []
    for idx, pointsgrid in enumerate(gridhelp):
        result = 0
        for j in vals:
            result = j.pdf(pointsgrid) + result
            if result >= 1: # less calcualting, threshold erreicht genügt
               clusteredpoints += 1
               contour += [pointsgrid]
               break
        #clusterres += [result]
        #if 0.5 <= result < 1:
         #   contourzero += [pointsgrid]
    print("zeit für epic schleife", time.time()-myttime)
    print(' ', clusterres)

    #print(contour)
    print(clusteredpoints, "Anzahl punkte in clustern")
    #myplots(contour)
    if len(contourzero)> 0:
        plotdemo(data, contourzero)
    else:
        print("no zeroes")
    plotdemo(data,contour)



