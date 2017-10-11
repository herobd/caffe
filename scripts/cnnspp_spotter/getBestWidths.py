import sys
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import math

inFile1 = sys.argv[1]
inFile2 = sys.argv[2]
inFile3 = sys.argv[3]

###
segCharWidth=None
if len(sys.argv)>4:
    segFile = sys.argv[4]
    segCharWidth=defaultdict(list)
    with open(segFile) as f:
        allMean=0
        lines=f.readlines()
        for line in lines:
            ss = line.split(',')
            word = ss[0].lower()
            xStart = int(ss[2])
            xEnd = int(ss[4])
            for i in range(6,len(ss),2):
                char = word[(i-6)/2]
                width = int(ss[i+1])-int(ss[i])
                segCharWidth[char].append(width)
        for char in segCharWidth:
            mean=0
            for w in segCharWidth[char]:
                mean+=w
            mean/=len(segCharWidth[char])
            std=0
            for w in segCharWidth[char]:
                std+=(w-mean)*(w-mean)
            std = math.sqrt(std/len(segCharWidth[char]))
            #print char+': '+str(std)
            segCharWidth[char]=mean
            allMean+=mean+0.0
        allMean/=len(segCharWidth)

###

values=defaultdict(dict)
widthsS=set()
ngrams=[]
a=[]
allMean2=0
allCount2=0
def read(inFile):
    global ngrams,values, widthsS, allMean2, allCount2
    widths=[]
    with open(inFile) as f:
        lines = f.readlines()
        #a = []
        #for n in lines:
        #    ns = n.split(',')
        #    a.append([int(ns[1])])
        #    ngrams.append(ns[0])
        ngramsH = lines[0].split(',')[1:]
        for i in range(1,len(lines)):
            ns = lines[i].split(',')
            if ns[0]=='Ngram':
                break
            width = int(ns[0])
            widthsS.add(width)
            widths.append(width)
            ns = ns[1:]
            for ii in range(len(ns)):
                v = ns[ii]
                values[ngramsH[ii]][width]=float(v)
        ngrams += ngramsH
        #widths = [x for x in widthsS]
        #widths.sort()
        for ngram in ngramsH:
            prevV2=-1
            prevV1=-1
            prevW=-1
            bestSMAP=-1
            bestWidth=-1
            for width in widths:
                curV=values[ngram][width]
                if prevV1!=-1:
                    if prevV2!=-1:
                        avg = (prevV2+prevV1+curV)/3.0
                    else:
                        avg = (prevV1+curV)/2.0
                    #print str(prevW)+' '+str(avg)
                    if avg>bestSMAP:
                        bestSMAP=avg
                        bestWidth=prevW
                prevV2=prevV1
                prevV1=curV
                prevW=width

            avg = (prevV1+prevV2)/2.0
            if avg>bestSMAP:
                bestSMAP=avg
                bestWidth=prevW
            if avg>=0:
                a.append([bestWidth])
                allMean2+=(0.0+bestWidth)/len(ngram)
                allCount2+=1
            else:
                a.append([-1])
            #print ngram+' '+str(bestWidth)
            #ngrams.append(ngram)


read(inFile1)
read(inFile2)
read(inFile3)

allMean2/=allCount2+0.0
for i in range(len(ngrams)):
    if a[i][0]<0:
        ngram = ngrams[i].strip().lower()
        segWidth=len(ngram)*allMean2
        if segCharWidth is not None:
            segWidth = 0
            for c in ngram:
                segWidth += segCharWidth[c]
        a[i][0] = round((len(ngram)*allMean2 + segWidth)/2.0)


#get smoothed max
#newV=defaultdict(list)

vs = np.array(a)
kmeans = KMeans(n_clusters=10, random_state=0).fit(vs)

averageError=0
newV = kmeans.labels_
for i in range(len(ngrams)):
    #closets mult of 4
    clust = kmeans.cluster_centers_[newV[i]][0]
    off = clust - 4*(int(clust)/4)
    if off<2:
        clust-=off
    else:
        clust+=4-off

    segWidth=a[i][0]
    if segCharWidth is not None:
        segWidth = 0
        for c in ngrams[i].strip().lower():
            segWidth += segCharWidth[c]
        #segWidth = str(segWidth)+', '
    #print ngrams[i].strip().lower()+', '+segWidth+str(a[i][0])+', '+str(int(clust))
    print ngrams[i].strip().upper()
    print int(round((a[i][0]+segWidth)/2.0))
    print int(clust)
#print allMean
#print allMean2

