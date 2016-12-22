

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
#%matplotlib inline
caffe_root = '/home/brianld/caffe/'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe.proto import caffe_pb2
import leveldb

import cv2


MODEL_FILE = '/home/brianld/caffe/models/siamese/bi_siamsese2_deploy.prototxt'
PRETRAINED_FILE = '/scratch/brianld/data/GW/net/bi_siamese2_iter_22000.caffemodel' 
LABEL_FILE = '/scratch/brianld/data/top100_bigrams_in_freq_order.txt'
DATA_FILE='/scratch/brianld/data/GW/q_bi_train.txt'
IMG_DIR='/scratch/brianld/data/GW/segmentations/'


cO = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
     '#ff00ff', '#990000', '#999900', '#009900', '#009999',
     '#000099', '#990099', '#550000', '#555500', '#005500',
     '#005555', '#000055', '#550055']
c=[]
i=0
with open(LABEL_FILE, 'r') as f:
    bigrams=f.readlines()
    for j in range(len(bigrams)):
        c.append(cO[i])
        i = (i+1)%len(cO)
        bigrams[j]=bigrams[j].strip()
#print(bigrams)
#get data
"""
db = leveldb.LevelDB('/scratch/brianld/data/GW/siamese_bi_valid_leveldb')
datum = caffe_pb2.Datum()

n=0
caffe_in = np.array((0,1,52,52))
for key, value in db.RangeIter():
    n+=2
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum) * 0.00390625
    #print(data.shape)
    data = np.reshape(data,(2,1,52,52))
    assert (data.shape[1]==1)
    caffe_in = np.append(caffe_in, data, axis=0) 
    #CxHxW to HxWxC in cv2
    #image = np.transpose(data, (1,2,0))
assert (n==caffe_in.shape[0])
print('batch of size: '+str(n))
assert(n>0)
"""
n=90
labels=[]
caffe_in = np.empty((n,1,52,52))
i=0
with open(DATA_FILE, 'r') as f:
    datas=f.readlines()
    for inst in datas:
        inst = inst.split()
        label = inst[1].upper()
        try:
            ind = bigrams.index(label)
            #if ind is not ValueError:
            labels.append(ind)
            data=cv2.imread(IMG_DIR+inst[0], 0) * 0.00390625
            data = cv2.resize(data,(52,52))
            caffe_in[i]=np.reshape(data,(1,52,52))
            i+=1
            if i>= n:
                break
        except ValueError:
            continue
assert(i==n),str(i)
# reshape and preprocess
#caffe_in = raw_data.reshape(n, 1, 52, 52) * 0.00390625 # manually scale data instead of using `caffe.io.Transformer`
#print(caffe_in.shape)
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)
#print(net.blobs['data'].data.shape)
net.blobs['data'].reshape(n, 1, 52,52) 
#print(net.blobs['data'].data.shape)
out = net.forward_all(data=caffe_in)



feat = out['feat']
#print(feat.shape)
f = plt.figure(figsize=(16,9))
labels=np.array(labels)
#print(labels)
results = PCA(feat)
tt=['.','+','o','v','^','x','d','D','*','h','s']
for i in range(100):
    #xs=feat[labels==i,0].flatten()
    #ys=feat[labels==i,1].flatten()
    xs=results.Y[labels==i,0].flatten()
    ys=results.Y[labels==i,1].flatten()
    #print(i,xs,ys)
    color=c[i]
    marker = tt[i%len(tt)]
    plt.plot(xs, ys, marker, c=color)

plt.legend(bigrams)
plt.grid()
plt.show()


