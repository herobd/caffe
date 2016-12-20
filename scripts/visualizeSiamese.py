

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
caffe_root = '/home/brianld/caffe/'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe.proto import caffe_pb2
import leveldb


MODEL_FILE = '/home/brianld/caffe/models/siamese/bi_siamsese_train.prototxt'
# decrease if you want to preview during training
PRETRAINED_FILE = '/scratch/brianld/data/GW/net/bi_siamese_iter_27000.caffemodel' 
LABEL_FILE = '/scratch/brianld/data/top100_bigrams_in_freq_order.txt'
caffe.set_mode_cpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)

#get data
db = leveldb.LevelDB('leveldb_file')
datum = caffe_pb2.Datum()

n=0
caffe_in = np.array((0,1,52,52))
for key, value in db.RangeIter():
    n+=1
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum) * 0.00390625
    assert (data.shape[0]==1),data.shape+''
    caffe_in = np.append(caffe_in, data[1,6:59,3:56], axis=0) 
    #CxHxW to HxWxC in cv2
    #image = np.transpose(data, (1,2,0))


# reshape and preprocess
#caffe_in = raw_data.reshape(n, 1, 52, 52) * 0.00390625 # manually scale data instead of using `caffe.io.Transformer`
out = net.forward_all(data=caffe_in)



feat = out['feat']
f = plt.figure(figsize=(16,9))
cO = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
     '#ff00ff', '#990000', '#999900', '#009900', '#009999',
     '#000099', '#990099', '#550000', '#555500', '#005500',
     '#005555', '#000055', '#550055']
c=[]
i=0
with open(LABEL_FILE, 'r') as f:
    bigrams=f.readlines()
    c.append(cO[i])
    i = (i+1)%len(cO)
for i in range(10):
        plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])

plt.legend(bigrams)
plt.grid()
plt.show()


