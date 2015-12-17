import sys
import re

splitFile = open(sys.argv[1])
splitLines = splitFile.readlines()
imagesFile = open(sys.argv[2])
imagesLines = imagesFile.readlines()

train=[]
test=[]
for line in splitLines:
    m = re.search(r'(\d+)\s+([01])',line)
    imgIndex=int(m.group(1))
    split = int(m.group(2))
    if split==1:
        train.append(imgIndex)
    else:
        test.append(imgIndex)

paths={}
for line in imagesLines:
    m = re.search(r'(\d+)\s+(.+)\s',line)
    imgIndex=int(m.group(1))
    path = m.group(2)
    paths[imgIndex]=path

classFile = open(sys.argv[3])
classLines = classFile.readlines()
attrFile = open(sys.argv[4])
attrLines = attrFile.readlines()

binaryAttrForImg={}
for line in attrLines:
    m = re.search(r'(\d+)\s+(\d+)\s+([01])\s+([1234])\s+\d*\.?\d+',line)
    imgIndex = int(m.group(1))
    attrIndex = int(m.group(2))
    isPresent = int(m.group(3))
    confidence = int(m.group(4))
    if imgIndex not in binaryAttrForImg:
        binaryAttrForImg[imgIndex]={}
    if isPresent==1 and confidence>2:
        binaryAttrForImg[imgIndex][attrIndex]=1
    else:
        binaryAttrForImg[imgIndex][attrIndex]=0

classes={}
for line in classLines:
    m = re.search(r'(\d+)\s+(\d+)',line)
    imgIndex = int(m.group(1))
    cls = int(m.group(2))
    classes[imgIndex]=cls

trainFile = open(sys.argv[-2],'w')
testFile = open(sys.argv[-1],'w')

for imgIndex in train:
    trainFile.write(paths[imgIndex]+' ')
    trainFile.write(str(classes[imgIndex]))
    for a in range(1,313):
        trainFile.write(' '+str(binaryAttrForImg[imgIndex][a]))
    trainFile.write('\n')

for imgIndex in test:
    testFile.write(paths[imgIndex]+' ')
    testFile.write(str(classes[imgIndex]))
    for a in range(1,313):
        testFile.write(' '+str(binaryAttrForImg[imgIndex][a]))
    testFile.write('\n')
#for i in xrange(len(lines)):
#	if i%every==0:
#		valFile.write(lines[i].strip())
#                for a in range(1,313):
#                    valFile.write(' '+str(binaryAttrForImg[i+1][a]))
 #               valFile.write('\n')
#	else:
#		trainFile.write(lines[i].strip())
#                for a in range(1,313):
#                    trainFile.write(' '+str(binaryAttrForImg[i+1][a]))
#                trainFile.write('\n')

trainFile.close()
testFile.close()
