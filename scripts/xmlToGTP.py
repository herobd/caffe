import re
import sys

inFile = sys.argv[1]
testName = sys.argv[2]
outTrainFile = sys.argv[3]
outTestFile = sys.argv[4]


with open(inFile) as f:
    lines = f.read().splitlines()

outTrain = open(outTrainFile,'w')
outTest = open(outTestFile,'w')

for line in lines:
    m = re.match(r'<spot word="([^"]+)" image="([^"]+)" x="(\d+)" y="(\d+)" w="(\d+)" h="(\d+)" ?/>',line)
    if m:
        char = m.group(1).lower()
        image = m.group(2)
        x1 = int(m.group(3))
        y1 = int(m.group(4))
        x2 = x1+int(m.group(5))-1
        y2 = y1+int(m.group(6))-1

        writeTo=outTrain
        if image==testName:
            writeTo=outTest

        writeTo.write(image+' '+str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2)+' '+char+'\n')

outTrain.close()
outTest.close()


