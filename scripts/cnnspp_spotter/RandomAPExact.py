import sys

def apExact(N,R):
    ap=0.0
    for i in range(1,R+1):
        for n in range(i,N-R+i+1):
            ap += dhyper(i,R,N-R,n)*pow(i/(0.0+n),2)
    ap /= R
    return ap


gtFile = sys.argv[1]
ngramFiles = []
for i in range(2,len(sys.argv)):
    ngramFiles.append(sys.argv[i])

words=[]
if gtFile[-3:]=='gtp':
    with open(gtFile) as f:
        for line in f.readlines():
            m = re.match(r'(.*\.\w+) (\d+) (\d+) (\d+) (\d+) (.*)',line.strip())
            words.append(m.group(6).lower())
else:
    with open(gtFile) as f:
        for line in f.readlines():
            m = re.match(r'(.*\.\w+) (.*)',line.strip())
            words.append(m.group(2).lower())

for ngramFile in ngramFiles:
    print ngramFile

