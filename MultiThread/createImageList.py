import os
import sys

imgDir = sys.argv[2]
file = sys.argv[1]

with open(file,'w') as f:
	for image in os.listdir(imgDir):
		f.write(imgDir+'/'+image+'\n')