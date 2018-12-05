import numpy as np
from PIL import Image
import os
import multiprocessing as mp


def loop(a):
    arr = np.zeros(len(a))
    for i in range(len(a)):
        if (int(a[i][0])-int(a[i][2]))<60:
            arr[i] = a[i][0]
    return arr

def cleanImage(pixels):
    c_cpu = mp.cpu_count()
    #print("No. of processors : ", c_cpu)
    pool = mp.Pool(c_cpu)
    out = pool.map(loop, pixels)
    pool.close()
    return np.asarray(out)

def mAndCLoop(a,m,i):
    print(a,m,i)
    for i in range(len(a)):
        if (int(a[i][0])-int(a[i][2]))<60:
            m[i] = m[i]+(1/i[0])*(a[i][0]-m[i])
    return m

def meanAndClean(pixels,meanImage,i):
    c_cpu = mp.cpu_count()
    #print("No. of processors : ", c_cpu)
    pool = mp.Pool(c_cpu)
    iList = [i]*len(pixels)
    out = pool.map(mAndCLoop, pixels,meanImage,iList)
    pool.close()
    return np.asarray(out)


baseDir = '/scratch/rmmartin/tcdat/'

first = True
i = 1
for season in os.listdir(baseDir):
    print(season)
    for basin in os.listdir('{}/{}'.format(baseDir,season)):
        print(basin)
        for storm in os.listdir('{}/{}/{}'.format(baseDir,season,basin)):
            for img in os.listdir('{}/{}/{}/{}/ir/geo/1km_bw/'.format(baseDir,season,basin,storm)):
                im = Image.open('{}/{}/{}/{}/ir/geo/1km_bw/{}'.format(baseDir,season,basin,storm,img))
                im = im.crop((int(im.width * .2),int(im.height * .2),int(im.width * .8),int(im.height * .8)))
                
                if first:
                    meanImage = np.zeros(im.size)
                    first = False
                
                pixels = np.asarray(im)
                #fixed = cleanImage(pixels)
                
                #arr = np.asarray(im)
                #meanImage = meanImage+(1/i)*(fixed[:,:]-meanImage)
                meanImage = meanAndClean(pixels,meanImage,i)

                i+=1
    im = Image.fromarray(meanImage.astype('uint8'),'L')     
    im.show()