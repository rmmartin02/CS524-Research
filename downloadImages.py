import sys, os
import urllib.request
import multiprocessing
from PIL import Image
import requests
import time
import datetime
import sys

baseDir = ''

def downloadImage(url):
    file_name = '{}/{}'.format(baseDir,url.split('/')[-1])
    #print(file_name)
    if os.path.exists(file_name):
        try:
            Image.open(file_name).load()
            print('Image {} already exists. Skipping download.'.format(file_name))
            return
        except:
            print('Image {} is corrupted. Redownloading'.format(file_name))
    print('Downloading: {}'.format(url))
    while True:
        try:
            pic = requests.get(url)
            if pic.status_code == 200:
                    with open(file_name,'wb') as f:
                        f.write(pic.content)
            Image.open(file_name).load()
            print('Finished Downloading: {}'.format(file_name))
            return
        except:
            time.sleep(.1)
            print('Retrying download: {}'.format(file_name))

def loader(file):
    url_list = []
    with open(file,'r') as f:
        url_list = f.readlines()
    for i in range(len(url_list)):
        url_list[i] = url_list[i].rstrip()
    print(multiprocessing.cpu_count() * 2)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() * 2)  # Num of CPUs
    pool.map(downloadImage, url_list)
    pool.close()
    pool.terminate()

    print('Finished all downloads')

# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    baseDir = sys.argv[1]
    urlFile = sys.argv[2]
    loader(urlFile)