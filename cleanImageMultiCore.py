import numpy as np
import multiprocessing as mp
from PIL import Image


def loop(a):
    arr = np.zeros(len(a))
    for i in range(len(a)):
        if (int(a[i][0])-int(a[i][2]))<60:
            arr[i] = a[i][0]
    return arr

def main():
    im = Image.open('maria.jpg')
    pixels = np.asarray(im)
    c_cpu = mp.cpu_count()
    n_loop = 10
    print("No. of processors : ", c_cpu)
    pool = mp.Pool(c_cpu)
    out = pool.map(loop, pixels)
    fixed = np.asarray(out)
    im = Image.fromarray(fixed.astype('uint8'),'L')     
    im.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()