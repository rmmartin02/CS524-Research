from PIL import Image
import numpy as np

def main():

    im = Image.open('maria.jpg')
    #left,upper,right,lower
    #im = im.crop((int(im.width * .2),int(im.height * .2),int(im.width * .8),int(im.height * .8)))
    pixels = np.asarray(im)
    fixed = np.zeros((len(pixels),len(pixels[0])))
    for x in range(len(pixels)):
    	for y in range(len(pixels[x])):
    		if (int(pixels[x][y][0])-int(pixels[x][y][2]))<60:
    			fixed[x][y] = pixels[x][y][0]
    im = Image.fromarray(fixed.astype('uint8'),'L')		
    im.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()