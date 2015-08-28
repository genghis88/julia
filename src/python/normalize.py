import pandas as pd
from PIL import Image
from os import listdir
from os.path import isfile, join
import sys

inputDir = sys.argv[1]
outputDir = sys.argv[2]

files = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]

images = []
for fName in files:
  fileName = join(inputDir,fName)
  im = Image.open(fileName)
  im = im.resize((20, 20), Image.ANTIALIAS)
  #convert from color to black and white
  im = im.convert('L')
  images.append(im)
  newFileName = outputDir + fName[0:fName.index('.Bmp')] + '.png'
  im.save(newFileName)
