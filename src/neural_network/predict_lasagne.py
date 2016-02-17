import pickle
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np

inputDir = sys.argv[1]
modelFile = sys.argv[2]
predictionsFile = sys.argv[3]

def writeToFile(files, y, predictionsFile):
  with open(predictionsFile,'w') as output:
    output.write('ID,Class\n');
    for num in range(len(files)):
      f = files[num]
      name = f[:f.index('.png')]
      #output.write(name + ',' + str(y[num]) + '\n')
      output.write(name + ',' + chr(int(y[num])) + '\n')

files = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]

files = map(lambda fileNumber: str(fileNumber) + '.png', sorted(map(lambda name: int(name[:name.index('.png')]), files)))

num_samples = len(files)
imageSize = 1024

xTest = np.zeros((num_samples, imageSize))
for fName in files:
  fileName = join(inputDir,fName)
  im = Image.open(fileName)
  index = int(fName[:fName.index('.png')])
  xTest[index - 6284, :] = np.reshape(im, (1, imageSize))

xTest /= xTest.std(axis = None)
xTest -= xTest.mean()

xTest = xTest.reshape(xTest.shape[0], 1, 32, 32).astype('float32')

model = open(modelFile,'rb')
net = pickle.load(model)
y = net.predict(xTest)
print y
writeToFile(files,y,predictionsFile)
