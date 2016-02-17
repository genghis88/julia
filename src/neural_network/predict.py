import pickle
from PIL import Image
from os import listdir
from os.path import isfile, join
import sys
import numpy as np

inputDir = sys.argv[1]
modelFile = sys.argv[2]
predictionsFile = sys.argv[3]

classMapping = {}
for i in range(26):
  classMapping[i] = chr(ord('a') + i)

for i in range(26):
  classMapping[i + 26] = chr(ord('A') + i)

for i in range(10):
  classMapping[i + 52] = chr(ord('0') + i)

def writeToFile(files, y, predictionsFile):
  with open(predictionsFile,'w') as output:
    output.write('ID,Class\n');
    for num in range(len(files)):
      f = files[num]
      name = f[:f.index('.png')]
      #output.write(name + ',' + str(y[num]) + '\n')
      output.write(name + ',' + classMapping[y[num]] + '\n')

files = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]

files = map(lambda fileNumber: str(fileNumber) + '.png', sorted(map(lambda name: int(name[:name.index('.png')]), files)))

testingData = []
for fName in files:
  fileName = join(inputDir,fName)
  im = Image.open(fileName)
  data = list(im.getdata())
  testingData.append(data)

model = open(modelFile,'rb')
clf = pickle.load(model)
num_samples = len(testingData)
testingData = np.reshape(testingData, (num_samples, 1, 32, 32))
y = clf.predict_classes(testingData, batch_size=64)
print y
writeToFile(files,y,predictionsFile)
