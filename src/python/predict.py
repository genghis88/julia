import pickle
from PIL import Image
from os import listdir
from os.path import isfile, join
import sys

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
      output.write(name + ',' + classMapping[y[num]] + '\n')

files = [ f for f in listdir(inputDir) if isfile(join(inputDir,f)) ]

testingData = []
for fName in files:
  fileName = join(inputDir,fName)
  im = Image.open(fileName)
  data = list(im.getdata())
  testingData.append(data)

model = open(modelFile,'rb')
clf = pickle.load(model)
y = clf.predict(testingData)
writeToFile(files,y,predictionsFile)
