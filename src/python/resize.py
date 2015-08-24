import pandas as pd
from PIL import Image

trainingLabels = pd.read_csv('./data/trainLabels.csv')
#print trainingLabels

images = []
for index, row in trainingLabels.iterrows():
  #print str(row['ID']) + ' ' + row['Class']
  fileName = './data/train/' + str(row['ID']) + '.Bmp'
  im = Image.open(fileName)
  im = im.resize((20, 20), Image.ANTIALIAS)
  images.append(im)
  newFileName = './data/train/resized/' + str(row['ID']) + '.bmp'
  im.save(newFileName)
