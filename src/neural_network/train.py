import pandas as pd
from PIL import Image
import pickle
from os.path import join
import sys
import numpy as np

labelsFile = sys.argv[1]
normalizedDataDirectory = sys.argv[2]
pickleFile = sys.argv[3]

trainingLabels = pd.read_csv(labelsFile)

imageSize = 1024
xTrain = np.zeros((trainingLabels.shape[0], imageSize))
for index, row in trainingLabels.iterrows():
  fileName = join(normalizedDataDirectory, str(row['ID']) + '.png')
  im = Image.open(fileName)
  xTrain[index, :] = np.reshape(im, (1, imageSize))

y = map(ord, trainingLabels['Class'])

from keras.models import Sequential

model = Sequential()

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

xTrain /= xTrain.std(axis = None)
xTrain -= xTrain.mean()

xTrain = xTrain.reshape(xTrain.shape[0], 1, 32, 32).astype('float32')

model.add(Convolution2D(input_shape=(1, 32, 32), nb_row=5, nb_col=5, nb_filter=32, init='uniform', dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))
model.add(Dropout(0.2))
model.add(Convolution2D(nb_row=5, nb_col=5, nb_filter=64, dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))
model.add(Dropout(0.2))
model.add(Convolution2D(nb_row=5, nb_col=5, nb_filter=128, dim_ordering='th'))
model.add(Flatten())
model.add(Dense(500, input_dim=1024, activation='softmax'))
model.add(Dense(1, input_dim=500, activation='softmax'))

sgd = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss='mae', class_mode='categorical')
model.fit(xTrain, y, nb_epoch=1, batch_size=1024, validation_split=0.2, verbose=1)
#objective_score = model.evaluate(xTrain, y, batch_size=64)
#print objective_score

predicted_y = model.predict_classes(xTrain, batch_size=1024)
predicted_y = map(chr, predicted_y)
print predicted_y
error = np.sum(trainingLabels['Class'] == predicted_y)
print error
for index, val in np.ndenumerate(predicted_y):
  print str(index) + ' ' + val + ' ' + trainingLabels['Class'].ix[index]

'''import neurolab as nl
dimensions = [[0,255] for i in range(400)]
num_samples = len(trainingData)
trainingData = np.reshape(trainingData, (num_samples, 400))
y = np.reshape(y, (num_samples, 1))
y = y / np.linalg.norm(y)
model = nl.net.newff(dimensions, [400, 400, 400, 1])
model.trainf = nl.train.train_gd
error = model.train(trainingData, y, epochs=100, show=10)
print 'error ' + str(error)'''

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(model, f)
