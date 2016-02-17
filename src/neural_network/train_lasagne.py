import pandas as pd
from PIL import Image
import pickle
from os.path import join
import sys
import numpy as np
from lasagne import layers
from lasagne.init import Constant
from lasagne.nonlinearities import softmax, sigmoid, rectify, tanh, ScaledTanH, elu, identity, softplus, leaky_rectify
from nolearn.lasagne import NeuralNet, BatchIterator

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

xTrain /= xTrain.std(axis = None)
xTrain -= xTrain.mean()

xTrain = xTrain.reshape(xTrain.shape[0], 1, 32, 32).astype('float32')

b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1

b = b.flatten()

inputLayer = layers.InputLayer(shape=(None, 1, 32, 32))

#Localization network
loc1Layer = layers.Conv2DLayer(inputLayer, num_filters=32, filter_size=(5,5))
loc2Layer = layers.MaxPool2DLayer(loc1Layer, pool_size=(2,2))
loc3Layer = layers.Conv2DLayer(loc2Layer, num_filters=64, filter_size=(5,5))
loc4Layer = layers.MaxPool2DLayer(loc3Layer, pool_size=(2,2))
loc5Layer = layers.Conv2DLayer(loc4Layer, num_filters=128, filter_size=(4,4))
loc6Layer = layers.DenseLayer(loc5Layer, num_units=512, nonlinearity=rectify)
#loc7Layer = layers.DenseLayer(loc6Layer, num_units=256, nonlinearity=tanh)
#loc8Layer = layers.DenseLayer(loc7Layer, num_units=128, nonlinearity=tanh)
#loc9Layer = layers.DenseLayer(loc8Layer, num_units=64, nonlinearity=elu)
locOutLayer = layers.DenseLayer(loc6Layer, num_units=6, nonlinearity=identity)

#Transformer network
transformLayer = layers.TransformerLayer(inputLayer, locOutLayer, downsample_factor=1.0)
#transformLayer = layers.TransformerLayer(inputLayer, locOutLayer, downsample_factor=2.0)
print 'Transformer layer output shape ', transformLayer.output_shape

#Classification network
conv1Layer = layers.Conv2DLayer(transformLayer, num_filters=32, filter_size=(3,3))
pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(2,2))  
dropout1Layer = layers.DropoutLayer(pool1Layer, p=0.2)

conv2Layer = layers.Conv2DLayer(dropout1Layer, num_filters=64, filter_size=(4,4))
pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2))  
dropout2Layer = layers.DropoutLayer(pool2Layer, p=0.2)

conv3Layer = layers.Conv2DLayer(dropout2Layer, num_filters=128, filter_size=(3,3))
#pool3Layer = layers.MaxPool2DLayer(conv3Layer, pool_size=(2,2))  
#dropout3Layer = layers.DropoutLayer(pool3Layer, p=0.2)

#conv4Layer = layers.Conv2DLayer(dropout3Layer, num_filters=256, filter_size=(2,2))

hidden3Layer = layers.DenseLayer(conv3Layer, num_units=2048, nonlinearity=tanh)
hidden4Layer = layers.DenseLayer(hidden3Layer, num_units=1024, nonlinearity=tanh)
hidden5Layer = layers.DenseLayer(hidden4Layer, num_units=512, nonlinearity=tanh)
hidden6Layer = layers.DenseLayer(hidden5Layer, num_units=256, nonlinearity=tanh)
hidden7Layer = layers.DenseLayer(hidden6Layer, num_units=128, nonlinearity=tanh)
outputLayer = layers.DenseLayer(hidden7Layer, num_units=62, nonlinearity=softmax)

'''conv1Layer = layers.Conv2DLayer(transformLayer, num_filters=32, filter_size=(3,3))
pool1Layer = layers.MaxPool2DLayer(conv1Layer, pool_size=(2,2))  
dropout1Layer = layers.DropoutLayer(pool1Layer, p=0.2)

conv2Layer = layers.Conv2DLayer(dropout1Layer, num_filters=64, filter_size=(2,2))
pool2Layer = layers.MaxPool2DLayer(conv2Layer, pool_size=(2,2))  
dropout2Layer = layers.DropoutLayer(pool2Layer, p=0.2)

conv3Layer = layers.Conv2DLayer(dropout2Layer, num_filters=128, filter_size=(2,2))

hidden5Layer = layers.DenseLayer(conv3Layer, num_units=512, nonlinearity=tanh)
hidden6Layer = layers.DenseLayer(hidden5Layer, num_units=256, nonlinearity=tanh)
hidden7Layer = layers.DenseLayer(hidden6Layer, num_units=128, nonlinearity=tanh)
outputLayer = layers.DenseLayer(hidden7Layer, num_units=62, nonlinearity=softmax)'''

net = NeuralNet(
  layers = outputLayer,
  update_learning_rate = 0.01,
  update_momentum = 0.9,
  
  batch_iterator_train = BatchIterator(batch_size = 100),
  batch_iterator_test = BatchIterator(batch_size = 100),
  
  use_label_encoder = True,
  #use_label_encoder = False,
  regression = False,
  max_epochs = 100,
  verbose = 1
)

'''
net = NeuralNet(
  layers = [
    ('input', layers.InputLayer),
    #('hidden1', layers.DenseLayer),
    #('transform', transformLayer),
    ('conv1', layers.Conv2DLayer),
    ('pool1', layers.MaxPool2DLayer),
    ('dropout1', layers.DropoutLayer),
    ('conv2', layers.Conv2DLayer),
    ('pool2', layers.MaxPool2DLayer),
    ('dropout2', layers.DropoutLayer),
    ('conv3', layers.Conv2DLayer),
    #('pool3', layers.MaxPool2DLayer),
    #('dropout3', layers.DropoutLayer),
    #('conv4', layers.Conv2DLayer),
    ('hidden4', layers.DenseLayer),
    ('hidden5', layers.DenseLayer),
    ('hidden6', layers.DenseLayer),
    ('hidden7', layers.DenseLayer),
    ('hidden8', layers.DenseLayer),
    ('output', layers.DenseLayer),
  ],
  input_shape = (None, 1, 32, 32),
  #hidden1_num_units = 6,
  #transform_num_filters = 6,
  #conv1_num_filters=32, conv1_filter_size=(5, 5), 
  conv1_num_filters=32, conv1_filter_size=(3, 3), 
  pool1_pool_size=(2, 2),
  dropout1_p=0.2,
  #conv2_num_filters=64, conv2_filter_size=(5, 5), 
  conv2_num_filters=64, conv2_filter_size=(4, 4), 
  pool2_pool_size=(2, 2),
  #pool2_pool_size=(2, 2), pool2_ignore_border=True,
  dropout2_p=0.2,
  #conv3_num_filters = 128, conv3_filter_size = (5, 5),
  conv3_num_filters = 128, conv3_filter_size = (3, 3),
  #pool3_pool_size=(2, 2),
  #dropout3_p=0.2,
  #conv4_num_filters = 256, conv4_filter_size = (2, 2),
  hidden4_num_units = 2048, hidden4_nonlinearity = tanh,
  hidden5_num_units = 1024, hidden5_nonlinearity = tanh,
  hidden6_num_units = 512, hidden6_nonlinearity = tanh,
  hidden7_num_units = 256, hidden7_nonlinearity = tanh,
  hidden8_num_units = 128, hidden8_nonlinearity = tanh,
  output_num_units = 62, output_nonlinearity = softmax,
  
  update_learning_rate = 0.01,
  update_momentum = 0.9,
  
  batch_iterator_train = BatchIterator(batch_size = 100),
  batch_iterator_test = BatchIterator(batch_size = 100),
  
  use_label_encoder = True,
  regression = False,
  max_epochs = 1000,
  verbose = 1
)'''

net.fit(xTrain, y)

with open(pickleFile,'wb') as f:
  sys.setrecursionlimit(20000)
  pickle.dump(net, f)
