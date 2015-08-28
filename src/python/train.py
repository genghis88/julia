import pandas as pd
from PIL import Image
import pickle
from os.path import join
import sys

labelsFile = sys.argv[1]
normalizedDataDirectory = sys.argv[2]
pickleFile = sys.argv[3]

trainingLabels = pd.read_csv(labelsFile)

trainingData = []
for index, row in trainingLabels.iterrows():
  fileName = join(normalizedDataDirectory, str(row['ID']) + '.png')
  im = Image.open(fileName)
  data = list(im.getdata())
  trainingData.append(data)

classMapping = {}
for i in range(26):
  classMapping[chr(ord('a') + i)] = i

for i in range(26):
  classMapping[chr(ord('A') + i)] = i + 26

for i in range(10):
  classMapping[chr(ord('0') + i)] = i + 52

y = list(trainingLabels['Class'])
y = map(lambda char: str(char), y)
y = map(lambda char: classMapping[char], y)

#from sklearn import svm
#clf = svm.SVC(kernel='poly', C=10**-3, degree=2)

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=5)

#from sklearn.ensemble import BaggingClassifier
#from sklearn.neighbors import KNeighborsClassifier
#clf = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), max_features=0.5)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20, max_features=20, min_samples_split=1)

from sklearn import cross_validation
scores = cross_validation.cross_val_score(clf, trainingData, y, cv=5)

print scores
#clf.fit(trainingData, y)

with open(pickleFile,'wb') as f:
  pickle.dump(clf, f)
