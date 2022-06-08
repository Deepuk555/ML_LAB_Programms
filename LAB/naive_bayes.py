import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

le = preprocessing.LabelEncoder()
clf = MultinomialNB()
data = pd.read_csv('data2.csv')

features = [feat for feat in data]
targetLabel = features[-1]
features.remove(features[-1])
print(features)

diff_values = []
for f in features:
    for v in data[f]:
        if v not in diff_values:
            diff_values.append(v)

print(diff_values)

dataArray = np.array(data.iloc[:, :-1])
print(dataArray)

le.fit(diff_values)
list(le.classes_)

trans = []
for d in dataArray:
    trans.append(le.transform(d))
print(trans)

target = data[targetLabel]
print(target)

target = np.array(target)
tar = []
for t in target:
    if t == "yes":
        tar.append(1)
    else:
        tar.append(0)
print(tar)

clf.fit(trans, tar)

predicting = ["sunny", "cool", "high", "strong"]
pre_array = le.transform(predicting)
# print(pre_array)

pre_array = np.reshape(pre_array, (1, 4))
print(pre_array)

print(clf.predict(pre_array))
