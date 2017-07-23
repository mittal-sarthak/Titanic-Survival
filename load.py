# Load CSV (using python)
import csv
import numpy
from time import time
filename = 'train.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
#print(data.shape)
print(data.shape)
import codecs

filename2 = 'train_choices.csv'
raw_data2 = open(filename2, 'rt')
reader2 = csv.reader(raw_data2, delimiter=',', quoting=csv.QUOTE_NONE)
x2 = list(reader2)
data2 = numpy.array(x2).astype('float')
#print(data.shape)
print(data2.shape)

filename3 = 'test.csv'
raw_data3 = open(filename3, 'rt')
reader3 = csv.reader(raw_data3, delimiter=',', quoting=csv.QUOTE_NONE)
x3 = list(reader3)
data3 = numpy.array(x3).astype('float')
#print(data.shape)
print(data3.shape)


filename4 = 'gender_submission.csv'
raw_data4 = open(filename4, 'rt')
reader4 = csv.reader(raw_data4, delimiter=',', quoting=csv.QUOTE_NONE)
x4 = list(reader4)
data4 = numpy.array(x4).astype('float')
#print(data.shape)
print(data4.shape)


#gaussian classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
t0 = time()
gnb.fit(data,data2.ravel())
print "training time:", round(time()-t0, 3), "s"

t1 = time()
y_pred=gnb.predict(data3)
print "predicting time:", round(time()-t1, 3), "s"
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(data4, y_pred)
print accuracy
#print gnb.score(data3, data4)


