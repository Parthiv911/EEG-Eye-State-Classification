import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization.batch_normalization import BatchNormalization

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import datasets, svm, metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd

import os
#Uncomment when using Windows if cuDNN does not work
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

data=pd.read_csv(r"eye_data.csv")

#Shuffle the Data Set
data=data.sample(frac=1)

#Splitting the Data Set into Train and Test Set
train_x=data.iloc[:10000,0:14]
train_y=data.iloc[:10000,14:15]

test_x=data.iloc[10000:13000,0:14]
test_y=data.iloc[10000:13000,14:15]

#Normalizing the data set
train_x=preprocessing.normalize(train_x)
test_x=preprocessing.normalize(test_x)

train_y=train_y.to_numpy().flatten()
test_y=test_y.to_numpy().flatten()

#Split the test data into a validation and a test data set
crossvalid_y=test_y[0:1000]
crossvalid_x=test_x[0:1000]

test_x=test_x[1000:]
test_y=test_y[1000:]

#Build the Support Vector Machine Classifier
#kernel='rbf',gamma=1000000,C=100
#classifier=svm.SVC(gamma=1000000,C=100)

clf=svm.SVC()

parameters = {'gamma': [1000,10000,100000,1000000], 'C': [1, 10,100,1000]}
classifier = GridSearchCV(clf, parameters,verbose=3,cv=2)

#Train the model
classifier.fit(train_x,train_y)

#Test on Validation Data Set
expected_crossvalid = crossvalid_y
predicted_crossvalid = classifier.predict(crossvalid_x)

#Confusion Matrix and Accuracy Score for Validation DataSet
print("Confusion matrix and Accuracy Score for Validation:")
print("Confusion matrix:")
confusion=metrics.confusion_matrix(expected_crossvalid,predicted_crossvalid)

print(confusion)
print("Accuracy Score: ",accuracy_score(expected_crossvalid,predicted_crossvalid))

#Test the model
expected_test = test_y
predicted_test = classifier.predict(test_x)

#Confusion Matrix and Accuracy Score for Test DataSet
print("\nConfusion matrix and Accuracy Score for Testing:")
print("Confusion matrix:")
confusion=metrics.confusion_matrix(expected_test,predicted_test)

print(confusion)
print("Accuracy Score: ",accuracy_score(expected_test,predicted_test))

#Covolutional Neural Networks with different architectures were tried but it was found that SVM produced far better results
'''
model=Sequential()

model.add(Conv1D(filters=20,kernel_size=2,input_shape=(14,1),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())

model.add(Conv1D(filters=30,kernel_size=2,input_shape=(14,1),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=40,kernel_size=2,input_shape=(14,1),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling1D(pool_size=2))


model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
#model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss="mean_squared_error",metrics=['binary_accuracy'])
model.fit([train_x],[train_y],epochs=10,verbose=1)

predicted=model.predict(test_x)

for i in range(0,predicted.shape[0]):
    if (predicted[i] >= 0.5):
        predicted[i]= 1
    else:
        predicted[i] = 0


print(accuracy_score(test_y,predicted))
'''
