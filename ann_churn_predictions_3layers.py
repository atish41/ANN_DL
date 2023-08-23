# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:50:58 2023

@author: ATISHKUMAR
"""
#ANN
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\sept_DL\7th\ANN_ 1st\Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#encoding categorical data
#apply label encoder for gender
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

#now apply one hat encoder for geographical column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#apply feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#splitting dataset into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)

#part 2 building ann

#intilazie ann
ann=tf.keras.models.Sequential()

#add input layer and hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#adding seacond hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#adding 3rd hinnden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


#adding output layer

ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#part 3
# training ann

#compling ann
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#training ann on training set
ann.fit(X_train,y_train, batch_size=32, epochs=200)

#part 4

#predicitng the test results'
y_pred=ann.predict(X_test)
y_pred= (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#mkaing confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

