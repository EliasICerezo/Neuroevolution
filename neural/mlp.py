from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score

def generate_mlp(dim, num_outputs):
  model = Sequential()
  model.add(Dense(32, input_shape=(dim,), activation='relu'))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(Dense(16, activation='relu'))
  model.add(keras.layers.normalization.BatchNormalization())
  model.add(Dense(num_outputs, activation='sigmoid'))
  return model

def load_data():
  (x_train,y_train), (x_test,y_test) = mnist.load_data()
  return x_train, y_train, x_test, y_test 


if __name__ == '__main__':
  x_train, y_train, x_test, y_test = load_data()
  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)
  model = generate_mlp(28*28,10)
  model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
  model.summary()
  model.fit(x=x_train, y=y_train, verbose=1, epochs=15)
  h = model.evaluate(x=x_test,y=y_test)
  
