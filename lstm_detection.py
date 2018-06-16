import pandas as pd
import numpy as np
import os, glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

epochs = 2
batch_size = 32
train_file = '/home/pc/bgp-feature-extractor/csv/train.csv'
test_file = '/home/pc/bgp-feature-extractor/csv/test.csv'

x_train = pd.read_csv(train_file, index_col = 0)
y_train = x_train['class']
x_train.drop(['class','timestamp'], 1, inplace = True)

x_test = pd.read_csv(test_file, index_col = 0)
y_test = x_test['class']
x_test.drop(['class','timestamp','timestamp2'], 1, inplace = True)

x_train = x_train.values
y_train = y_train.values
x_test = x_test.values
y_test = y_test.values

x_train = x_train.reshape(-1, 44)
x_test = x_test.reshape(-1, 44)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(44,)))
model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling1D(pool_size = 2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
