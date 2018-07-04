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
from keras.layers import LSTM, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import warnings

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print ' - val_f1: %f - val_precision: %f - val_recall %f' %(_val_f1, _val_precision, _val_recall)

        return

metrics = Metrics()

epochs = 20
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
print x_train.shape
# x_train = x_train.reshape(-1, 44)
# x_test = x_test.reshape(-1, 44)
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
print x_train.shape
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# y_train = x_train.astype('float32')
# y_test = x_test.astype('float32')

model = Sequential()
# print x_train[0].shape
# model.add(Dense(10, activation='relu', input_shape=(44, )))
# model.add(Dense(10, activation='tanh'))
# model.add(Dense(10, activation='relu'))
# model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
# model.add(MaxPooling1D(pool_size = 2))
# model.add(LSTM(10, return_sequences = False))
model.add(LSTM(10, return_sequences = False, input_shape = (1, 44)  ))
# model.add(Dense(10, activation='relu'))
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr = 0.0001),
              # optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_test, y_test,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_train, y_train))
                    # callbacks=[metrics])
score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
