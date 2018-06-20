import pandas as pd
import numpy as np
import os, glob

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import os
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

metrics = Metrics()

epochs = 10000
batch_size = 320000
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

# print x_train
# x_train -= x_train.mean(axis = 0)
# x_train /= x_train.std(axis = 0)
# # x_test -= x_test.mean(axis = 0)
# x_test /= x_test.std(axis = 0)
# print '*'*100
# print x_train
# print np.round(x_train.mean(axis = 0))
# print np.round(x_train.std(axis = 0))
# scaler = MinMaxScaler(feature_range=(0, 1))
# x_train = scaler.fit_transform(x_train)
# print x_train
# x_test = scaler.fit_transform(x_test)
# print x_test

model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (x_train[0].shape)))
# model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0000001),
              metrics=['accuracy'])

validation_data = x_test
validation_target = y_test

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(validation_data, validation_target))
                    # callbacks=[metrics])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
