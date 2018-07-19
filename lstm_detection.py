from __future__ import division
import pandas as pd
import numpy as np
import glob
import keras
import os, sys
# from time import time
import time
import tensorflow as tf
from bgpanomalies import *
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop, Adam
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import TensorBoard

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        # print val_predict
        val_targ = self.validation_data[1]
        # print val_targ
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print '--> val_f1: %f \n--> val_precision: %f \n--> val_recall %f' %(_val_f1, _val_precision, _val_recall)
        return


# Only compute a batch-wise average of recall.

def recall(y_true, y_pred):
    print    y_true
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def csv_to_xy(val_file):
    x_val = pd.read_csv(val_file, index_col = 0, delimiter=',')
    y_val = x_val['class']
    x_val.drop(['class', 'timestamp', 'timestamp2'], 1, inplace = True)
    x_val = x_val.values
    y_val = y_val.values
    print x_val.shape

    x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
    y_val = y_val.reshape(-1, 1)

    x_val = x_val.astype('float32')
    y_val = y_val.astype('float32')

    x_val -= x_val.mean(axis = 0)
    x_val /= x_val.std(axis = 0)

    return (x_val, y_val)

def confusion_matr(y_pred, y_test):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pos = 0
    neg = 0
    pred_pos = 0
    pred_neg = 0

    for i in xrange(len(y_pred)):
        print str(y_pred[i]) + ' == ' + str(y_test[i])
        if y_pred[i] == y_test[i]:
            if y_test[i] == 1:
                tp += 1
                pos += 1
                pred_pos += 1
            else:
                tn += 1
                neg += 1
                pred_neg += 1
        else:
            if y_test[i] == 1:
                fn += 1
                pos += 1
                pred_neg += 1
            else:
                fp += 1
                neg += 1
                pred_pos += 1
    confusion = dict()
    confusion['tp'] = tp
    confusion['tn'] = tn
    confusion['fp'] = fp
    confusion['fn'] = fn

    return confusion

    print '--------------'
    print 'pos->' + str(pos)
    print 'neg->' + str(neg)
    print 'pred_pos->' + str(pred_pos)
    print 'pred_neg->' + str(pred_neg)
    print '--------------'
    print 'tp->' + str(tp)
    print 'tn->' + str(tn)
    print 'fp->' + str(fp)
    print 'fn->' + str(fn)
    print '--------------'

metrics = Metrics()
epochs = int(sys.argv[1])
batch_size = 32
epsilon = 0.000000000000001

train_files = []
nimda_dataset = BGPDataset('nimda')
code_red_dataset = BGPDataset('code-red')
slammer_dataset = BGPDataset('slammer')
moscow_dataset = BGPDataset('moscow_blackout')

train_files += moscow_dataset.get_files(timebin = 5)
# train_files += code_red_dataset.get_files(timebin = 5)
test_file = slammer_dataset.get_files(5, peer='513')[0]

train_vals = []

for file in train_files:
    train_vals.append(csv_to_xy(file))

test_val = csv_to_xy(test_file)
x_test = test_val[0]
y_test = test_val[1]

validation_data = x_test
validation_target = y_test

model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape = (x_test[0].shape), batch_size = batch_size))
# # model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
model.add(LSTM(100, return_sequences = True, stateful = True, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = True, stateful = True, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = True, stateful = True, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = True, stateful = True, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences = False, stateful = True, activation='sigmoid'))
model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences = True, stateful = True, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences = True, stateful = True, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences = False, stateful = True, activation='sigmoid'))
# model.add(Dropout(0.2))
# model.add(LSTM(50, return_sequences = False, stateful = True))
# model.add(LSTM(100, return_sequences = False))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])

# tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
tensorboard = TensorBoard(log_dir="logs/lstm")
for epoch in range(epochs):
    i = 0
    for sequence in train_vals:
        x_train = sequence[0]
        y_train = sequence[1]
        validation_data = x_train
        validation_target = y_train
        print '*******Training with file: ' + train_files[i] + '***************'
        i += 1
        class_weight = {0: 1., 1: 4}
        hist = model.fit(x_train, y_train,
                            # batch_size=batch_size,
                            epochs=200,
                            verbose=1,
                            shuffle=False,
                            validation_data=(validation_data, validation_target),
                            class_weight=class_weight,
                            callbacks=[metrics, tensorboard])
        model.reset_states()

y_pred = model.predict(x_test, verbose = 2).round()

confusion = confusion_matr(y_pred, y_test)
tp = confusion['tp']
tn = confusion['tn']
fp = confusion['fp']
fn = confusion['fn']

acc = (tp + tn)/(tp + tn + fp + fn)
print 'acc->' + str(np.round(acc*100, decimals=2)) + '%'
precision = tp/(tp + fp + epsilon)
print 'precision->' + str(np.round(precision*100, decimals=2)) + '%'
recall = (tp)/(tp + fn)
print 'recall->' + str(np.round(recall*100, decimals=2)) + '%'
f1 = 2*(precision*recall)/(precision + recall + epsilon)
print 'f1->' + str(np.round(f1*100, decimals=2)) + '%'
# score = model.evaluate(x_test, y_test, verbose = 2)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
