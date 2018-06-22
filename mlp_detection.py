from __future__ import division
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

metrics = Metrics()

epochs = 50
batch_size = 32
epsilon = 0.000000000000001

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
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

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
model.add(Dense(256, activation='relu', input_shape = (x_train[0].shape)))
# model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0000001),
              metrics=['accuracy'])

validation_data = x_test
validation_target = y_test
class_weight = {0: 1.,  1: 4.}

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    # validation_data=(validation_data, validation_target),
                    class_weight=class_weight)
                    # callbacks=[metrics])

y_pred = model.predict(x_test, verbose = 2).round()
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

print type(y_test[0][0])
print type(y_pred[0][0])

acc = (tp + tn)/(tp + tn + fp + fn)
print 'acc->' + str(np.round(acc*100, decimals=2)) + '%'
precision = tp/(tp + fp + epsilon)
print 'precision->' + str(np.round(precision*100, decimals=2)) + '%'
print 'precision->' + str(precision_score(y_test, y_pred.round())*100) + '%'
recall = (tp)/(tp + fn)
print 'recall->' + str(np.round(recall*100, decimals=2)) + '%'
print 'recall->' + str(np.round(recall_score(y_test, y_pred.round())*100, decimals=2)) + '%'
f1 = 2*(precision*recall)/(precision + recall + epsilon)
print 'f1->' + str(np.round(f1*100, decimals=2)) + '%'
print 'f1->' + str(np.round(f1_score(y_test, y_pred.round())*100, decimals=2)) + '%'




# score = model.evaluate(x_test, y_test, verbose = 2)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
