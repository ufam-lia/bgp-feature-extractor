from __future__ import division
import pandas as pd
import numpy as np
import glob
import keras
import os, sys, csv
import time
from operator import itemgetter

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

def print_header(file):
    # print '*'*123
    # print '*'*123
    # print '*'*123
    # print '*'*123
    print '*******Training with file: ' + file + '***************'
    # print '*'*123
    # print '*'*123
    # print '*'*123
    # print '*'*123

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
        # print str(y_pred[i]) + ' == ' + str(y_test[i])
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

    return confusion

class F1EarlyStop(Callback):
    def __init__(self, patience=40):
        super(Callback, self).__init__()
        self.patience = patience

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.f1_history = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        # print val_predict
        val_targ = self.validation_data[1]
        # print val_targ
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        epoch_metrics = dict()
        epoch_metrics['f1'] = _val_f1
        epoch_metrics['precision'] = _val_precision
        epoch_metrics['recall'] = _val_recall
        epoch_metrics['epoch'] = epoch

        self.f1_history.append(epoch_metrics)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # print '--> val_f1: %f \n--> val_precision: %f \n--> val_recall %f' %(_val_f1, _val_precision, _val_recall)

        getting_better = True
        for i in range(self.patience):
            if len(self.val_f1s) > self.patience:
                last_index = -(i + 1)
                if self.val_f1s[last_index] >= self.val_f1s[last_index - 1]:
                    getting_better = True
                    # print 'break ' + str(i)
                    break
                else:
                    # print 'self.val_f1s[last_index]  -> '+str(self.val_f1s[last_index] )
                    # print 'self.val_f1s[last_index - 1] -> '+str(self.val_f1s[last_index - 1])
                    # print i
                    getting_better = False

        if not getting_better:
            self.model.stop_training = True
        return

def csv_to_xy(val_file):
    x_val = pd.read_csv(val_file, index_col = 0, delimiter=',')
    x_val = x_val.fillna(0)
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

def find_best_model(dataset, f1_history):
    l = sorted(f1_history, key=itemgetter('f1'), reverse = True)
    best_model = l[0]
    best_model['f1'] = np.round(best_model['f1'],3)
    best_model['precision'] = np.round(best_model['precision'],3)
    best_model['recall'] = np.round(best_model['recall'],3)

    print 'Best model find @ epoch ' + str(best_model['epoch'])
    print 'F1 ->  ' + str(best_model['f1'])
    print 'Precision ->  ' + str(best_model['precision'])
    print 'Recall ->  ' + str(best_model['recall'])
    best_model['dataset'] = dataset
    return best_model

def dicts_to_csv(dicts):
    fieldnames = ['dataset', 'epoch', 'f1', 'precision', 'recall']
    keys = dicts[0].keys()
    with open('best_metrics.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(dicts)

def main():
    epochs = int(sys.argv[1])
    batch_size = 32
    epsilon = 0.000000000000001

    nimda_dataset = BGPDataset('nimda')
    code_red_dataset = BGPDataset('code-red')
    slammer_dataset = BGPDataset('slammer')
    moscow_dataset = BGPDataset('moscow_blackout')
    all_datasets = BGPDataset('*')

    train_files = []
    train_files += all_datasets.get_files(timebin = 5)
    # train_files += slammer_dataset.get_files(timebin = 5, peer ='559')
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
    # model.add(LSTM(100, return_sequences = True, stateful = True, activation='sigmoid'))
    # model.add(Dropout(0.2))
    # model.add(LSTM(100, return_sequences = True, stateful = True, activation='sigmoid'))
    # model.add(Dropout(0.2))
    # model.add(LSTM(100, return_sequences = True, stateful = True, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences = False, stateful = True, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])

    # tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard = TensorBoard(log_dir="logs/lstm")
    f1early = F1EarlyStop(patience = 10)
    best_models = []
    for epoch in range(epochs):
        i = 0
        print '###### ROUND %d' % (i+1)
        for sequence in train_vals:
            x_train = sequence[0]
            y_train = sequence[1]
            validation_data = x_train
            validation_target = y_train
            print_header(train_files[i])

            class_weight = {0: 1., 1: 4}
            hist = model.fit(x_train, y_train,
                                # batch_size=batch_size,
                                epochs=2,
                                verbose=0,
                                shuffle=False,
                                validation_data=(validation_data, validation_target),
                                class_weight=class_weight,
                                callbacks=[f1early, tensorboard])
            best_model = find_best_model(train_files[i], f1early.f1_history)
            best_models.append(best_model)
            model.reset_states()
            i += 1

    dicts_to_csv(best_models)
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
    # print('Training loss:', score[0])
    # print('Training accuracy:', score[1])


if __name__ == "__main__":
    main()
