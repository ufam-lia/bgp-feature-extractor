from __future__ import division
import pandas as pd
import numpy as np
import glob
import os, sys, csv
import time
from operator import itemgetter
from collections import defaultdict
import random
import argparse

from bgpanomalies import *
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop, Adam
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

# import keras
# import tensorflow as tf
# from keras import backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)))

def print_header(file):
    print( '*******Training with file: ' + file + '***************')

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
        # print( val_predict)
        val_targ = self.validation_data[1]
        # print( val_targ)
        _val_f1 = f1_score(val_targ, val_predict, average=None)
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)
        epoch_metrics = dict()
        epoch_metrics['f1'] = _val_f1
        epoch_metrics['precision'] = _val_precision
        epoch_metrics['recall'] = _val_recall
        epoch_metrics['epoch'] = epoch

        self.f1_history.append(epoch_metrics)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        # print( '--> val_f1: %f \n--> val_precision: %f \n--> val_recall %f' %(_val_f1, _val_precision, _val_recall))

        getting_better = True
        for i in range(self.patience):
            if len(self.val_f1s) > self.patience:
                last_index = -(i + 1)
                if self.val_f1s[last_index] >= self.val_f1s[last_index - 1]:
                    getting_better = True
                    # print( 'break ' + str(i))
                    break
                else:
                    # print( 'self.val_f1s[last_index]  -> '+str(self.val_f1s[last_index] ))
                    # print( 'self.val_f1s[last_index - 1] -> '+str(self.val_f1s[last_index - 1]))
                    # print( i)
                    getting_better = False

        if not getting_better:
            self.model.stop_training = True
        return

def find_best_model(dataset, f1_history):
    l = sorted(f1_history, key=itemgetter('f1'), reverse = True)
    best_model = l[0]
    best_model['f1'] = np.round(best_model['f1'],3)
    best_model['precision'] = np.round(best_model['precision'],3)
    best_model['recall'] = np.round(best_model['recall'],3)

    print( 'Best model found @ epoch ' + str(best_model['epoch']))
    print( 'F1 ->  ' + str(best_model['f1']))
    print( 'Precision ->  ' + str(best_model['precision']))
    print( 'Recall ->  ' + str(best_model['recall']))
    best_model['dataset'] = dataset
    return best_model

def dicts_to_csv(dicts, fieldnames, filename):
    with open(filename + '.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(dicts)

def lists_to_csv(dicts, fieldnames, filename):
    with open(filename + '.csv', 'wb') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(dicts.keys())
        writer.writerows(zip(*dicts.values()))

def get_train_datasets(exclude_dataset, multi = False, anomaly = False, base_path='/home/pc/bgp-feature-extractor/datasets/'):
    nimda_dataset             = BGPDataset('nimda', anomaly, base_path=base_path)
    code_red_dataset          = BGPDataset('code-red', anomaly, base_path=base_path)
    slammer_dataset           = BGPDataset('slammer', anomaly, base_path=base_path)
    moscow_dataset            = BGPDataset('moscow_blackout', anomaly, base_path=base_path)
    as9121_dataset            = BGPDataset('as9121', anomaly, base_path=base_path)
    aws_leak_dataset          = BGPDataset('aws-leak', anomaly, base_path=base_path)
    as_3561_filtering_dataset = BGPDataset('as-3561-filtering', anomaly, base_path=base_path)
    as_path_error_dataset     = BGPDataset('as-path-error', anomaly, base_path=base_path)
    malaysian_dataset         = BGPDataset('malaysian-telecom', anomaly, base_path=base_path)
    japan_dataset             = BGPDataset('japan-earthquake', anomaly, base_path=base_path)

    train_files = []

    if 'code-red' not in exclude_dataset:
        train_files += code_red_dataset.get_files(timebin = [1, 5, 15], peer ='513', multi = multi)
        train_files += code_red_dataset.get_files(timebin = [1, 5, 15], peer ='6893', multi = multi)

    if 'nimda' not in exclude_dataset:
        train_files += nimda_dataset.get_files(timebin = [1, 5], peer ='513', multi = multi)
        train_files += nimda_dataset.get_files(timebin = [1, 5], peer ='559', multi = multi)
        train_files += nimda_dataset.get_files(timebin = [1, 5, 15], peer ='6893', multi = multi)

    if 'slammer' not in exclude_dataset:
        train_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='513', multi = multi)
        train_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='559', multi = multi)
        train_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='6893', multi = multi)

    if 'moscow-blackout' not in exclude_dataset:
        train_files += moscow_dataset.get_files(timebin = [1, 5], peer ='1853', multi = multi)
        train_files += moscow_dataset.get_files(timebin = [1, 5], peer ='12793', multi = multi)

    if 'aws-leak' not in exclude_dataset:
        train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='15547', multi = multi)
        # train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='25091', multi = multi)
        train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='34781', multi = multi)

    if 'as9121' not in exclude_dataset:
        # train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='1853', multi = multi)
        # train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='12793', multi = multi)
        train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='13237', multi = multi)

    if 'as-3561-filtering' not in exclude_dataset:
        # train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='1286', multi = multi)
        # train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3257', multi = multi)
        # train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3333', multi = multi)
        pass

    if 'as-path-error' not in exclude_dataset:
        # train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3257', multi = multi)
        # train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3333', multi = multi)
        # train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '9057', multi = multi)
        pass

    if 'malaysian-telecom' not in exclude_dataset:
        # train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '513', multi = multi)
        # train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '20932', multi = multi)
        train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '25091', multi = multi)
        # train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '34781', multi = multi)

    if 'japan-earthquake' not in exclude_dataset:
        train_files += japan_dataset.get_files(timebin = [1,5], peer = '2497', multi = multi)
        train_files += japan_dataset.get_files(timebin = [1,5], peer = '10026', multi = multi)
    return train_files

'''
as9121_13237_1
as9121_13237_5
aws-leak_15547_1
aws-leak_34781_1
aws-leak_34781_5
japan-earthquake_10026_1
japan-earthquake_2497_1
malaysian-telecom_25091_1
moscow_blackout_12793_1
moscow_blackout_1853_1
nimda_513_1
nimda_559_5
nimda_6893_1
slammer_513_1
slammer_513_5
slammer_559_1
slammer_6893_1
slammer_6893_5
'''
def get_optimal_train_datasets(exclude_dataset, multi = False, anomaly = False, base_path='/home/pc/bgp-feature-extractor/datasets/'):
    nimda_dataset             = BGPDataset('nimda', anomaly, base_path=base_path)
    code_red_dataset          = BGPDataset('code-red', anomaly, base_path=base_path)
    slammer_dataset           = BGPDataset('slammer', anomaly, base_path=base_path)
    moscow_dataset            = BGPDataset('moscow_blackout', anomaly, base_path=base_path)
    as9121_dataset            = BGPDataset('as9121', anomaly, base_path=base_path)
    aws_leak_dataset          = BGPDataset('aws-leak', anomaly, base_path=base_path)
    as_3561_filtering_dataset = BGPDataset('as-3561-filtering', anomaly, base_path=base_path)
    as_path_error_dataset     = BGPDataset('as-path-error', anomaly, base_path=base_path)
    malaysian_dataset         = BGPDataset('malaysian-telecom', anomaly, base_path=base_path)
    japan_dataset             = BGPDataset('japan-earthquake', anomaly, base_path=base_path)

    train_files = []

    if 'code-red' not in exclude_dataset:
        # train_files += code_red_dataset.get_files(timebin = [1, 5, 15], peer ='513', multi = multi)
        train_files += code_red_dataset.get_files(timebin = [5], peer ='6893', multi = multi)

    if 'nimda' not in exclude_dataset:
        train_files += nimda_dataset.get_files(timebin = [1], peer ='513', multi = multi)
        train_files += nimda_dataset.get_files(timebin = [5], peer ='559', multi = multi)
        train_files += nimda_dataset.get_files(timebin = [1], peer ='6893', multi = multi)

    if 'slammer' not in exclude_dataset:
        train_files += slammer_dataset.get_files(timebin = [1, 5], peer ='513', multi = multi)
        train_files += slammer_dataset.get_files(timebin = [1], peer ='559', multi = multi)
        train_files += slammer_dataset.get_files(timebin = [1, 5], peer ='6893', multi = multi)

    if 'moscow-blackout' not in exclude_dataset:
        train_files += moscow_dataset.get_files(timebin = [1], peer ='1853', multi = multi)
        train_files += moscow_dataset.get_files(timebin = [1], peer ='12793', multi = multi)

    if 'aws-leak' not in exclude_dataset:
        train_files += aws_leak_dataset.get_files(timebin = [1], peer ='15547', multi = multi)
        # train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='25091', multi = multi)
        train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='34781', multi = multi)

    if 'as9121' not in exclude_dataset:
        # train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='1853', multi = multi)
        # train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='12793', multi = multi)
        train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='13237', multi = multi)

    if 'as-3561-filtering' not in exclude_dataset:
        # train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='1286', multi = multi)
        # train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3257', multi = multi)
        # train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3333', multi = multi)
        pass

    if 'as-path-error' not in exclude_dataset:
        # train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3257', multi = multi)
        # train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3333', multi = multi)
        # train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '9057', multi = multi)
        pass

    if 'malaysian-telecom' not in exclude_dataset:
        # train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '513', multi = multi)
        # train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '20932', multi = multi)
        train_files += malaysian_dataset.get_files(timebin = [1], peer = '25091', multi = multi)
        # train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '34781', multi = multi)

    if 'japan-earthquake' not in exclude_dataset:
        train_files += japan_dataset.get_files(timebin = [1], peer = '2497', multi = multi)
        train_files += japan_dataset.get_files(timebin = [1], peer = '10026', multi = multi)
    return train_files

def get_test_datasets(test_datasets, multi = False, anomaly = False, base_path='/home/pc/bgp-feature-extractor/datasets/'):
    nimda_dataset             = BGPDataset('nimda', anomaly, base_path=base_path)
    code_red_dataset          = BGPDataset('code-red', anomaly, base_path=base_path)
    slammer_dataset           = BGPDataset('slammer', anomaly, base_path=base_path)
    moscow_dataset            = BGPDataset('moscow_blackout', anomaly, base_path=base_path)
    as9121_dataset            = BGPDataset('as9121', anomaly, base_path=base_path)
    aws_leak_dataset          = BGPDataset('aws-leak', anomaly, base_path=base_path)
    as_3561_filtering_dataset = BGPDataset('as-3561-filtering', anomaly, base_path=base_path)
    as_path_error_dataset     = BGPDataset('as-path-error', anomaly, base_path=base_path)
    malaysian_dataset         = BGPDataset('malaysian-telecom', anomaly, base_path=base_path)
    japan_dataset             = BGPDataset('japan-earthquake', anomaly, base_path=base_path)

    test_files = []

    if 'code-red' in test_datasets:
        test_files += code_red_dataset.get_files(timebin = [1, 5, 15], peer ='513', multi = multi)
        test_files += code_red_dataset.get_files(timebin = [1, 5, 15], peer ='6893', multi = multi)

    if 'nimda' in test_datasets:
        test_files += nimda_dataset.get_files(timebin = [1, 5], peer ='513', multi = multi)
        test_files += nimda_dataset.get_files(timebin = [1, 5], peer ='559', multi = multi)
        test_files += nimda_dataset.get_files(timebin = [1, 5, 15], peer ='6893', multi = multi)

    if 'slammer' in test_datasets:
        test_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='513', multi = multi)
        test_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='559', multi = multi)
        test_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='6893', multi = multi)

    if 'moscow-blackout' in test_datasets:
        test_files += moscow_dataset.get_files(timebin = [1, 5], peer ='1853', multi = multi)
        test_files += moscow_dataset.get_files(timebin = [1, 5], peer ='12793', multi = multi)

    if 'aws-leak' in test_datasets:
        test_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='15547', multi = multi)
        # test_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='25091', multi = multi)
        test_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='34781', multi = multi)

    if 'as9121' in test_datasets:
        # test_files += as9121_dataset.get_files(timebin = [1, 5], peer ='1853', multi = multi)
        # test_files += as9121_dataset.get_files(timebin = [1, 5], peer ='12793', multi = multi)
        test_files += as9121_dataset.get_files(timebin = [1, 5], peer ='13237', multi = multi)

    if 'as-3561-filtering' in test_datasets:
        test_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='1286', multi = multi)
        test_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3257', multi = multi)
        test_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3333', multi = multi)

    if 'as-path-error' in test_datasets:
        test_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3257', multi = multi)
        test_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3333', multi = multi)
        test_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '9057', multi = multi)

    if 'malaysian-telecom' in test_datasets:
        test_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '513', multi = multi)
        test_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '20932', multi = multi)
        test_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '25091', multi = multi)
        test_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '34781', multi = multi)

    if 'japan-earthquake' in test_datasets:
        test_files += japan_dataset.get_files(timebin = [1,5], peer = '2497', multi = multi)
        test_files += japan_dataset.get_files(timebin = [1,5], peer = '10026', multi = multi)

    return test_files

def add_lag(data, lag=1):
    df = pd.DataFrame(data)
    # df.to_csv('lag1.csv')
    columns = [df.shift(i) for i in range(0, lag+1)]
    df = pd.concat(columns, axis=1)
    df = df.fillna(0)
    # df.to_csv('lag2.csv')
    df = df.values
    return df

def eval_single_file(model, test_file, lag):
    num_classes = 4
    x_test, y_test = csv_to_xy(test_file, num_classes, lag)

    y_pred = model.predict(x_test, verbose=2).round()
    dataset = test_file.split('dataset_')[1].split('.csv')[0]
    filename = 'y_pred_' + dataset + '_' + str(random.randint(0, 1000)) + '.csv'
    np.savetxt(filename, y_pred, delimiter=',')
    print filename

def csv_to_xy(val_file, num_classes, lag):
    x_val = pd.read_csv(val_file, index_col = 0, delimiter=',')
    x_val = x_val.fillna(0)
    y_val = x_val['class']
    x_val.drop(['class', 'timestamp', 'timestamp2'], 1, inplace = True)
    x_val = x_val.values
    y_val = y_val.values

    x_val = x_val.astype('float32')
    y_val = y_val.astype('float32')
    x_val[np.isnan(x_val)] = 0

    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    x_val = scaler.fit_transform(x_val)
    np.savetxt('oi.csv',x_val, delimiter=',')

    if lag > 0:
        x_val = add_lag(x_val, lag=lag)
        x_val = x_val.reshape(x_val.shape[0], lag+1, x_val.shape[1]//(lag+1))

    if num_classes > 2:
        y_val = to_categorical(y_val, num_classes=num_classes)
        y_val = y_val.reshape(-1, num_classes)
    else:
        y_val = y_val.reshape(-1, 1)

    return x_val, y_val

def calc_metrics(y_test, y_pred, multi=False):
    if multi:
        f1         = f1_score(y_test, y_pred, average=None)
        recall     = recall_score(y_test, y_pred, average=None)
        precision  = precision_score(y_test, y_pred, average=None)
        accuracy  = accuracy_score(y_test, y_pred)
    else:
        f1         = f1_score(y_test, y_pred, average='binary')
        recall     = recall_score(y_test, y_pred, average='binary')
        precision  = precision_score(y_test, y_pred, average='binary')
        accuracy  = accuracy_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def print_metrics(accuracy, precision, recall, f1, test_file):
    print '*'*100
    print test_file
    print 'accuracy->' + str(accuracy)
    print 'precision->' + str(precision)
    print 'recall->' + str(recall)
    print 'f1->' + str(f1)
    print

def save_metrics(accuracy, precision, recall, f1, test_file, df, epoch, lag):
    print type(precision)
    if (type(precision) == np.float64) or (type(precision) == float):
        num_classes=1
        df.loc[test_file,'accuracy'] = accuracy
        df.loc[test_file,'epoch'] = epoch
        df.loc[test_file,'lag'] = lag
        for i in range(0, num_classes):
            df.loc[test_file,'precision_' + str(i)] = precision
            df.loc[test_file,'recall_' + str(i)] = recall
            df.loc[test_file,'f1_' + str(i)] = f1
    else:
        num_classes=4
        df.loc[test_file,'accuracy'] = accuracy
        df.loc[test_file,'epoch'] = epoch
        df.loc[test_file,'lag'] = lag
        for i in range(0, num_classes):
            df.loc[test_file,'precision_' + str(i)] = precision[i]
            df.loc[test_file,'recall_' + str(i)] = recall[i]
            df.loc[test_file,'f1_' + str(i)] = f1[i]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--epochs', help='Number of epochs for all sequences', required=True)
    parser.add_argument('-i','--inner', help='Number of epochs per each sequence', required=False)
    parser.add_argument('-l','--lag', help='Number of timesteps (1 = current step + last step)', required=True)
    parser.add_argument('-t','--test', help='Test datasets (might be a comma-separated list)', required=True)
    parser.add_argument('-d','--ignore', help='List of datasets that must be ignored', required=False)
    parser.add_argument('-m','--multi', dest='multi',help='Enable multi-way datasets', action='store_true')
    parser.add_argument('-o','--one', dest='multi',help='Disable multi-way datasets', action='store_false')
    parser.set_defaults(multi=False)
    args = vars(parser.parse_args())


    epochs = int(args['epochs'])
    lag = int(args['lag'])
    test_events = args['test'].split(',')
    batch_size = 32
    epsilon = 1e-10
    df = pd.DataFrame()
    multi = args['multi']

    if args['inner'] is not None:
        inner_epochs = int(args['inner'])
    else:
        inner_epochs = 1

    if args['ignore'] is not None:
        ignored_events = args['ignore'].split(',')
        ignored_events += test_events
    else:
        ignored_events = test_events

    train_files = get_optimal_train_datasets(ignored_events, multi = multi, base_path='/home/pc/bgp-feature-extractor/datasets/ratios/')
    test_files = get_test_datasets(test_events, multi = multi, base_path='/home/pc/bgp-feature-extractor/datasets/ratios/')
    # test_file = BGPDataset('aws-leak').get_files(5, peer='15547')[0]
    # test_file = BGPDataset('japan-earthquake').get_files(5, peer='2497')[0]

    print 'TRAIN'
    for f in train_files:
        print f

    print 'TEST'
    for f in test_files:
        print f

    if multi:
        num_classes = 4
    else:
        num_classes = 2

    train_vals = []
    for file in train_files:
        x_val, y_val = csv_to_xy(file, num_classes, lag)
        train_vals.append(((x_val, y_val), file))

    test_vals = []
    for file in test_files:
        x_val, y_val = csv_to_xy(file, num_classes, lag)
        test_vals.append(((x_val, y_val), file))

    if len(test_vals) > 0:
        validation_data = test_vals[0][0][0]
        validation_target = test_vals[0][0][1]
    else:
        print 'No test set found with name(s): ' + str(test_events)

    model = Sequential()
    # model.add(Dense(10, activation='sigmoid', input_shape = (x_test[0].shape), batch_size = batch_size))
    model.add(LSTM(10, return_sequences=True, batch_input_shape=(batch_size,validation_data.shape[1], validation_data.shape[2]), stateful=True, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(LSTM(10, return_sequences=False, stateful = False, activation='sigmoid'))
    model.add(Dropout(01))
    # model.add(LSTM(100, return_sequences=False, stateful = True, activation='sigmoid'))
    # model.add(Dropout(0.2))

    if multi:
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001),metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001),metrics=['accuracy'])

    # model.summary()
    # ann_viz(model, view=True, filename="network.gv", title="nn-arch")
    # plot_model(model, show_shapes=True, to_file='network.png')
    # print('Fig saved')

    f1early = F1EarlyStop(patience = 10)
    tensorboard = TensorBoard(log_dir="logs/lstm")

    round = 0
    for epoch in range(epochs):
        print( '\n\n###### ROUND %d \n' % (round+1))
        round += 1
        i = 0

        random.shuffle(train_vals)
        for train_samples in train_vals:
            filename = train_samples[1]
            x_train, y_train = (train_samples[0][0], train_samples[0][1])
            validation_data = x_train
            validation_target = y_train

            validation_data = test_vals[0][0][0]
            validation_target = test_vals[0][0][1]
            if not multi:
                y_val = to_categorical(y_train, num_classes=2)
                y_val = y_val.reshape(-1, 2)
                y_train_labels = y_val
            else:
                y_train_labels = y_train

            y_integers = np.argmax(y_train_labels, axis=1)
            y_classes = pd.DataFrame(y_train_labels).idxmax(1, skipna=False)
            label_encoder = LabelEncoder()
            label_encoder.fit(list(y_classes))
            y_integers = label_encoder.transform(list(y_classes))

            # Create dict of labels : integer representation
            labels_and_integers = dict(zip(y_classes, y_integers))
            class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
            sample_weights = compute_sample_weight('balanced', y_integers)
            class_weights_dict = dict(zip(label_encoder.transform(list(label_encoder.classes_)), class_weights))

            print filename
            df1 = pd.DataFrame(sample_weights)
            df2 = pd.DataFrame(y_train)
            df1.to_csv('df1.csv',sep=',')
            df2.to_csv('df2.csv',sep=',')
            # class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
            # d_class_weights = dict(enumerate(class_weights))
            # print filename + 'class_weights -> ' + str(d_class_weights)
            # class_weight = {0: 1., 1: 4, 2:4, 3:4}

            hist = model.fit(x_train, y_train,
                            # batch_size=batch_size,
                            sample_weight=sample_weights,
                            epochs=inner_epochs,
                            verbose=1,
                            validation_data=(validation_data, validation_target),
                            callbacks=[f1early],
                            # callbacks=[f1early, tensorboard],
                            # class_weight=class_weights,
                            shuffle=False)
            #Evaluate after each sequence processed
            # y_pred = model.predict(x_test, verbose = 0).round()
            model.reset_states()
            i += 1

        for test_samples in test_vals:
            test_file = test_samples[1]
            x_test, y_test = test_samples[0]
            y_pred = model.predict(x_test, verbose = 0).round()

            accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred, multi=multi)
            # print_metrics(precision, recall, f1, test_file)
        print epoch
        if ((epoch % 1) == 0 and epoch >= 1) or (epoch == epochs-1):
            print( '####VALIDATION')
            for test_samples in test_vals:
                test_file = test_samples[1].split('/')[-1]
                x_test, y_test = test_samples[0]
                y_pred = model.predict(x_test, verbose = 0).round()
                accuracy, precision, recall, f1 = calc_metrics(y_test, y_pred, multi=multi)
                test_file_annotated = test_file.split('.csv')[0]+ '_' + str(epoch) + 'epochs'
                df = save_metrics(accuracy, precision, recall, f1, test_file, df, epoch, lag)

                model_name = 'test_' + test_file + '_' + str(epoch+1) + 'x' + str(inner_epochs)+'x'+str(lag)
                print model_name

                y_csv = pd.DataFrame()
                if multi:
                    for i in range(0, num_classes):
                        y_test_list = map(lambda x: x[i], y_test)
                        y_csv['y_test_'+str(i)] = pd.Series(list(y_test_list))

                    for i in range(0, num_classes):
                        y_pred_list = map(lambda x: x[i], y_pred)
                        y_csv['y_pred_'+str(i)] = pd.Series(list(y_pred_list))
                else:
                    y_pred_list = map(lambda x: x[0], y_pred)
                    y_test_list = map(lambda x: x[0], y_test)
                    y_csv['y_pred'] = pd.Series(list(y_pred_list))
                    y_csv['y_test'] = pd.Series(list(y_test_list))
                print 'results_saved'
                y_csv.to_csv('results/predictions/y_pred_' + model_name + '.csv', quoting=3)

            model_name = 'test_' + args['test'].replace(',','-') + '_' + str(epoch+1) + 'x' + str(inner_epochs)+'x'+str(lag)
            df.to_csv('results/results_'+model_name+'.csv', sep=',')
            model.save('models/'+model_name + '.h5')
            print 'Results saved: results_'+model_name+'.csv'

if __name__ == "__main__":
    main()
