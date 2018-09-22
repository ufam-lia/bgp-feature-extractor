from __future__ import division
import pandas as pd
import numpy as np
import glob
import os, sys, csv
import time
from operator import itemgetter
from collections import defaultdict
import random
import tensorflow as tf
from bgpanomalies import *
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop, Adam
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

# K.set_session(K.tf.Session(config=K.tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)))

def print_header(file):
    print( '*******Training with file: ' + file + '***************')

def calc_metrics(y_pred, y_test, print_metrics = True):
    epsilon = 0.000000000000001
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pos = 0
    neg = 0
    pred_pos = 0
    pred_neg = 0

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

def csv_to_xy(val_file):
    x_val = pd.read_csv(val_file, index_col = 0, delimiter=',')
    x_val = x_val.fillna(0)
    y_val = x_val['class']
    x_val.drop(['class', 'timestamp', 'timestamp2'], 1, inplace = True)
    x_val = x_val.values
    y_val = y_val.values

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

def get_optimal_datasets(exclude_dataset):
    nimda_dataset             = BGPDataset('nimda')
    code_red_dataset          = BGPDataset('code-red')
    slammer_dataset           = BGPDataset('slammer')
    moscow_dataset            = BGPDataset('moscow_blackout')
    as9121_dataset            = BGPDataset('as9121')
    aws_leak_dataset          = BGPDataset('aws-leak')
    as_3561_filtering_dataset = BGPDataset('as-3561-filtering')
    as_path_error_dataset     = BGPDataset('as-path-error')
    malaysian_dataset         = BGPDataset('malaysian-telecom')
    japan_dataset             = BGPDataset('japan-earthquake')

    train_files = []

    if 'code-red' not in exclude_dataset:
        train_files += code_red_dataset.get_files(timebin = [1, 5, 15], peer ='513')
        train_files += code_red_dataset.get_files(timebin = [1, 5, 15], peer ='6893')

    if 'nimda' not in exclude_dataset:
        train_files += nimda_dataset.get_files(timebin = [1, 5], peer ='513')
        train_files += nimda_dataset.get_files(timebin = [1, 5], peer ='559')
        train_files += nimda_dataset.get_files(timebin = [1, 5, 15], peer ='6893')

    if 'slammer' not in exclude_dataset:
        train_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='513')
        train_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='559')
        train_files += slammer_dataset.get_files(timebin = [1, 5, 15], peer ='6893')

    if 'moscow' not in exclude_dataset:
        train_files += moscow_dataset.get_files(timebin = [1, 5], peer ='1853')
        train_files += moscow_dataset.get_files(timebin = [1, 5], peer ='12793')

    if 'aws-leak' not in exclude_dataset:
        train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='15547')
        train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='25091')
        train_files += aws_leak_dataset.get_files(timebin = [1, 5], peer ='34781')

    if 'as9121' not in exclude_dataset:
        train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='1853')
        train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='12793')
        train_files += as9121_dataset.get_files(timebin = [1, 5], peer ='13237')

    if 'as-3561-filtering' not in exclude_dataset:
        train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='1286')
        train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3257')
        train_files += as_3561_filtering_dataset.get_files(timebin = [1, 5], peer ='3333')

    if 'as-path-error' not in exclude_dataset:
        train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3257')
        train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '3333')
        train_files += as_path_error_dataset.get_files(timebin = [1, 5], peer = '9057')

    if 'malaysian-telecom' not in exclude_dataset:
        train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '513')
        train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '20932')
        train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '25091')
        train_files += malaysian_dataset.get_files(timebin = [1, 5], peer = '34781')

    if 'japan-earthquake' not in exclude_dataset:
        train_files += japan_dataset.get_files(timebin = [1,5], peer = '2497')

    return train_files

def get_optimal_datasets_multi(exclude_dataset):
    nimda_dataset             = BGPDataset('nimda')
    code_red_dataset          = BGPDataset('code-red')
    slammer_dataset           = BGPDataset('slammer')
    moscow_dataset            = BGPDataset('moscow_blackout')
    as9121_dataset            = BGPDataset('as9121')
    aws_leak_dataset          = BGPDataset('aws-leak')
    as_3561_filtering_dataset = BGPDataset('as-3561-filtering')
    as_path_error_dataset     = BGPDataset('as-path-error')
    malaysian_dataset         = BGPDataset('malaysian-telecom')
    japan_dataset             = BGPDataset('japan-earthquake')

    train_files = []

    if 'code-red' not in exclude_dataset:
        train_files += code_red_dataset.get_files_multi(timebin = [1, 5, 15], peer ='513')
        train_files += code_red_dataset.get_files_multi(timebin = [1, 5, 15], peer ='6893')

    if 'nimda' not in exclude_dataset:
        train_files += nimda_dataset.get_files_multi(timebin = [1, 5], peer ='513')
        train_files += nimda_dataset.get_files_multi(timebin = [1, 5], peer ='559')
        train_files += nimda_dataset.get_files_multi(timebin = [1, 5, 15], peer ='6893')

    if 'slammer' not in exclude_dataset:
        train_files += slammer_dataset.get_files_multi(timebin = [1, 5, 15], peer ='513')
        train_files += slammer_dataset.get_files_multi(timebin = [1, 5, 15], peer ='559')
        train_files += slammer_dataset.get_files_multi(timebin = [1, 5, 15], peer ='6893')

    if 'moscow' not in exclude_dataset:
        train_files += moscow_dataset.get_files_multi(timebin = [1, 5], peer ='1853')
        train_files += moscow_dataset.get_files_multi(timebin = [1, 5], peer ='12793')

    if 'aws-leak' not in exclude_dataset:
        train_files += aws_leak_dataset.get_files_multi(timebin = [1, 5], peer ='15547')
        train_files += aws_leak_dataset.get_files_multi(timebin = [1, 5], peer ='25091')
        train_files += aws_leak_dataset.get_files_multi(timebin = [1, 5], peer ='34781')

    if 'as9121' not in exclude_dataset:
        train_files += as9121_dataset.get_files_multi(timebin = [1, 5], peer ='1853')
        train_files += as9121_dataset.get_files_multi(timebin = [1, 5], peer ='12793')
        train_files += as9121_dataset.get_files_multi(timebin = [1, 5], peer ='13237')

    if 'as-3561-filtering' not in exclude_dataset:
        train_files += as_3561_filtering_dataset.get_files_multi(timebin = [1, 5], peer ='1286')
        train_files += as_3561_filtering_dataset.get_files_multi(timebin = [1, 5], peer ='3257')
        train_files += as_3561_filtering_dataset.get_files_multi(timebin = [1, 5], peer ='3333')

    if 'as-path-error' not in exclude_dataset:
        train_files += as_path_error_dataset.get_files_multi(timebin = [1, 5], peer = '3257')
        train_files += as_path_error_dataset.get_files_multi(timebin = [1, 5], peer = '3333')
        train_files += as_path_error_dataset.get_files_multi(timebin = [1, 5], peer = '9057')

    if 'malaysian-telecom' not in exclude_dataset:
        train_files += malaysian_dataset.get_files_multi(timebin = [1, 5], peer = '513')
        train_files += malaysian_dataset.get_files_multi(timebin = [1, 5], peer = '20932')
        train_files += malaysian_dataset.get_files_multi(timebin = [1, 5], peer = '25091')
        train_files += malaysian_dataset.get_files_multi(timebin = [1, 5], peer = '34781')

    if 'japan-earthquake' not in exclude_dataset:
        train_files += japan_dataset.get_files_multi(timebin = [1,5], peer = '2497')

    return train_files

def add_lag(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(0, lag+1)]
    df = pd.concat(columns, axis=1)
    df = df.fillna(0)
    df = df.values
    return df

def eval_single_file(model, test_file, lag):
    test_val = csv_to_xy(test_file)
    y_test = test_val[1]
    x_test = test_val[0]
    x_test = add_lag(x_test, lag=lag)
    x_test = x_test.reshape(x_test.shape[0], lag+1, x_test.shape[1]//(lag+1))
    y_test = y_test.reshape(-1, 1)
    y_pred = model.predict(x_test, verbose=2).round()
    dataset = test_file.split('dataset_')[1].split('.csv')[0]
    filename = 'y_pred_' + dataset + '_' + str(random.randint(0, 1000)) + '.csv'
    np.savetxt(filename, y_pred, delimiter=',')
    print filename

def main():
    epochs = int(sys.argv[1])
    inner_epochs = int(sys.argv[2])
    lag = int(sys.argv[3])
    batch_size = 32
    epsilon = 0.000000000000001

    train_files = get_optimal_datasets_multi(['slammer','aws-leak','japan-earthquake'])
    test_file = BGPDataset('malaysian-telecom').get_files(5, peer='20932')[0]

    for f in train_files:
        print f
    train_vals = []
    for file in train_files:
        train_vals.append((csv_to_xy(file), file))

    test_val = csv_to_xy(test_file)
    y_test = to_categorical(test_val[1], num_classes=4)
    x_test = test_val[0]
    # print y_test

    x_test = add_lag(x_test, lag=lag)
    x_test = x_test.reshape(x_test.shape[0], lag+1, x_test.shape[1]//(lag+1))
    y_test = y_test.reshape(-1, 4)

    validation_data = x_test
    validation_target = y_test

    model = Sequential()
    # model.add(Dense(10, activation='sigmoid', input_shape = (x_test[0].shape), batch_size = batch_size))
    model.add(LSTM(100, return_sequences=True, batch_input_shape=(batch_size,x_test.shape[1], x_test.shape[2]), stateful=True, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=True, stateful = True, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False, stateful = True, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))

    # model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])

    # tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    # ann_viz(model, view=True, filename="network.gv", title="nn-arch")
    # plot_model(model, show_shapes=True, to_file='network.png')
    # print( 'Fig saved')

    tensorboard = TensorBoard(log_dir="logs/lstm")
    f1early = F1EarlyStop(patience = 10)
    best_models = []
    model_history = defaultdict(list)
    model_history_all = defaultdict(list)

    round = 0
    for epoch in range(epochs):
        print( '\n\n###### ROUND %d \n' % (round+1))
        round += 1
        i = 0

        random.shuffle(train_vals)
        for sequence_tuple in train_vals:
            filename = sequence_tuple[1]
            sequence = sequence_tuple[0]
            x_train = sequence[0]
            y_train = to_categorical(sequence[1], num_classes=4)
            # print y_train
            # for i in y_train:
            #     print i
            x_train = add_lag(x_train, lag=lag)
            x_train = x_train.reshape(x_train.shape[0], lag+1, x_train.shape[1]//(lag+1))
            y_train = y_train.reshape(-1, 4)
            validation_data = x_train
            validation_target = y_train
            print_header(filename)

            # class_weight = {0: 1., 1: 4}
            class_weight = {0: 1., 1: 4, 2:4, 3:4}
            hist = model.fit(x_train, y_train,
                                # batch_size=batch_size,
                                epochs=int(inner_epochs),
                                verbose=1,
                                validation_data=(validation_data, validation_target),
                                callbacks=[f1early, tensorboard],
                                class_weight=class_weight,
                                shuffle=False)
            #Evaluate after each sequence processed
            y_pred = model.predict(x_test, verbose = 2).round()
            print( '####TRAINING')
            model.reset_states()
            i += 1

    test_name = test_file.split('/')[5]
    y_pred = model.predict(x_test, verbose = 2).round()
    print( '####VALIDATION')
    f1         = f1_score(y_test, y_pred, average=None)
    recall     = recall_score(y_test, y_pred, average=None)
    precision  = precision_score(y_test, y_pred, average=None)

    print f1
    print recall
    print precision
    # print( 'acc->' + str(np.round(confusion['acc']*100, decimals=2)) + '%')
    # print( 'precision->' + str(np.round(confusion['precision']*100, decimals=2)) + '%')
    # print( 'recall->' + str(np.round(confusion['recall']*100, decimals=2)) + '%')
    # print( 'f1->' + str(np.round(confusion['f1']*100, decimals=2)) + '%')

    model_name = 'test_' + test_name + '_' + str(epochs) + 'x' + str(inner_epochs)+'x'+str(lag)

    y_csv = pd.DataFrame()
    y_pred_list = map(lambda x: x[0], y_pred)
    y_test_list = map(lambda x: x[0], y_test)
    y_csv['y_pred'] = pd.Series(list(y_pred_list))
    y_csv['y_test'] = pd.Series(list(y_test_list))
    y_csv.to_csv('y_pred_' + model_name + '.csv', quoting=3)
    model.save(model_name + '.h5')

if __name__ == "__main__":
    main()
