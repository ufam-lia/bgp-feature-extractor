import pandas as pd
import numpy as np
import os, glob, sys

features_path = '/home/pc/bgp-feature-extractor/datasets/'
LABELS_DROP = ['news','nadas','flaps','origin_changes','as_path_avg','unique_as_path_max',\
               'unique_as_path_avg','rare_ases_max','rare_ases_avg','number_rare_ases','edit_distance_max',\
               'edit_distance_avg','ann_to_shorter','ann_to_longer','class','origin_2','imp_wd_dpath','imp_wd_spath']
rrc = 'rrc00'
def drop_columns(csv):
    for i in xrange(0, 200):
        col = 'edit_distance_dict_' + str(i) #+ '_'
        # print col
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)
        col = 'edit_distance_unique_dict_' + str(i) #+ '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)

    for label in LABELS_DROP:
        csv.drop(label, 1, inplace=True)
    return csv

def add_label(csv, start, end):
    labels = []
    for ts in csv['timestamp2']:
        if ts >= start and ts <= end:
            labels.append(1)
        else:
            labels.append(0)
    csv['class'] = pd.Series(labels)
    return csv

def train(train_files):
    df = pd.DataFrame()
    if len(train_files) > 0:
        for f in train_files:
            csv = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)
            csv = drop_columns(csv)
            df = df.append(csv, sort = True)

        df.reset_index(drop=True, inplace=True)
        df.to_csv(features_path + 'top10_train_' + rrc + '.csv', quoting=3)

def test(test_files):
    df = pd.DataFrame()
    if len(test_files) > 0:
        for f in test_files:
            csv = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)
            csv = add_label(csv, 1000818000, 1001030400)
            csv = drop_columns(csv)
            df = df.append(csv, sort = True)

        df.reset_index(drop = True, inplace = True)
        df.to_csv(features_path + 'top10_test_' + rrc + '.csv', quoting=3)

def main(argv):
    global rrc
    rrc = sys.argv[1]
    train_files = sorted(glob.glob(features_path + 'features-code-red-' + rrc + '*'))
    test_files  = sorted(glob.glob(features_path + 'features-nimda-' + rrc + '*'))
    train(train_files)
    test(test_files)

if __name__ == "__main__":
    main(sys.argv)
