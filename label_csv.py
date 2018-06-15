import pandas as pd
import numpy as np
import os, glob

features_path = '/home/pc/bgp-feature-extractor/csv/'
def drop_columns(csv):
    for i in xrange(11, 200):
        col = 'edit_distance_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)
        col = 'edit_distance_unique_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)
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

files_train = sorted(glob.glob(features_path + 'features-2001*'))
files = sorted(glob.glob(features_path + 'features-nimda-2001*'))

df = pd.DataFrame()
#Train
for f in files_train:
    csv = pd.read_csv(f, index_col=0, delimiter = ',', quoting = 3)
    csv = drop_columns(csv)
    df = df.append(csv, sort = True)
    csv.to_csv(f, quoting = 3)

df.reset_index(drop = True, inplace = True)
df.to_csv(features_path + 'code-red-rrc00.csv', quoting = 3)
df.to_csv(features_path + 'train.csv', quoting = 3)

#Test
df = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f, index_col=0, delimiter = ',', quoting = 3)
    csv = add_label(csv, 1000818000, 1001030400)
    csv = drop_columns(csv)
    csv.to_csv(f, quoting = 3)
    df = df.append(csv, sort = True)

df.reset_index(drop = True, inplace = True)
df.to_csv(features_path + 'nimda-rrc00.csv', quoting = 3)
df.to_csv(features_path + 'test.csv', quoting = 3)
