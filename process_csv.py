import pandas as pd
import numpy as np
import os, glob

features_path = '/home/pc/bgp-feature-extractor/csv/'

files_train = sorted(glob.glob(features_path + 'features-2001*'))
files_test = sorted(glob.glob(features_path + 'features-nimda-2001*'))

df_train = pd.DataFrame()
for f in files_train:
    csv = pd.read_csv(f, index_col=0, delimiter = ';')
    for i in xrange(11, 30):
        col = 'edit_distance_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)
        col = 'edit_distance_unique_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)

    csv.to_csv(f)
    df_train = df_train.append(csv, sort = True)

df_test = pd.DataFrame()
for f in files_test:
    csv = pd.read_csv(f, index_col=0, delimiter = ';')
    for i in xrange(11, 30):
        col = 'edit_distance_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)
        col = 'edit_distance_unique_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)

    csv.to_csv(f)
    df_test = df_test.append(csv, sort = True)

df_train.reset_index(drop = True, inplace = True)
df_test.reset_index(drop = True, inplace = True)

y_train = df_train['class']
# y_test = df_test['class']

df_train.drop(['timestamp', 'class'], 1, inplace = True)
# df_test.drop(['timestamp', 'class'], 1, inplace = True)

print df_train
print '-'*220
print df_test
