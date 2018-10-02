import pandas as pd
import numpy as np
import os, glob, sys

features_path = '/home/pc/bgp-feature-extractor/datasets/'
# LABELS_DROP = ['news','nadas','flaps','origin_changes','as_path_avg','unique_as_path_max',\
#                'unique_as_path_avg','rare_ases_max','rare_ases_avg','number_rare_ases','edit_distance_max',\
#                'edit_distance_avg','ann_to_shorter','ann_to_longer','origin_2','imp_wd_dpath','imp_wd_spath']
LABELS_DROP = ['news','nadas','flaps','origin_changes','unique_as_path_max',\
               'rare_ases_max','number_rare_ases','edit_distance_max',\
               'ann_to_shorter','ann_to_longer','origin_2','imp_wd_dpath','imp_wd_spath']
rrc = 'rrc00'
peer = '1'

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

def adjust_to_batch_size(csv, batch_size):
    diff = (32 - csv.shape[0] % batch_size) if (csv.shape[0] % batch_size) != 0 else 0
    last_line = pd.DataFrame(csv.tail(1), columns=csv.columns)
    print diff
    for i in range(0, diff):
        csv = csv.append(last_line, sort=True)
    return csv

def preprocessing(files, name='name', start=0, end=0):
    df = pd.DataFrame()
    if len(files) > 0:
        for f in files:
            csv = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)
            csv = add_label(csv, start, end)
            csv = drop_columns(csv)
            df = df.append(csv, sort = True)

        df = adjust_to_batch_size(df, 32)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(features_path + 'dataset_' + name + '_' + rrc + '.csv', quoting=3)

def main(argv):
    global rrc
    global peer
    rrc = sys.argv[1]
    peer = sys.argv[2]

    code_red_files  = sorted(glob.glob(features_path + 'features-code-red-' + rrc + '-' + peer + '*'))
    nimda_files  = sorted(glob.glob(features_path + 'features-nimda-' + rrc + '-' + peer + '*'))
    slammer_files = sorted(glob.glob(features_path + 'features-slammer-' + rrc + '-' + peer + '-*'))
    moscow_files = sorted(glob.glob(features_path + 'features-moscow-blackout-' + rrc + '-' + peer + '-*'))

    preprocessing(code_red_files, name='code-red_'+peer, start=995553071, end=995591487)
    preprocessing(nimda_files, name='nimda_'+peer, start=1000818222, end=1001030344)
    preprocessing(slammer_files, name='slammer_'+peer, start=1043472590, end=1043540404)
    # AS13237
    preprocessing(moscow_files, name='moscow_blackout_'+peer, start=1116996909, end=1117017309)
    #Other ASes
    # preprocessing(moscow_files, name='moscow_blackout_'+peer, start=1116996009, end=1117006209)

if __name__ == "__main__":
    main(sys.argv)
