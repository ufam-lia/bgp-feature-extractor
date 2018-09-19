import pandas as pd
import numpy as np
import os, glob, sys

features_path = '/home/pc/bgp-feature-extractor/datasets/'
LABELS_DROP = ['news','nadas','flaps','origin_changes','as_path_avg','unique_as_path_max',\
               'unique_as_path_avg','rare_ases_max','rare_ases_avg','number_rare_ases','edit_distance_max',\
               'edit_distance_avg','ann_to_shorter','ann_to_longer','origin_2','imp_wd_dpath','imp_wd_spath']
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

def add_column(csv, col):
    n_rows = csv.shape[0]
    csv[col] = pd.Series([0 for x in xrange(0, n_rows+1)])
    return csv

def fix_columns(csv):
    for i in xrange(0, 11):
        col = 'edit_distance_dict_' + str(i) #+ '_'
        if col not in csv.keys():
            csv = add_column(csv, col)

        col = 'edit_distance_unique_dict_' + str(i) #+ '_'
        if col not in csv.keys():
            csv = add_column(csv, col)

    for i in xrange(0, 3):
        col = 'origin_' + str(i) #+ '_'
        if col not in csv.keys():
            csv = add_column(csv, col)

def add_label(csv, start, end, label):
    labels = []
    for ts in csv['timestamp2']:
        if ts >= start and ts <= end:
            labels.append(label)
        else:
            labels.append(0)
    csv['class'] = pd.Series(labels)
    return csv

def adjust_to_batch_size(csv, batch_size):
    diff = (32 - csv.shape[0] % batch_size) if (csv.shape[0] % batch_size) != 0 else 0
    last_line = pd.DataFrame(csv.tail(1), columns=csv.columns)
    for i in range(0, diff):
        csv = csv.append(last_line, sort=True)
    return csv

def preprocessing(files, name='name', start=0, end=0, label=1):
    df = pd.DataFrame()
    df2 = pd.DataFrame()

    if len(files) > 0:
        for f in files:
            csv = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)
            csv2 = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)

            csv = add_label(csv, start, end, 1)
            csv2 = add_label(csv2, start, end, label)

            df = df.append(csv, sort = True)
            df2 = df2.append(csv2, sort = True)

            df = df.fillna(0)
            df2 = df2.fillna(0)

        df = adjust_to_batch_size(df, 32)
        df2 = adjust_to_batch_size(df2, 32)

        df.reset_index(drop=True, inplace=True)
        df2.reset_index(drop=True, inplace=True)

        df.to_csv(features_path + 'dataset_' + name + '_' + rrc + '.csv', quoting=3)
        df2.to_csv(features_path + 'dataset_multi_' + name + '_' + rrc + '.csv', quoting=3)

def main(argv):
    global rrc
    global peer
    rrc = sys.argv[1]
    peer = sys.argv[2]
    timescales = ['1', '5', '10', '15', '60', '120']
    for ts in timescales:
        nimda_files              = sorted(glob.glob(features_path + 'features-nimda-' + rrc + '-' + peer +  '-' + ts + '.csv'))
        slammer_files            = sorted(glob.glob(features_path + 'features-slammer-' + rrc + '-' + peer +  '-' + ts + '.csv'))
        code_red_files           = sorted(glob.glob(features_path + 'features-code-red-' + rrc + '-' + peer +  '-' + ts + '.csv'))
        moscow_files             = sorted(glob.glob(features_path + 'features-moscow-blackout-' + rrc + '-' + peer +  '-' + ts + '.csv'))
        malaysian_telecom_files  = sorted(glob.glob(features_path + 'features-malaysian-telecom-' + rrc + '-' + peer + '-' + ts + '.csv'))
        aws_leak_files           = sorted(glob.glob(features_path + 'features-aws-leak-' + rrc + '-' + peer + '-' + ts + '.csv'))
        as_path_error_files      = sorted(glob.glob(features_path + 'features-as-path-error-' + rrc + '-' + peer + '-' + ts + '.csv'))
        as_3561_filtering_files  = sorted(glob.glob(features_path + 'features-as-3561-filtering-' + rrc + '-' + peer + '-' + ts + '.csv'))
        as9121_files             = sorted(glob.glob(features_path + 'features-as9121-' + rrc + '-' + peer + '-' + ts + '.csv'))
        japan_files              = sorted(glob.glob(features_path + 'features-japan-earthquake-' + rrc + '-' + peer + '-' + ts + '.csv'))

        preprocessing(code_red_files, name='code-red_'+peer+'_'+ts, start=995553071, end=995591487, label=1)
        preprocessing(nimda_files, name='nimda_'+peer+'_'+ts, start=1000818222, end=1001030344, label=1)
        preprocessing(slammer_files, name='slammer_'+peer+'_'+ts, start=1043472590, end=1043540404, label=1)
        preprocessing(aws_leak_files, name='aws-leak_'+peer+'_'+ts, start=1461345001,end=1461349210, label=2)
        preprocessing(as_3561_filtering_files, name='as-3561-filtering_'+peer+'_'+ts, start=986578087,end=986579527, label=2)
        preprocessing(as9121_files, name='as9121_'+peer+'_'+ts, start=1103916000, end=1103918580, label=2)
        preprocessing(japan_files, name='japan-earthquake_'+peer+'_'+ts, start=1299834783, end=1299857943, label=3)

        if peer == '3257':
            preprocessing(as_path_error_files, name='as-path-error_'+peer+'_'+ts, start=1002484580,end=1002504620, label=2)
        elif peer == '9057':
            preprocessing(as_path_error_files, name='as-path-error_'+peer+'_'+ts, start=1002484700,end=1002499460, label=2)
        elif peer == '3333':
            preprocessing(as_path_error_files, name='as-path-error_'+peer+'_'+ts, start=1002484580,end=1002499400, label=2)

        if peer == '13237':
            preprocessing(moscow_files, name='moscow_blackout_'+peer, start=1116996909, end=1117017309, label=3)
        else:
            preprocessing(moscow_files, name='moscow_blackout_'+peer+'_'+ts, start=1116996009, end=1117006209, label=3)

        if peer == '513':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434104640, label=2)
        elif peer == '25091':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434109560, label=2)
        elif peer == '34781':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434104880, label=2)
        elif peer == '20932':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434107700, label=2)

if __name__ == "__main__":
    main(sys.argv)
