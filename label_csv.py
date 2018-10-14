import pandas as pd
import numpy as np
import os, glob, sys
import random

features_path = '/home/pc/bgp-feature-extractor/datasets/'
# LABELS_DROP = ['news','nadas','flaps','origin_changes','as_path_avg','unique_as_path_max',\
#                'unique_as_path_avg','rare_ases_max','rare_ases_avg','number_rare_ases','edit_distance_max',\
#                'edit_distance_avg','ann_to_shorter','ann_to_longer','origin_2','imp_wd_dpath','imp_wd_spath']
LABELS_DROP = ['news','nadas','flaps','origin_changes','unique_as_path_max',\
               'rare_ases_max','number_rare_ases','edit_distance_max',\
               'ann_to_shorter','ann_to_longer','origin_2','imp_wd_dpath','imp_wd_spath']

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
    return csv

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
    diff = (batch_size - csv.shape[0] % batch_size) if (csv.shape[0] % batch_size) != 0 else 0
    # print 'batch_size'
    # print batch_size
    # print csv.shape[0]
    # print csv.shape[0] % batch_size
    # print diff

    last_line = pd.DataFrame(csv.tail(1), columns=csv.columns)
    for i in range(0, diff):
        csv = csv.append(last_line, sort=True)
    return csv

def randomize_dataset(dataset, start, end):
    margin = 10
    before_length = dataset[dataset['timestamp2'] < start].shape[0]
    after_length = dataset[dataset['timestamp2'] > end].shape[0]
    before_clip = random.randint(0, before_length - margin)
    after_clip = random.randint(dataset.shape[0] - after_length, dataset.shape[0]) + margin
    return dataset[before_clip:after_clip]

def preprocessing(files, name='name', start=0, end=0, label=1):
    df = pd.DataFrame()
    df_multi = pd.DataFrame()
    anomaly_multi = pd.DataFrame()
    df_annotated = pd.DataFrame()

    if len(files) > 0:
        for f in files:
            print f
            csv           = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)
            csv_multi     = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)
            csv_annotated = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)

            csv = fix_columns(csv)
            csv_multi = fix_columns(csv_multi)
            csv_annotated = fix_columns(csv_annotated)

            csv = add_label(csv, start, end, 1)
            csv_multi = add_label(csv_multi, start, end, label)

            mark = csv['announcements'].max()
            csv_annotated = add_label(csv_annotated, start, end, mark/2)

            # csv = drop_columns(csv)
            # csv_multi = drop_columns(csv_multi)
            df = df.append(csv, sort = True)
            df_multi = df_multi.append(csv_multi, sort = True)
            df_annotated = df_annotated.append(csv_annotated, sort = True)


            df = df.fillna(0)
            df_multi = df_multi.fillna(0)

        df = adjust_to_batch_size(df, 32)
        df_multi = adjust_to_batch_size(df_multi, 32)
        df.reset_index(drop=True, inplace=True)
        df_multi.reset_index(drop=True, inplace=True)

        anomaly_multi = df_multi[df_multi['class'] != 0]
        anomaly_multi.reset_index(drop=True, inplace=True)
        df_annotated.insert(df_annotated.shape[1], 'ann2',df_annotated['announcements'])
        df_annotated.insert(df_annotated.shape[1], 'class2',df_annotated['class'])

        if not os.path.exists(features_path + '/annotated'):
            os.makedirs(features_path + '/annotated')

        anomaly_multi = adjust_to_batch_size(anomaly_multi, 2)
        print anomaly_multi.shape
        anomaly_multi.to_csv(features_path + 'anomaly_multi_' + name + '_' + rrc + '.csv', quoting=3)
        df.to_csv(features_path + 'dataset_' + name + '_' + rrc + '.csv', quoting=3)
        df_multi.to_csv(features_path + 'dataset_multi_' + name + '_' + rrc + '.csv', quoting=3)
        df_annotated.to_csv(features_path + 'annotated/dataset_multi_' + name + '_' + rrc + '.csv', quoting=3)

        for c in range(0,5):
            df = randomize_dataset(df, start, end)
            df_multi = randomize_dataset(df_multi, start, end)
            df.to_csv(features_path + 'dataset_' + name + '_' + rrc + '_rand' + str(c) + '.csv', quoting=3)
            df_multi.to_csv(features_path + 'dataset_multi_' + name + '_' + rrc + '_rand' + str(c) + '.csv', quoting=3)

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

        preprocessing(nimda_files, name='nimda_'+peer+'_'+ts, start=1000818222, end=1001030344, label=1)
        preprocessing(slammer_files, name='slammer_'+peer+'_'+ts, start=1043472590, end=1043540404, label=1)
        preprocessing(aws_leak_files, name='aws-leak_'+peer+'_'+ts, start=1461345001,end=1461349210, label=2)
        preprocessing(as_3561_filtering_files, name='as-3561-filtering_'+peer+'_'+ts, start=986578087,end=986579527, label=2)
        preprocessing(as9121_files, name='as9121_'+peer+'_'+ts, start=1103879947, end=1103884629, label=2)

        if peer == '513':
            preprocessing(code_red_files, name='code-red_'+peer+'_'+ts, start=995555750, end=995587250, label=1)
        if peer == '559':
            preprocessing(code_red_files, name='code-red_'+peer+'_'+ts, start=995560190, end=995588450, label=1)
        if peer == '6893':
            preprocessing(code_red_files, name='code-red_'+peer+'_'+ts, start=995555750, end=995587250, label=1)

        if peer == '2497':
            preprocessing(japan_files, name='japan-earthquake_'+peer+'_'+ts, start=1299834783, end=1299857943, label=3)
        if peer == '10026': # 8:56 - 15:15
            preprocessing(japan_files, name='japan-earthquake_'+peer+'_'+ts, start=1299833704, end=1299856504, label=3)

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
