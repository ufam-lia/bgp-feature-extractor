import pandas as pd
import numpy as np
import os, glob, sys
import random

features_path = '/home/pc/bgp-feature-extractor/datasets/'
analysis_files = dict()
summary_files = dict()

def drop_columns(csv):
    for i in xrange(0, 200):
        col = 'edit_distance_dict_' + str(i) #+ '_'
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

def add_ratio_columns(csv):
    labels = []
    #announcements vs withdraw
    csv['ratio_ann'] = (csv['announcements']/(csv['withdrawals']+csv['announcements'])).replace(np.inf, 0)
    csv['ratio_wd']  = (csv['withdrawals']/(csv['withdrawals']+csv['announcements'])).replace(np.inf, 0)

    #origins
    count_origins = csv['origin_0'] + csv['origin_1'] + csv['origin_2']
    csv['ratio_origin0'] = (csv['origin_0']/count_origins).replace(np.inf, 0)
    csv['ratio_origin1'] = (csv['origin_1']/count_origins).replace(np.inf, 0)
    csv['ratio_origin2'] = (csv['origin_2']/count_origins).replace(np.inf, 0)

    #announcements types
    csv['ratio_dups']   = (csv['dups']/csv['announcements']).replace(np.inf, 0)
    csv['ratio_flaps']  = (csv['flaps']/csv['announcements']).replace(np.inf, 0)
    csv['ratio_imp_wd'] = (csv['imp_wd']/csv['announcements']).replace(np.inf, 0)
    csv['ratio_nadas']  = (csv['nadas']/csv['announcements']).replace(np.inf, 0)
    csv['ratio_news']   = (csv['news']/csv['announcements']).replace(np.inf, 0)

    #longer vs shorter
    csv['ratio_longer']  = (csv['ann_to_longer']/csv['announcements']).replace(np.inf, 0)
    csv['ratio_shorter']   = (csv['ann_to_shorter']/csv['announcements']).replace(np.inf,    0)
    csv['ratio_longer2']  = (csv['ann_to_longer']/(csv['ann_to_longer'] + csv['ann_to_shorter'])).replace(np.inf, 0)
    csv['ratio_shorter2']   = (csv['ann_to_shorter']/(csv['ann_to_longer'] + csv['ann_to_shorter'])).replace(np.inf, 0)

    #withdrawals
    count_withdrawals = csv['imp_wd'] + csv['withdrawals']
    csv['ratio_imp_wd2'] = (csv['imp_wd']/count_withdrawals).replace(np.inf, 0)
    csv['ratio_exp_wd'] = (csv['withdrawals']/count_withdrawals).replace(np.inf, 0)
    csv['ratio_wd_dups'] = (csv['wd_dups']/csv['withdrawals']).replace(np.inf, 0)
    csv['ratio_imp_wd_dpath'] = (csv['imp_wd_dpath']/csv['imp_wd']).replace(np.inf, 0)
    csv['ratio_imp_wd_spath'] = (csv['imp_wd_spath']/csv['imp_wd']).replace(np.inf, 0)

    return csv

def adjust_to_batch_size(csv, batch_size):
    diff = (batch_size - csv.shape[0] % batch_size) if (csv.shape[0] % batch_size) != 0 else 0

    last_line = pd.DataFrame(csv.tail(1), columns=csv.columns)
    for i in range(0, diff):
        csv = csv.append(last_line, sort=True)
    return csv

def analyze_dataset(csv, start, end):
    columns = ['announcements', 'withdrawals','ratio_ann', 'ratio_wd','ratio_longer','ratio_origin0',\
               'ratio_origin2','origin_changes','ratio_dups','ratio_flaps','ratio_imp_wd','ratio_nadas','ratio_news',\
               'ratio_imp_wd2','ratio_exp_wd','ratio_imp_wd_dpath','ratio_imp_wd_spath','edit_distance_avg'\
               ,'as_path_avg','rare_ases_avg','number_rare_ases','ratio_longer2','ratio_shorter2','ratio_shorter']
    analysis = dict()
    for col in columns:
        analysis.update(analyze_column(csv, col, start, end))
    return analysis

def summarize_dataset(dataset):
    columns = ['ratio_ann', 'ratio_wd','ratio_longer','ratio_origin0',\
               'ratio_origin2','origin_changes','ratio_dups','ratio_flaps','ratio_imp_wd','ratio_nadas','ratio_news',\
               'ratio_imp_wd2','ratio_exp_wd','ratio_imp_wd_dpath','ratio_imp_wd_spath','edit_distance_avg'\
               ,'as_path_avg','rare_ases_avg','number_rare_ases','ratio_longer2','ratio_shorter2','ratio_shorter']
    analysis = dict()
    for col in columns:
        analysis.update(summarize_column(dataset, col))
    return analysis

def analyze_column(csv, column, start, end):
    df = pd.DataFrame()

    before = csv[csv['timestamp2'] < start][column]
    during = csv[(csv['timestamp2'] >= start) & (csv['timestamp2'] <= end)][column]
    after = csv[csv['timestamp2'] > end][column]

    mean_before = before.mean()
    mean_during = during.mean()
    mean_after = after.mean()

    median_before = before.median()
    median_during = during.median()
    median_after = after.median()

    if mean_before > 0:
        mean_delta_before = 1 - (mean_during/mean_before)
    else:
        mean_delta_before = 1

    if mean_after > 0:
        mean_delta_after = 1 - (mean_during/mean_after)
    else:
        mean_delta_after = 1

    if median_before > 0:
        median_delta_before = 1 - (median_during/median_before)
    else:
        median_delta_before = 1

    if median_after > 0:
        median_delta_after = 1 - (median_during/median_after)
    else:
        median_delta_after = 1

    analysis = dict()
    analysis.update(evaluate_delta(mean_delta_before, mean_delta_after, column + '_mean'))
    analysis.update(evaluate_delta(median_delta_before, median_delta_after, column + '_median'))

    return analysis

def summarize_column(csv, column):
    df = pd.DataFrame()

    feature = csv[column]

    mean = feature.mean()
    median = feature.median()

    analysis = dict()
    analysis[column + '_mean'] = mean
    analysis[column + '_median'] = median

    return analysis

def evaluate_delta(delta_before, delta_after, column):
    analysis = ''
    analysis_dict = dict()
    signal = 'higher' if delta_before < 0  else 'lower'

    if abs(delta_before) < 0.1:
        analysis += 'pretty much the same'
    elif abs(delta_before) >= 0.1 and abs(delta_before)<0.25:
        analysis += 'slightly ' + signal
    elif abs(delta_before) >= 0.25 and abs(delta_before) < 0.5:
        analysis += signal
    elif abs(delta_before) >= 0.5:
        analysis += 'much ' + signal
    analysis_dict[column + '_before'] = analysis

    analysis = ''
    if abs(delta_after) < 0.1:
        analysis += 'pretty much the same'
    elif abs(delta_after) >= 0.1 and abs(delta_after)<0.25:
        analysis += 'slightly ' + signal
    elif abs(delta_after) >= 0.25 and abs(delta_after) < 0.5:
        analysis += signal
    elif abs(delta_after) >= 0.5:
        analysis += 'much ' + signal
    analysis_dict[column + '_after'] = analysis

    return analysis_dict

def randomize_dataset(dataset, start, end):
    margin = 10
    before_length = dataset[dataset['timestamp2'] < start].shape[0]
    after_length = dataset[dataset['timestamp2'] > end].shape[0]
    before_clip = random.randint(0, before_length - margin)
    after_clip = random.randint(dataset.shape[0] - after_length, dataset.shape[0]) + margin
    return dataset[before_clip:after_clip]

def preprocess_file(file, csv, name='name', base_path='', start=0, end=0, label=1, drop=False, ratio=False, batch_size=32, randomize=False):
    df = pd.DataFrame()
    anomaly = pd.DataFrame()

    csv = pd.read_csv(file, index_col=0, delimiter = ',', quoting=3)
    csv = fix_columns(csv)
    csv = add_label(csv, start, end, label)

    if ratio:
        csv = add_ratio_columns(csv)
        if label < 4:
            key_file = file.split('/')[-1]
            analyze_file(key_file, csv, start, end, label)

    if drop:
        csv = drop_columns(csv)

    df = df.append(csv, sort = True)
    df = df.fillna(0)
    df.reset_index(drop=True, inplace=True)
    df = adjust_to_batch_size(df, batch_size)
    df.to_csv(features_path + base_path + name + '_' + rrc + '.csv', quoting=3)

    if randomize:
        for c in range(0,5):
            df = randomize_dataset(df, start, end)
            df.to_csv(features_path + base_path + name + '_' + rrc + '_rand' + str(c) + '.csv', quoting=3)

def preprocessing(files, name='name', start=0, end=0, label=1):
    df = pd.DataFrame()
    df_multi = pd.DataFrame()
    anomaly_multi = pd.DataFrame()
    df_annotated = pd.DataFrame()

    if len(files) > 0:
        f = files[0]
        # print f
        csv = pd.read_csv(f, index_col=0, delimiter = ',', quoting=3)
        mark = csv['announcements'].max()

        #Save to files
        if not os.path.exists(features_path + '/annotated'):
            os.makedirs(features_path + '/annotated')

        if not os.path.exists(features_path + '/ratios'):
            os.makedirs(features_path + '/ratios')

        preprocess_file(f, csv, name=name, base_path='ratios/dataset_', start=start, end=end, label=1, drop=True, ratio=True, batch_size=32)
        preprocess_file(f, csv, name=name, base_path='ratios/dataset_multi_', start=start, end=end, label=label, drop=True, ratio=True, batch_size=32)
        preprocess_file(f, csv, name=name, base_path='annotated/dataset_multi_', start=start, end=end, label=mark, drop=True, ratio=True, batch_size=32)

def analyze_file(key_file, csv_annotated, start, end, label):
    global analysis_files
    global summary_files

    analysis = pd.read_csv('analysis.csv')
    if key_file not in analysis.keys().tolist():
        if label == 1:
            anomaly_class = 'indirect_'
        elif label == 2:
            anomaly_class = 'directz_'
        elif label == 3:
            anomaly_class = 'outage_'

        key_file = key_file.split('features-')[1].split('.csv')[0]
        key_file = anomaly_class + key_file
        analysis_files[key_file] = analyze_dataset(csv_annotated, start, end)
        summary_files[key_file] = summarize_dataset(csv_annotated)

def main(argv):
    global rrc
    global peer
    global analysis_files
    global summary_files

    rrc = sys.argv[1]
    peer = sys.argv[2]
    # timescales = ['1']
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
        # preprocessing(as_3561_filtering_files, name='as-3561-filtering_'+peer+'_'+ts, start=986578087,end=986579527, label=2)

        if peer == '15547':
            preprocessing(aws_leak_files, name='aws-leak_'+peer+'_'+ts, start=1461345001,end=1461349210, label=2)
        if peer == '34781':
            preprocessing(aws_leak_files, name='aws-leak_'+peer+'_'+ts, start=1461345001,end=1461349210, label=2)

        if peer == '13237':
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
        if peer == '1853':
            preprocessing(moscow_files, name='moscow_blackout_'+peer+'_'+ts, start=1116996009, end=1117006209, label=3)
        if peer == '12793':
            preprocessing(moscow_files, name='moscow_blackout_'+peer+'_'+ts, start=1116996009, end=1117006209, label=3)

        if peer == '513':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434104640, label=2)
        if peer == '25091':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434109560, label=2)
        if peer == '34781':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434104880, label=2)
        if peer == '20932':
            preprocessing(malaysian_telecom_files, name='malaysian-telecom_'+peer+'_'+ts, start=1434098520, end=1434107700, label=2)

        df_analysis_episode = pd.DataFrame(analysis_files)
        df_summary_episode = pd.DataFrame(summary_files)

        if os.path.isfile('analysis.csv'):
            df_analysis = pd.read_csv('analysis.csv', index_col=0)
            df_analysis = pd.concat([df_analysis, df_analysis_episode], axis=1, sort=True)
        else:
            df_analysis = df_analysis_episode

        if os.path.isfile('anomalies_comparison.csv'):
            df_summary = pd.read_csv('anomalies_comparison.csv', index_col=0)
            df_summary = pd.concat([df_summary, df_summary_episode], axis=1, sort=True)
        else:
            df_summary = df_summary_episode

        df_analysis = df_analysis[sorted(df_analysis.columns.tolist())]
        df_analysis.to_csv('analysis.csv')

        df_summary = df_summary[sorted(df_summary.columns.tolist())]
        df_summary.to_csv('anomalies_comparison.csv')

if __name__ == "__main__":
    main(sys.argv)
