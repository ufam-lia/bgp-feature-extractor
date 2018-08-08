import sys
import os, time
from time import time, tzset
import argparse
import glob
import cPickle as pickle
import argparse
from bgpmetrics_as import Metrics
import pandas as pd
from multiprocessing import Pool
from bgpanomalies import BGPAnomaly

os.environ['TZ'] = 'US'
tzset()

def main():
    parser = argparse.ArgumentParser(description='Process BGP timeseries')
    parser.add_argument('--days', type=int)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--rrc', type=str)

    rrc = sys.argv[1]
    peer = sys.argv[2]
    anomaly = sys.argv[3]
    c = 0

    metrics = Metrics()
    anomaly = BGPAnomaly(anomaly, rrc)
    days = anomaly.get_files()
    if anomaly.get_rib() is not None:
        metrics.init_rib(anomaly.get_rib())

    for update_files in days:
        update_files = sorted(update_files)

        for f in update_files:
            metrics.add_updates(f, peer)
            print f + ': ' + str(metrics.count_updates)

        file = f.split('.')
        pickle.dump(metrics.prefix_lookup, open(file[0] + file[1] + file[2] + '-lookup.pkl', "wb"))

        day = update_files[0].split('.')[1]
    features = metrics.get_features()
    features_dict = features.to_dict()
    df = features.to_dataframe()
    # output_filename = 'features-'+ anomaly.event +'-'+ rrc +'-'+ peer +'-'+ day +'-'+ metrics.minutes_window +'.csv'
    output_filename = 'features-'+ anomaly.event +'-'+ rrc +'-'+ peer +'-'+ metrics.minutes_window +'.csv'
    df.to_csv(output_filename, sep=',', encoding='utf-8')
    print output_filename + ': OK'
    # metrics.plot()

if __name__ == '__main__':
    main()
