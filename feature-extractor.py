from __future__ import division
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
from collections import OrderedDict
import operator

os.environ['TZ'] = 'US'
tzset()

def main():
    """
    Computes timeseries features for a given anomaly event
    :param collector: Collector name
    :param peer: AS number of target peer
    :param anomaly: Anomaly name
    """
    parser = argparse.ArgumentParser(description='Process BGP timeseries')
    parser.add_argument('-c','--collector', help='Name of the collector', required=True)
    parser.add_argument('-p','--peer', help='Peer considered (0, if all peer must be considered)', required=True)
    parser.add_argument('-a','--anomaly', help='Anomaly event name', required=True)
    parser.add_argument('-t','--timesteps', help='Timestep window that must be considered', required=True)
    parser.add_argument('-r','--rib', dest='rib',help='Disable RIB initialization', action='store_true')
    parser.set_defaults(multi=False)
    args = vars(parser.parse_args())

    collector = args['collector']
    peer = args['peer']
    anomaly = args['anomaly']
    anomaly = args['anomaly']
    timesteps = int(args['timesteps'])
    rib = args['rib']
    c = 0

    metrics = Metrics(timesteps)
    anomaly = BGPAnomaly(anomaly, collector, '*')
    days = anomaly.get_files()
    if anomaly.get_rib() is not None:
        if rib:
            print "#Initializing RIB"
            metrics.init_rib(anomaly.get_rib(), peer)
            pass

    for update_files in days:
        update_files = sorted(update_files)
        upds_prev = dict()

        for f in update_files:
            metrics.add_updates(f, peer)
            print f + ': ' + str(metrics.count_updates)

            peer_upds = OrderedDict(sorted((metrics.peer_upds).items(), key = operator.itemgetter(1), reverse = True))

        file = f.split('.')
        # pickle.dump(metrics.prefix_lookup, open(file[0] + file[1] + file[2] + '-lookup.pkl', "wb"))

        day = update_files[0].split('.')[1]

    features = metrics.get_features()
    features_dict = features.to_dict()
    df = features.to_dataframe()
    output_filename = 'features-'+ anomaly.event +'-'+ collector +'-'+ peer +'-'+ metrics.minutes_window +'.csv'
    df = df.fillna(0)
    df.to_csv(output_filename, sep=',', encoding='utf-8')
    print output_filename + ': OK'
    # metrics.plot()

if __name__ == '__main__':
    main()
