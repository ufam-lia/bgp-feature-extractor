import sys
import os, time
import os.path
from os.path import isfile
from optparse import OptionParser
import datetime as dt
from datetime import datetime
from mrtparse import *
from collections import defaultdict, OrderedDict
from mrtprint import *
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import cPickle as pickle
import argparse
import operator
from bgpmetrics_as import Metrics
import pandas as pd
from multiprocessing import Pool

def dd():
    return defaultdict(int)

def ddlist():
    return defaultdict(list)

def ddd():
    return defaultdict(dd)
()
def dddlist():
    return defaultdict(ddlist)

count_prefixes_ts = defaultdict(dd)
count_prefixes_upds = defaultdict(int)

timestamps = OrderedDict()
os.environ['TZ'] = 'US'
time.tzset()

BGPMessageST = [BGP4MP_ST['BGP4MP_MESSAGE'],BGP4MP_ST['BGP4MP_MESSAGE_AS4'],BGP4MP_ST['BGP4MP_MESSAGE_LOCAL'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_LOCAL'],BGP4MP_ST['BGP4MP_MESSAGE_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_LOCAL_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_LOCAL_ADDPATH']]
BGPStateChangeST = [ BGP4MP_ST['BGP4MP_STATE_CHANGE'], BGP4MP_ST['BGP4MP_STATE_CHANGE_AS4']]


def main():
    parser = argparse.ArgumentParser(description='Process BGP timeseries')
    parser.add_argument('--days', type=int)
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--rrc', type=str)

    rrc = sys.argv[1]
    peer = sys.argv[2]
    # rrc = args['rrc']
    #Traverse months
    # for i in [6,7,8,9,10,11]
    #Traverse files

    update_files = []
    rib_files = []

    #RIB files
    rib_files = rib_files + glob.glob("/home/pc/ripe-ris/code-red/" + rrc + "/bview*.gz")

    #Code Red v2
    days = []
    days.append(glob.glob("/home/pc/ripe-ris/moscow-blackout/" + rrc + "/updates.20050524.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/moscow-blackout/" + rrc + "/updates.20050525.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/moscow-blackout/" + rrc + "/updates.20050526.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/moscow-blackout/" + rrc + "/updates.20010919.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/moscow-blackout/" + rrc + "/updates.20010920.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/moscow-blackout/" + rrc + "/updates.20010921.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/moscow-blackout/" + rrc + "/updates.20010922.*.gz"))

    update_files = sorted(update_files)
    anomaly = days[0][0].split('/')[4]
    c = 0

    # metrics.init_rib(rib_files[0])
    for update_files in days:
        metrics = Metrics()
        update_files = sorted(update_files)
        # prefix_episodes = pool.map(metrics.add_updates, update_files)
        for f in update_files:
            metrics.add_updates(f, peer)
            file = f.split('.')
            # pickle.dump(metrics.prefix_lookup, open(file[0] + file[1] + file[2] + '-lookup.pkl', "wb"))
            print f + ': ' + str(metrics.count_updates)

            # pool.close()
            # pool.join()
        day = update_files[0].split('.')[1]
        features = metrics.get_features()
        features_dict = features.to_dict()
        df = features.to_dataframe()
        df.to_csv('features-'+ anomaly +'-' + rrc + '-'+peer+'-' + day + '.csv', sep=',', encoding='utf-8')
        print day + ': OK'
        # metrics.plot()
    #Nimda
    # days = []
    # days.append(glob.glob("/home/pc/ripe-ris/nimda/" + rrc + "/updates.20010916.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/nimda/" + rrc + "/updates.20010917.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/nimda/" + rrc + "/updates.20010918.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/nimda/" + rrc + "/updates.20010919.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/nimda/" + rrc + "/updates.20010920.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/nimda/" + rrc + "/updates.20010921.*.gz"))
    # days.append(glob.glob("/home/pc/ripe-ris/nimda/" + rrc + "/updates.20010922.*.gz"))
    #
    # update_files = sorted(update_files)
    # c = 0
    #
    # # metrics.init_rib(rib_files[0])
    # for update_files in days:
    #     metrics = Metrics()
    #     update_files = sorted(update_files)
    #     # prefix_episodes = pool.map(metrics.add_updates, update_files)
    #     for f in update_files:
    #         metrics.add_updates(f)
    #         file = f.split('.')
    #         pickle.dump(metrics.prefix_lookup, open(file[0] + file[1] + file[2] + '-lookup.pkl', "wb"))
    #         print f + ': ' + str(metrics.count_updates)
    #
    #         # pool.close()
    #         # pool.join()
    #     day = update_files[0].split('.')[1]
    #     features = metrics.get_features()
    #     features_dict = features.to_dict()
    #     df = features.to_dataframe()
    #     df.to_csv('features-nimda-" + rrc + "-' + day + '.csv', sep=',', encoding='utf-8')
    #     print day + ': OK'
    #

if __name__ == '__main__':
    main()
