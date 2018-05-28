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
from bgpmetrics import Metrics
import pandas as pd

def dd():
    return defaultdict(int)

def ddlist():
    return defaultdict(list)

def ddd():
    return defaultdict(dd)

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

    args = parser.parse_args()

    #Traverse months
    # for i in [6,7,8,9,10,11]
    #Traverse files

    update_files = []
    rib_files = []

    #RIB files
    rib_files = rib_files + glob.glob("/home/pc/ripe-ris/code-red/rrc03/bview*.gz")
    # rib_files = rib_files + glob.glob("/home/pc/ripe-ris/bview.20180501.0000.gz")
    # rib_files = rib_files + glob.glob("/home/pc/ripe-ris/bview.20180501.0000.gz")

    #Update files
    # update_files = update_files + glob.glob("/home/pc/Downloads/updates.20180515.20*.gz")
    # update_files = update_files + glob.glob("/home/pc/ripe-ris/updates.20180523.20**.gz")

    # update_files = update_files + glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010713.*.gz")
    # update_files = update_files + glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010716.010*.gz")

    days = []
    # days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010712.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010713.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010714.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010715.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010716.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010717.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010718.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010719.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc03/updates.20010720.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010721.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010722.*.gz"))
    days.append(glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010723.*.gz"))

    update_files = sorted(update_files)
    c = 0

    # metrics.init_rib(rib_files[0])
    for update_files in days:
        metrics = Metrics()
        update_files = sorted(update_files)
        for f in update_files:
            metrics.add_updates(f)
            print f + ': ' + str(metrics.count_updates)

        day = update_files[0].split('.')[1]
        features = metrics.get_features()
        features_dict = features.to_dict()
        df = features.to_dataframe()
        df.to_csv('features-' + day + '.csv', sep=';', encoding='utf-8')
        print day + ': OK'
        # metrics.plot()

if __name__ == '__main__':
    main()
