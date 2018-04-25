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

# class Metrics(object):
#     """docstring for Metrics."""
#     def __init__(self, arg):
#         super(Metrics, self).__init__()
#         self.arg = arg

def main():
    parser = argparse.ArgumentParser(description='Process BGP timeseries')
    parser.add_argument('--days', type=int)
    parser.add_argument('--start', type=str)

    args = parser.parse_args()

    #Traverse months
    # for i in [6,7,8,9,10,11]
    #Traverse files

    files = []
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010713.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010714.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010715.*.gz")
    files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010716.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010716.010*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010716.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010717.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010718.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010719.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010720.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010721.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010722.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/rrc00/updates.20010723.*.gz")
    files = sorted(files)

    metrics = Metrics()

    for f in files:
        metrics.add(f)

    metrics.plot()

if __name__ == '__main__':
    main()
