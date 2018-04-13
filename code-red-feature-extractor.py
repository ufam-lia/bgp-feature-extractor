import sys
import os, time
import os.path
from os.path import isfile
from optparse import OptionParser
import datetime as dt
from datetime import datetime
from mrtparse import *
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import cPickle as pickle
import argparse

def dd():
    return defaultdict(int)

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

    files = []
    files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010713.0*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010714.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010715.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010716.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010717.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010718.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010719.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010720.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010721.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010722.*.gz")
    # files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010723.*.gz")
    files = sorted(files)

    days = args.days
    days_checked = 0
    count_ts = defaultdict(int)
    upds_per_file = []
    announc_per_file = []
    announc513_per_file = []
    announc559_per_file = []
    announc1755_per_file = []
    announc6893_per_file = []
    for f in files:
        count_updates = 0
        count_announcements = 0
        count_updates_as513 = 0
        count_updates_as559 = 0
        count_updates_as1755 = 0
        count_updates_as6893 = 0

        d = Reader(f)
        current_day = dt.datetime.fromtimestamp(d.next().mrt.ts).date()

        for m in d:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            # if current_day != dt.datetime.fromtimestamp(m.ts).date():
            #     days_checked += 1
            #     if days_checked <= days:
            #         break


            if (m.type == MRT_T['BGP4MP'] or m.type == MRT_T['BGP4MP_ET']) \
               and m.subtype in BGPMessageST \
               and m.bgp.msg is not None \
               and m.bgp.msg.type == BGP_MSG_T['UPDATE']:
                #Total updates per prefix
                count_updates += 1
                count_ts[m.ts] += 1

                if (m.bgp.msg.wd_len == 0) and (m.bgp.msg.attr_len > 0):
                    count_announcements += 1

        print f + ': ' + str(count_updates)

    fig = plt.figure(1)

    count_ts = OrderedDict(sorted(count_ts.items()))
    plt.subplot(1,1,1)
    plt.plot(range(len(count_ts.keys())), count_ts.values(), lw=0.95, color = 'blue')

    # plt.xticks(range(0, len(count_ts.keys()), 86400), [datetime.fromtimestamp(x) for x in count_ts.keys() if (x % )])
    # plt.plot(range(len(timeseries_ann_6893)), timeseries_ann_6893, lw=0.95, color = 'k')
    output = str(random.randint(1, 1000))
    print output + '.png'
    fig.savefig(output, bboxes_inches = '30', dpi = 400)
    # plt.show()

if __name__ == '__main__':
    main()
