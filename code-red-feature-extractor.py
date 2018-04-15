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
    files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010713.*.gz")
    files = files + glob.glob("/home/pc/ripe-ris/code-red/updates.20010714.*.gz")
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
    bin_size = 60
    count_ts = defaultdict(int)
    count_origin = defaultdict(dd)
    count_origin2 = defaultdict(dd)
    count_ts_upds_ases = defaultdict(dd)
    count_ts_upds_prefixes = defaultdict(dd)

    upds_per_file = []
    announc_per_file = []
    for f in files:
        count_updates = 0
        count_announcements = 0
        d = Reader(f)
        current_day = dt.datetime.fromtimestamp(d.next().mrt.ts).date()
        first_ts = d.next().mrt.ts

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
                bin = (m.ts - first_ts)/bin_size
                count_updates += 1
                count_ts[bin] += 1
                count_ts_upds_ases[bin][m.bgp.peer_as] += 1

                #Total updates per prefix
                if m.bgp.msg.nlri is not None:
                    for nlri in m.bgp.msg.nlri:
                        count_ts_upds_prefixes[bin][nlri.prefix + '/' + str(nlri.plen)] += 1

                if m.bgp.msg.attr is not None:
                    for a in m.bgp.msg.attr:
                        if BGP_ATTR_T[a.type] == 'ORIGIN':
                            count_origin[bin][a.origin] += 1
                            count_origin2[bin][ORIGIN_T[a.origin]] += 1

                if (m.bgp.msg.wd_len == 0) and (m.bgp.msg.attr_len > 0):
                    count_announcements += 1

        print f + ': ' + str(count_updates)

    fig = plt.figure(1)
    count_ts = OrderedDict(sorted(count_ts.items()))

    #Filling blanks in the timeseries
    for i in range(count_ts.keys()[-1]):
        count_ts[i]

    plt.subplot(1,1,1)
    plt.plot(range(len(count_ts.keys())), count_ts.values(), lw=0.5, color = 'blue')

    # plt.xticks(range(0, len(count_ts.keys()), 86400), [datetime.fromtimestamp(x) for x in count_ts.keys() if (x % )])
    # plt.plot(range(len(timeseries_ann_6893)), timeseries_ann_6893, lw=0.95, color = 'k')
    output = str(random.randint(1, 1000))
    print output + '.png'
    fig.savefig(output, bboxes_inches = '30', dpi = 400)

    print 'first_ts -> ' + str(dt.datetime.fromtimestamp(first_ts).date())
    print 'count_ts[bin] -> ' + str(count_ts)
    print 'count_origin[bin] -> ' + str(count_origin2)
    print 'count_ts_upds_prefixes[bin] -> ' + str(count_ts_upds_prefixes)

    for k, v in count_ts_upds_prefixes.iteritems():
        print 'prefixes ' + str(k) + ' -> ' + str(len(v))

    for k, v in count_ts.iteritems():
        print 'upds -> ' + str(k) + ' -> ' + str(v)
    # plt.show()

if __name__ == '__main__':
    main()
