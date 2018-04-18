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
import operator

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
    days_checked    = 0
    bin_size = 60*15
    window_size = 60
    count_ts = defaultdict(int)

    #Volume features init
    updates = defaultdict(int)
    withdrawals = defaultdict(int)
    implicit_withdrawals = defaultdict(int)
    announcements = defaultdict(int)
    new_announcements = defaultdict(int)
    dup1_announcements = defaultdict(int)
    dup2_announcements = defaultdict(int)

    max_prefix = defaultdict(int)
    mean_prefix = defaultdict(int)
    count_origin = defaultdict(dd)
    count_ts_upds_ases = defaultdict(dd)
    upds_prefixes = defaultdict(dd)
    first_ts = 0

    #Routing table
    prefix_lookup = dict(dict(str))

    for f in files:
        count_updates = 0
        count_announcements = 0
        d = Reader(f)
        current_day = dt.datetime.fromtimestamp(d.next().mrt.ts).date()
        if first_ts == 0:
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
                count_updates += 1
                #Total number of annoucements/withdrawals/updates
                bin = (m.ts - first_ts)/bin_size
                window = (m.ts - first_ts)/window_size
                updates[bin] += 1
                # updates_ases[bin][m.bgp.peer_as] += 1

                if m.bgp.msg.nlri is not None:
                    announcements[bin] += 1
                    for nlri in m.bgp.msg.nlri:
                        prefix = nlri.prefix + '/' + str(nlri.plen)
                        upds_prefixes[bin][prefix] += 1
                        if upds_prefixes[bin][prefix] > 1:
                            #compara com rota anterior
                                #se for a msm rota pode ser dup1 ou dup2
                                    #se for msm attr eh dup1
                                    #senao eh dup2 e implicit withdraw 2
                                #se for rota diferente pode ser um implicit withdrawal
                        else:
                            new_announcements[bin] += 1
                            #guarda rota

                if (m.bgp.msg.wd_len > 0):
                    withdrawals[bin] += 1
                    if len(m.bgp.msg.attr) > 0:
                        print '*' * 500
                        exit


                if m.bgp.msg.attr is not None:
                    for a in m.bgp.msg.attr:
                        if BGP_ATTR_T[a.type] == 'ORIGIN':
                            count_origin[bin][a.origin] += 1
        print f + ': ' + str(count_updates)

    for bin, prefix_count in upds_prefixes.iteritems():
        max_prefix[bin] = np.array(upds_prefixes[bin].values()).max()
        mean_prefix[bin] = np.array(upds_prefixes[bin].values()).mean()


    #Ordering timeseries
    updates = defaultdict(int, dict(sorted(updates.items(), key = operator.itemgetter(0))))
    announcements = defaultdict(int, dict(sorted(announcements.items(), key = operator.itemgetter(0))))
    withdrawals = defaultdict(int, dict(sorted(withdrawals.items(), key = operator.itemgetter(0))))
    max_prefix = defaultdict(int, dict(sorted(max_prefix.items(), key = operator.itemgetter(0))))
    mean_prefix = defaultdict(int, dict(sorted(mean_prefix.items(), key = operator.itemgetter(0))))


    #Filling blanks in the timeseries
    for i in range(updates.keys()[-1]):
        updates[i]
        announcements[i]
        withdrawals[i]
        max_prefix[i]
        mean_prefix[i]

    fig = plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(range(len(updates.keys())), updates.values(), lw=1.25, color = 'black')
    plt.plot(range(len(announcements.keys())), announcements.values(), lw=0.5, color = 'blue')
    plt.plot(range(len(withdrawals.keys())), withdrawals.values(), lw=0.5, color = 'red')

    plt.subplot(1,2,2)
    plt.plot(range(len(max_prefix.keys())), max_prefix.values(), lw=0.5, color = 'blue')
    plt.plot(range(len(mean_prefix.keys())), mean_prefix.values(), lw=0.5, color = 'red')

    # plt.xticks(range(0, len(count_ts.keys()), 86400), [datetime.fromtimestamp(x) for x in count_ts.keys() if (x % )])
    # plt.plot(range(len(timeseries_ann_6893)), timeseries_ann_6893, lw=0.95, color = 'k')
    output = str(random.randint(1, 1000)) + '.png'
    fig.savefig(output, bboxes_inches = '30', dpi = 400)
    print output
    os.system('shotwell ' + output + ' &')

    print 'first_ts -> ' + str(dt.datetime.fromtimestamp(first_ts))
    # print 'count_origin[bin] -> ' + str(count_origin)
    # print 'upds_prefixes[bin] -> ' + str(upds_prefixes)

    for k, v in upds_prefixes.iteritems():
        print 'prefixes ' + str(k) + ' -> ' + str(len(v))

    for k, v in updates.iteritems():
        print 'upds -> ' + str(k) + ' -> ' + str(v)

    for k, v in upds_prefixes.iteritems():
        print 'upds_prefixes -> ' + str(dt.datetime.fromtimestamp(first_ts + bin_size*k)) + ' -> ' + str(len(v))

    plt.show()

if __name__ == '__main__':
    main()
