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
from scipy import signal

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
    files = files + glob.glob("/home/pc/ripe-ris/nimda/rrc00/updates.20010913.0*.gz")
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

                    if m.bgp.peer_as == '513':
                        count_updates_as513 += 1
                    # if m.bgp.peer_as == '559':
                    #     count_updates_as559 += 1
                    # if m.bgp.peer_as == '1755':
                    #     count_updates_as1755 += 1
                    # if m.bgp.peer_as == '6893':
                    #     count_updates_as6893 += 1

        print f + ': ' + str(count_updates)


        upds_per_file.append(count_updates)
        announc_per_file.append(count_announcements)
        announc513_per_file.append(count_updates_as513)
        # announc559_per_file.append(count_updates_as559)
        # announc1755_per_file.append(count_updates_as1755)
        # announc6893_per_file.append(count_updates_as6893)

    count_ts = OrderedDict(sorted(count_ts.items()))

    timeseries = np.array(count_ts.values())
    timeseries_ann = np.array(announc_per_file)
    timeseries_ann_513 = np.array(announc513_per_file)
    # timeseries_ann_559 = np.array(announc559_per_file)
    # timeseries_ann_1755 = np.array(announc1755_per_file)
    # timeseries_ann_6893 = np.array(announc6893_per_file)

    #Scales
    # widths = np.array([20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920])
    widths = np.array([20, 40, 80, 160, 320, 640, 1280, 2560, 5120])

    #Wavelet transform
    cwtmatr = signal.cwt(timeseries, signal.ricker, widths)

    #Initialize vars
    plot_index = 0
    std_threshold = 2.5
    n_of_rows = 13

    fig = plt.figure(1)
    print cwtmatr.shape
    for i in range(cwtmatr.shape[0]):

        plt.subplot(cwtmatr.shape[0], 2, plot_index+1)
        plt.plot(range(cwtmatr.shape[1]), cwtmatr[i],lw=0.3)

        plt.subplot(cwtmatr.shape[0], 2, plot_index+2)
        plt.imshow(cwtmatr*5, extent=[-1, 1, 50, 1], cmap='PRGn', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

        plot_index += 2

    plt.show()

    fig = plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.plot(range(len(timeseries)), timeseries, lw=0.95, color = 'green')
    # plt.plot(range(len(timeseries_ann)), timeseries_ann, lw=0.95, color = 'orange', linestyle = '--')
    # plt.plot(range(len(timeseries_ann_513)), timeseries_ann_513, lw=0.95, color = 'red')

    count_ts = OrderedDict(sorted(count_ts.items()))
    print count_ts.keys()


    plt.subplot(1, 1, 1)
    plt.plot(range(len(count_ts.keys())), count_ts.values(), lw=0.95, color = 'blue')

    # plt.xticks(range(0, len(count_ts.keys()), 86400), [datetime.fromtimestamp(x) for x in count_ts.keys() if (x % )])
    # plt.plot(range(len(timeseries_ann_559)), timeseries_ann_559, lw=0.95, color = 'blue')
    # plt.plot(range(len(timeseries_ann_1755)), timeseries_ann_1755, lw=0.95, color = 'pink')
    # plt.plot(range(len(timeseries_ann_6893)), timeseries_ann_6893, lw=0.95, color = 'k')

    output = str(random.randint(1, 1000))
    print output + '.png'
    fig.savefig(output, bboxes_inches = '30', dpi = 400)
    plt.show()

if __name__ == '__main__':
    main()
