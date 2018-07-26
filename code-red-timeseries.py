import sys
import os, time
import os.path
from os.path import isfile
from optparse import OptionParser
import datetime as dt
from mrtparse import *
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

import numpy as np
import glob
import random
import cPickle as pickle

import argparse

def dd():
    return defaultdict(int)

def cc(arg):
    return colorConverter.to_rgba(arg, alpha=0.6)

count_prefixes_ts = defaultdict(dd)
count_prefixes_upds = defaultdict(int)
timestamps = OrderedDict()
os.environ['TZ'] = 'US'
time.tzset()

BGPMessageST = [BGP4MP_ST['BGP4MP_MESSAGE'],BGP4MP_ST['BGP4MP_MESSAGE_AS4'],BGP4MP_ST['BGP4MP_MESSAGE_LOCAL'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_LOCAL'],BGP4MP_ST['BGP4MP_MESSAGE_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_LOCAL_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_LOCAL_ADDPATH']]
BGPStateChangeST = [ BGP4MP_ST['BGP4MP_STATE_CHANGE'], BGP4MP_ST['BGP4MP_STATE_CHANGE_AS4']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int)
    parser.add_argument('--start', type=str)
    args = parser.parse_args()

    print(args.days)
    print(args.start)

    number_of_days = 1

    global count_prefixes_ts
    global count_prefixes_upds
    global timestamps

    count_updates = 0

    #Traverse months
    # for i in [6,7,8,9,10,11]
    #Traverse files
    # files = glob.glob("/home/pc/ripe-ris/code-red/update*.gz")
    files = glob.glob("/home/pc/ripe-ris/code-red/updates.20010714.0*.gz")
    files = sorted(files)

    upds_per_file = []
    upds_per_file_as513 = []
    for f in files:
        c = 0
        count_updates = 0
        count_updates_as513 = 0

        d = Reader(f)
        current_day = dt.datetime.fromtimestamp(d.next().mrt.ts).date()

        for m in d:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue
            c += 1

            if m.type == MRT_T['BGP4MP'] or m.type == MRT_T['BGP4MP_ET']:
                if m.subtype in BGPMessageST:
                    if m.bgp.msg is not None:
                        if m.bgp.msg.type == BGP_MSG_T['UPDATE']:
                            count_updates += 1
                            if m.bgp.peer_as == '513':
                                count_updates_as513 += 1

        print f + ': ' + str(count_updates)
        upds_per_file.append(count_updates)
        upds_per_file_as513.append(count_updates_as513)

    timeseries = np.array(upds_per_file)
    timeseries_as513 = np.array(upds_per_file_as513)

    #Dates as axis labels
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    verts = []
    ts1 = list(zip(range(len(timeseries)), timeseries))
    ts2 = list(zip(range(len(timeseries_as513)), timeseries_as513))
    verts.append(ts1)
    verts.append(ts2)
    poly = PolyCollection(verts, facecolors=[cc('r'), cc('g')])
    poly.set_alpha(0.85)
    ax.add_collection3d(poly, zs=[1,2], zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0, len(timeseries))
    ax.set_ylabel('Y')
    ax.set_ylim3d(1, 2)
    ax.set_zlabel('Z')
    ax.set_zlim3d(timeseries.min(), timeseries.max())


    # plt.subplot(1,1,1)
    # plt.plot(range(len(timeseries)), timeseries, lw=1, color = 'black')
    # plt.plot(range(len(timeseries_as513)), timeseries_as513, lw=1, linestyle = '--', color = 'orange')
    # fig.savefig(str(random.randint(1, 1000)),bboxes_inches = 'tight',dpi=240)
    plt.show()


    fig = plt.figure(1)
    plt.subplot(1,1,1)
    plt.plot(range(len(timeseries)), timeseries, lw=1, color = 'black')
    plt.plot(range(len(timeseries_as513)), timeseries_as513, lw=1, linestyle = '--', color = 'orange')
    fig.savefig(str(random.randint(1, 1000)),bboxes_inches = 'tight',dpi=240)
    plt.show()

if __name__ == '__main__':
    main()
