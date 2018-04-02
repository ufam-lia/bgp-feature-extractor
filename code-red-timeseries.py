import sys
import os, time
import os.path
from os.path import isfile
from optparse import OptionParser
import datetime as dt
from mrtparse import *
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import glob

import cPickle as pickle

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
    number_of_days = 1

    if len(sys.argv) > 1:
        k = int(sys.argv[1])

    global count_prefixes_ts
    global count_prefixes_upds
    global timestamps

    #Traverse months
    # for i in [6,7,8,9,10,11]
    #Traverse files
    files = glob.glob("/home/pc/ripe-ris/code-red/*.gz")
    files = sorted(files)

    c = 0

    for f in files:
        d = Reader(f)
        c += 1
        current_day = dt.datetime.fromtimestamp(d.next().mrt.ts).date()

        for m in d:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            c += 1
            if c % 500 == 0:
                print dt.datetime.fromtimestamp(m.ts)

            if m.type == MRT_T['BGP4MP'] or m.type == MRT_T['BGP4MP_ET']:
                if m.subtype in BGPMessageST:
                    if m.bgp.msg.type == BGP_MSG_T['UPDATE']:
                        #Total updates per prefix
                        if m.bgp.msg.nlri is not None:
                            for nlri in m.bgp.msg.nlri:
                                print str(nlri.prefix) + '/' + str(nlri.plen)
                                # count_prefixes_ts[nlri.prefix + '/' + str(nlri.plen)][m.ts] += 1
                                # count_prefixes_upds[nlri.prefix + '/' + str(nlri.plen)] += 1
                                # timestamps[m.ts] = 1

if __name__ == '__main__':
    main()
