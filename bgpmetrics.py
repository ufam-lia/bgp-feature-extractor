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

def dd():
    return defaultdict(int)

def ddlist():
    return defaultdict(list)

def ddd():
    return defaultdict(dd)

def dddlist():
    return defaultdict(ddlist)

BGPMessageST = [BGP4MP_ST['BGP4MP_MESSAGE'],BGP4MP_ST['BGP4MP_MESSAGE_AS4'],BGP4MP_ST['BGP4MP_MESSAGE_LOCAL'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_LOCAL'],BGP4MP_ST['BGP4MP_MESSAGE_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_LOCAL_ADDPATH'],BGP4MP_ST['BGP4MP_MESSAGE_AS4_LOCAL_ADDPATH']]
BGPStateChangeST = [ BGP4MP_ST['BGP4MP_STATE_CHANGE'], BGP4MP_ST['BGP4MP_STATE_CHANGE_AS4']]


class Metrics(object):
    """docstring for Metrics."""
    def __init__(self):
        super(Metrics, self).__init__()

        self.bin_size = 60*60
        self.window_size = 60
        self.count_ts = defaultdict(int)

        #Volume features init
        self.updates = defaultdict(int)
        self.withdrawals = defaultdict(int)
        self.implicit_withdrawals_spath = defaultdict(int)
        self.implicit_withdrawals_dpath = defaultdict(int)
        self.announcements = defaultdict(int)
        self.new_announcements = defaultdict(int)
        self.dup1_announcements = defaultdict(int)
        self.dup2_announcements = defaultdict(int)
        self.attr_count = defaultdict(int)
        self.max_prefix = defaultdict(int)
        self.mean_prefix = defaultdict(int)
        self.count_origin = defaultdict(dd)
        self.count_ts_upds_ases = defaultdict(dd)
        self.upds_prefixes = defaultdict(dd)
        self.first_ts = 0

        #Routing table
        self.prefix_lookup = defaultdict(ddd)
        self.prefix_history = defaultdict(ddlist)
        self.prefix_wd = []
        self.prefix_dup = []
        self.prefix_imp = []
        self.count_updates = 0
        self.count_announcements = 0

    def add(self, file):
        self.count_updates = 0
        self.count_announcements = 0

        d = Reader(file)
        current_day = dt.datetime.fromtimestamp(d.next().mrt.ts).date()
        if self.first_ts == 0:
            self.first_ts = d.next().mrt.ts

        for m in d:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            if (m.type == MRT_T['BGP4MP'] or m.type == MRT_T['BGP4MP_ET']) \
               and m.subtype in BGPMessageST \
               and m.bgp.msg is not None \
               and m.bgp.msg.type == BGP_MSG_T['UPDATE']:
                self.count_updates += 1
                #Total number of annoucements/withdrawals/updates
                bin = (m.ts - self.first_ts)/self.bin_size
                window = (m.ts - self.first_ts)/self.window_size
                self.updates[bin] += 1
                # updates_ases[bin][m.bgp.peer_as] += 1

                if m.bgp.msg.nlri is not None:
                    # print_mrt(m)
                    # print_bgp4mp(m)
                    # print '_'*60

                    for nlri in m.bgp.msg.nlri:
                        self.announcements[bin] += 1
                        prefix = nlri.prefix + '/' + str(nlri.plen)
                        self.upds_prefixes[bin][prefix] += 1
                        # self.prefix_history[m.bgp.peer_as][prefix].append((m, 'A'))

                        if self.prefix_lookup[m.bgp.peer_as].has_key(prefix):
                            #Init vars
                            is_implicit_wd = False
                            is_implicit_dpath = False
                            current_attrs = self.prefix_lookup[m.bgp.peer_as][prefix]
                            self.prefix_lookup[m.bgp.peer_as][prefix] = defaultdict(str)

                            #Traverse attributes
                            for attr in m.bgp.msg.attr:
                                if BGP_ATTR_T[attr.type] == 'ORIGIN':
                                    if attr.origin <> current_attrs['ORIGIN']:
                                        self.prefix_lookup[m.bgp.peer_as][prefix]['ORIGIN'] = attr.origin
                                        is_implicit_wd = True

                                elif BGP_ATTR_T[attr.type] == 'AS_PATH':
                                    if attr.as_path != current_attrs['AS_PATH']:
                                        self.prefix_lookup[m.bgp.peer_as][prefix]['AS_PATH'] = attr.as_path
                                        is_implicit_wd = True
                                        is_implicit_dpath = True

                                elif BGP_ATTR_T[attr.type] == 'NEXT_HOP':
                                    if attr.next_hop != current_attrs['NEXT_HOP']:
                                        self.prefix_lookup[m.bgp.peer_as][prefix]['NEXT_HOP'] = attr.next_hop
                                        is_implicit_wd = True

                                elif BGP_ATTR_T[attr.type] == 'MULTI_EXIT_DISC':
                                    if attr.med != current_attrs['MULTI_EXIT_DISC']:
                                        self.prefix_lookup[m.bgp.peer_as][prefix]['MULTI_EXIT_DISC'] = attr.med
                                        is_implicit_wd = True

                                elif BGP_ATTR_T[attr.type] == 'ATOMIC_AGGREGATE':
                                    if current_attrs['ATOMIC_AGGREGATE'] == True:
                                        self.prefix_lookup[m.bgp.peer_as][prefix]['ATOMIC_AGGREGATE'] = True
                                        is_implicit_wd = True

                                elif BGP_ATTR_T[attr.type] == 'AGGREGATOR':
                                    if attr.aggr != current_attrs['AGGREGATOR']:
                                        self.prefix_lookup[m.bgp.peer_as][prefix]['AGGREGATOR'] = attr.aggr
                                        is_implicit_wd = True

                                elif BGP_ATTR_T[attr.type] == 'COMMUNITY':
                                    if attr.comm != current_attrs['COMMUNITY']:
                                        self.prefix_lookup[m.bgp.peer_as][prefix]['COMMUNITY'] = attr.comm
                                        is_implicit_wd = True

                            if is_implicit_wd == True:
                                if is_implicit_dpath == True:
                                    self.implicit_withdrawals_dpath[bin] += 1
                                else:
                                    self.implicit_withdrawals_spath[bin] += 1
                            else:
                                self.dup1_announcements[bin] += 1
                        else:
                            self.new_announcements[bin] += 1

                            for attr in m.bgp.msg.attr:
                                self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[attr.type]] = attr

                if (m.bgp.msg.wd_len > 0):
                    self.withdrawals[bin] += 1
                    for nlri in m.bgp.msg.withdrawn:
                        prefix = nlri.prefix + '/' + str(nlri.plen)
                        peer = m.bgp.peer_as

                        # if len(prefix_history[peer][prefix]) > 0  and \
                        #     prefix_history[peer][prefix][-1][1] == 'W':
                        #     prefix_wd.append((peer, prefix))
                        self.prefix_history[peer][prefix].append((m, 'W'))



                if m.bgp.msg.attr is not None:
                    for attr in m.bgp.msg.attr:
                        if BGP_ATTR_T[attr.type] == 'ORIGIN':
                            self.count_origin[bin][attr.origin] += 1

        print file + ': ' + str(self.count_updates)

    def sort_dict(self, unsort_dict):
        return defaultdict(int, dict(sorted(unsort_dict.items(), key = operator.itemgetter(0))))

    def plot(self):
        for bin, prefix_count in self.upds_prefixes.iteritems():
            self.max_prefix[bin] = np.array(self.upds_prefixes[bin].values()).max()
            self.mean_prefix[bin] = np.array(self.upds_prefixes[bin].values()).mean()

        #Ordering timeseries
        self.updates = self.sort_dict(self.updates)
        self.announcements = self.sort_dict(self.announcements)
        self.withdrawals = self.sort_dict(self.withdrawals)
        self.max_prefix = self.sort_dict(self.max_prefix)
        self.mean_prefix = self.sort_dict(self.mean_prefix)
        self.implicit_withdrawals_dpath = self.sort_dict(self.implicit_withdrawals_dpath)
        self.implicit_withdrawals_spath = self.sort_dict(self.implicit_withdrawals_spath)
        self.dup1_announcements = self.sort_dict(self.dup1_announcements)
        self.new_announcements = self.sort_dict(self.new_announcements)


        #Filling blanks in the timeseries
        for i in range(self.updates.keys()[-1]):
            self.updates[i]
            self.announcements[i]
            self.withdrawals[i]
            self.max_prefix[i]
            self.mean_prefix[i]
            self.implicit_withdrawals_dpath[i]
            self.implicit_withdrawals_spath[i]
            self.dup1_announcements[i]
            self.new_announcements[i]

        fig = plt.figure(1)
        plt.subplot(1,2,1)
        plt.plot(range(len(self.updates.keys())), self.updates.values(), lw=1.25, color = 'black')
        plt.plot(range(len(self.announcements.keys())), self.announcements.values(), lw=0.5, color = 'blue')
        plt.plot(range(len(self.withdrawals.keys())), self.withdrawals.values(), lw=0.5, color = 'red')

        plt.subplot(1,2,2)
        plt.plot(range(len(self.max_prefix.keys())), self.max_prefix.values(), lw=0.5, color = 'blue')
        plt.plot(range(len(self.mean_prefix.keys())), self.mean_prefix.values(), lw=0.5, color = 'red')

        # plt.xticks(range(0, len(count_ts.keys()), 86400), [datetime.fromtimestamp(x) for x in count_ts.keys() if (x % )])
        # plt.plot(range(len(timeseries_ann_6893)), timeseries_ann_6893, lw=0.95, color = 'k')
        output = str(random.randint(1, 1000)) + '.png'
        fig.savefig(output, bboxes_inches = '30', dpi = 400)
        print output
        os.system('shotwell ' + output + ' &')

        print 'first_ts -> ' + str(dt.datetime.fromtimestamp(self.first_ts))
        # print 'count_origin[bin] -> ' + str(count_origin)
        # print 'upds_prefixes[bin] -> ' + str(upds_prefixes)

        for k, v in self.upds_prefixes.iteritems():
            print 'prefixes ' + str(k) + ' -> ' + str(len(v))

        for k, v in self.updates.iteritems():
            print 'upds -> ' + str(k) + ' -> ' + str(v)

        for k, v in self.upds_prefixes.iteritems():
            print 'upds_prefixes -> ' + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size*k)) + ' -> ' + str(len(v))

        for k, v in self.implicit_withdrawals_dpath.iteritems():
            print 'implicit_withdrawals_dpath -> ' + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size*k)) + ' -> ' + str(v)

        for k, v in self.implicit_withdrawals_spath.iteritems():
            print 'implicit_withdrawals_spath -> ' + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size*k)) + ' -> ' + str(v)

        for k, v in self.dup1_announcements.iteritems():
            print 'dup1_announcements -> ' + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size*k)) + ' -> ' + str(v)

        for k, v in self.new_announcements.iteritems():
            print 'new_announcements -> ' + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size*k)) + ' -> ' + str(v)

        for k, v in self.announcements.iteritems():
            print 'announcements -> ' + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size*k)) + ' -> ' + str(v)
