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

def is_bgp_update(m):
    return (m.type == MRT_T['BGP4MP'] \
            or m.type == MRT_T['BGP4MP_ET']) \
            and m.subtype in BGPMessageST \
            and m.bgp.msg is not None \
            and m.bgp.msg.type == BGP_MSG_T['UPDATE']
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
        #init
        self.count_updates = 0
        self.count_announcements = 0
        d = Reader(file)

        if self.first_ts == 0:
            self.first_ts = d.next().mrt.ts

        for m in d:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            if is_bgp_update(m):
                #Init
                bin = (m.ts - self.first_ts)/self.bin_size
                window = (m.ts - self.first_ts)/self.window_size

                #Total number of annoucements/withdrawals/updates
                self.count_updates += 1
                self.updates[bin] += 1
                peer = m.bgp.peer_as
                # updates_ases[bin][m.bgp.peer_as] += 1

                if m.bgp.msg.nlri is not None:
                    # print_mrt(m)
                    # print_bgp4mp(m)
                    # print '_'*60
                    self.classify_announcement(m)
                self.classify_withdrawal(m)

                if m.bgp.msg.attr is not None:
                    for attr in m.bgp.msg.attr:
                        if BGP_ATTR_T[attr.type] == 'ORIGIN':
                            self.count_origin[bin][attr.origin] += 1

    def classify_withdrawal(self, m):
        if (m.bgp.msg.wd_len > 0):
            self.withdrawals[bin] += 1

            for nlri in m.bgp.msg.withdrawn:
                prefix = nlri.prefix + '/' + str(nlri.plen)

                # if len(prefix_history[peer][prefix]) > 0  and \
                #     prefix_history[peer][prefix][-1][1] == 'W':
                #     prefix_wd.append((peer, prefix))
                # self.prefix_history[peer][prefix].append((m, 'W'))

    def classify_announcement(self, m):
        for nlri in m.bgp.msg.nlri:
            self.announcements[bin] += 1
            prefix = nlri.prefix + '/' + str(nlri.plen)
            self.upds_prefixes[bin][prefix] += 1
            # self.prefix_history[m.bgp.peer_as][prefix].append((m, 'A'))

            if self.prefix_lookup[m.bgp.peer_as].has_key(prefix):
                #Init vars
                is_implicit_wd = False
                is_implicit_dpath = False
                current_attr = self.prefix_lookup[m.bgp.peer_as][prefix]
                self.prefix_lookup[m.bgp.peer_as][prefix] = defaultdict(str)

                #Traverse attributes
                for new_attr in m.bgp.msg.attr:
                    attr_name = BGP_ATTR_T[new_attr.type]

                    #Check if there is different attributes
                    if not self.is_equal(new_attr, current_attr):
                        self.prefix_lookup[m.bgp.peer_as][prefix][attr_name] = new_attr

                        is_implicit_wd = True
                        if attr_name == 'AS_PATH':
                            is_implicit_dpath = True
                            break

                #Figure which counter will be incremented
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

    def plot(self):
        for bin, prefix_count in self.upds_prefixes.iteritems():
            self.max_prefix[bin] = np.array(self.upds_prefixes[bin].values()).max()
            self.mean_prefix[bin] = np.array(self.upds_prefixes[bin].values()).mean()

        self.sort_timeseries()
        self.fill_blanks_timeseries()
        self.plot_timeseries()
        self.print_dicts()

    def print_dicts(self):
        self.print_dict('upds_prefixes', self.upds_prefixes)
        self.print_dict('updates', self.updates)
        self.print_dict('upds_prefixes', self.upds_prefixes)
        self.print_dict('implicit_withdrawals_dpath', self.implicit_withdrawals_dpath)
        self.print_dict('implicit_withdrawals_spath', self.implicit_withdrawals_spath)
        self.print_dict('dup1_announcements', self.dup1_announcements)
        self.print_dict('new_announcements', self.new_announcements)
        self.print_dict('announcements', self.announcements)

    def plot_timeseries(self):
        fig = plt.figure(1)
        plt.subplot(1,2,1)
        plt.plot(range(len(self.updates.keys())), self.updates.values(), lw=1.25, color = 'black')
        plt.plot(range(len(self.announcements.keys())), self.announcements.values(), lw=0.5, color = 'blue')
        plt.plot(range(len(self.withdrawals.keys())), self.withdrawals.values(), lw=0.5, color = 'red')

        plt.subplot(1,2,2)
        plt.plot(range(len(self.max_prefix.keys())), self.max_prefix.values(), lw=0.5, color = 'blue')
        plt.plot(range(len(self.mean_prefix.keys())), self.mean_prefix.values(), lw=0.5, color = 'red')

        output = str(random.randint(1, 1000)) + '.png'
        fig.savefig(output, bboxes_inches = '30', dpi = 400)
        print output
        os.system('shotwell ' + output + ' &')

    def sort_dict(self, unsort_dict):
        return defaultdict(int, dict(sorted(unsort_dict.items(), key = operator.itemgetter(0))))

    def print_dict(self, name, dict):
        for k, v in dict.iteritems():
            print name + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size*k)) + ' -> ' + str(v)

    def is_equal(self, new_attr, old_attr):
        if BGP_ATTR_T[new_attr.type] == 'ORIGIN':
            return new_attr.origin == old_attr['ORIGIN']

        elif BGP_ATTR_T[new_attr.type] == 'AS_PATH':
            return new_attr.as_path == old_attr['AS_PATH']

        elif BGP_ATTR_T[new_attr.type] == 'NEXT_HOP':
            return new_attr.next_hop == old_attr['NEXT_HOP']

        elif BGP_ATTR_T[new_attr.type] == 'MULTI_EXIT_DISC':
            return new_attr.med == old_attr['MULTI_EXIT_DISC']

        elif BGP_ATTR_T[new_attr.type] == 'ATOMIC_AGGREGATE':
            return True == old_attr['ATOMIC_AGGREGATE']

        elif BGP_ATTR_T[new_attr.type] == 'AGGREGATOR':
            return new_attr.aggr == old_attr['AGGREGATOR']

        elif BGP_ATTR_T[new_attr.type] == 'COMMUNITY':
            return new_attr.comm == old_attr['COMMUNITY']

    def is_implicit_wd(self, bgp_msg):
        pass

    def sort_timeseries(self):
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

    def fill_blanks_timeseries(self):
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
