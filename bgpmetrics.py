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
import pdir
from getsizeoflib import total_size
from guppy import hpy
import gc

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
            # and m.bgp.msg.type == BGP_MSG_T['UPDATE']

def is_bgp_open(m):
    return (m.type == MRT_T['BGP4MP'] \
            or m.type == MRT_T['BGP4MP_ET']) \
            and m.subtype in BGPMessageST \
            and m.bgp.msg is not None \
            and m.bgp.msg.type == BGP_MSG_T['OPEN']

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
        self.dup_announcements = defaultdict(int)
        self.new_ann_after_wd = defaultdict(int)
        self.flap_announcements = defaultdict(int)
        self.ann_after_wd_unknown = defaultdict(int)
        self.attr_count = defaultdict(int)
        self.max_prefix = defaultdict(int)
        self.mean_prefix = defaultdict(int)
        self.count_origin = defaultdict(dd)
        self.count_ts_upds_ases = defaultdict(dd)
        self.upds_prefixes = defaultdict(dd)
        self.first_ts = 0
        self.diff_counter = defaultdict(int)
        self.error_counter = defaultdict(int)
        self.msg_counter = defaultdict(int)

        #Routing table
        self.prefix_lookup = defaultdict(dddlist)
        self.prefix_withdrawals = defaultdict(dd)
        self.prefix_history = defaultdict(ddlist)
        self.prefix_wd = set()
        self.prefix_dup = set()
        self.prefix_imp = set()
        self.prefix_nada = set()
        self.prefix_flap = set()
        self.count_updates = 0
        self.count_announcements = 0
        self.counter = 0
        self.count_msgs = defaultdict(int)
        self.peer_upds = defaultdict(int)

    def add(self, file):
        #init
        self.count_updates = 0
        self.count_announcements = 0
        self.peer_upds = defaultdict(int)
        d = Reader(file)

        if self.first_ts == 0:
            self.first_ts = d.next().mrt.ts

        gc.collect()

        for m in d:

            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            if is_bgp_update(m):
                self.count_msgs[BGP_MSG_T[m.bgp.msg.type]] += 1
                
                # if BGP_MSG_T[m.bgp.msg.type] != 'UPDATE':
                #     print m.bgp.peer_as
                #     print BGP_MSG_T[m.bgp.msg.type]

                #Init
                self.bin = (m.ts - self.first_ts)/self.bin_size
                window = (m.ts - self.first_ts)/self.window_size

                #Total number of annoucements/withdrawals/updates
                self.count_updates += 1
                self.updates[self.bin] += 1
                self.peer_upds[m.bgp.peer_as] += 1
                print m.bgp.peer_as
                
                if m.bgp.msg.nlri is not None:
                    self.classify_announcement(m)
                    # self.clasify_as_path(m)
                self.classify_withdrawal(m)

                self.count_origin_attr(m)

            elif is_bgp_open(m):
                print_bgp4mp(m)
                pass

    def classify_as_path(self, m):
        for attr in m.bgp.msg.attr:
            if BGP_ATTR_T[attr.type] == 'AS_PATH':
                for as_path in attr.as_path:
                    pass
                    #   - [ ] Maximum AS-PATH length
                    #   - [ ] Average AS-PATH length
                    #   - [ ] Maximum unique AS-PATH length
                    #   - [ ] Average unique AS-PATH length
                    #   - [ ] Maximum of rare ASes in the path
                    #   - [ ] Average of rare ASes in the path
                    #   - [ ] Maximum edit distance
                    #   - [ ] Average edit distance
                    #   - [ ] Maximum edit distance equals $n$ ($n = 1,2,...$)
                    #   - [ ] Maximum AS-path edit distance equals $n$ ($n = 1,2,...$)
                    # - Stateful
                    #   - [ ] Observation of rare ASes in the path
                    #   - [ ] Announcement to longer path
                    #   - [ ] Announcement to shorter path
                    #   - [ ] AS-PATH change according to geographic location
                    #   - [ ] Prefix origin change
                    #   - [ ] Number of new paths announced after withdrawing an old path
                    #   - [ ] Number of new-path announcements
                    #   - [ ] Interarrival time of different types of events (average)
                    #   - [ ] Interarrival time of different types of events (standard deviation)

    def count_origin_attr(self, m):
        if m.bgp.msg.attr is not None:
            for attr in m.bgp.msg.attr:
                if BGP_ATTR_T[attr.type] == 'ORIGIN':
                    self.count_origin[self.bin][attr.origin] += 1

    def classify_withdrawal(self, m):
        self.peer_upds[m.bgp.peer_as] += 1
        
        if (m.bgp.msg.wd_len > 0):
            self.withdrawals[self.bin] += 1

            for nlri in m.bgp.msg.withdrawn:
                prefix = nlri.prefix + '/' + str(nlri.plen)
                self.prefix_withdrawals[m.bgp.peer_as][prefix] = True
                self.print_classification(m, 'WITHDRAW', prefix)
                # self.prefix_history[m.bgp.peer_as][prefix].append(m)
                self.msg_counter[m.bgp.peer_as + '@' + prefix] += 1

    def classify_announcement(self, m):
        self.peer_upds[m.bgp.peer_as] += 1

        for nlri in m.bgp.msg.nlri:
            self.announcements[self.bin] += 1
            prefix = nlri.prefix + '/' + str(nlri.plen)
            self.upds_prefixes[self.bin][prefix] += 1
            #Store history
            # self.prefix_history[m.bgp.peer_as][prefix].append(m)
            self.msg_counter[m.bgp.peer_as + '@' + prefix] += 1

            if self.prefix_lookup[m.bgp.peer_as].has_key(prefix) and not self.prefix_withdrawals[m.bgp.peer_as][prefix]:
                self.classify_reannouncement(m, prefix)
            else:
                self.classify_new_announcement(m, prefix)


    def classify_reannouncement(self, m, prefix):
        is_implicit_wd = False
        is_implicit_dpath = False

        current_attr = self.prefix_lookup[m.bgp.peer_as][prefix]

        self.prefix_lookup[m.bgp.peer_as][prefix] = defaultdict(list)

        #If update msg and RIB state have a diff number of attributes, then it's a implicit wd
        if len(current_attr.keys()) != len(m.bgp.msg.attr):
            is_implicit_wd = True

        #Traverse attributes
        for new_attr in m.bgp.msg.attr:
            attr_name = BGP_ATTR_T[new_attr.type]

            #Check if there is different attributes
            if not self.is_equal(new_attr, current_attr):
                is_implicit_wd = True
                if attr_name == 'AS_PATH':
                    is_implicit_dpath = True
            self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[new_attr.type]] = new_attr

        #Figure it out which counter will be incremented
        if is_implicit_wd:
            # self.prefix_imp.add(prefix)
            if is_implicit_dpath:
                self.implicit_withdrawals_dpath[self.bin] += 1
                self.print_classification(m, 'IMPLICIT_DIFF_PATH', prefix)
            else:
                self.implicit_withdrawals_spath[self.bin] += 1
                self.print_classification(m, 'IMPLICIT_SAME_PATH', prefix)
        else:
            self.dup_announcements[self.bin] += 1
            self.print_classification(m, 'DUPLICATE', prefix)

        del current_attr

    def classify_new_announcement(self, m, prefix):
        if not self.prefix_withdrawals[m.bgp.peer_as][prefix]:
            self.new_announcements[self.bin] += 1
            for attr in m.bgp.msg.attr:
                self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[attr.type]] = attr
            # self.print_classification(m, 'NEW ANNOUNCEMENT', prefix)

        elif self.prefix_lookup[m.bgp.peer_as][prefix]['ORIGIN'] != []:
            #Init vars
            self.prefix_withdrawals[m.bgp.peer_as][prefix] = False
            current_attr = self.prefix_lookup[m.bgp.peer_as][prefix]
            self.prefix_lookup[m.bgp.peer_as][prefix] = defaultdict(list)

            is_diff_announcement = False
            #Traverse attributes
            for new_attr in m.bgp.msg.attr:
                attr_name = BGP_ATTR_T[new_attr.type]
                #Check if there is different attributes
                if not self.is_equal(new_attr, current_attr):
                    is_diff_announcement = True

                self.prefix_lookup[m.bgp.peer_as][prefix][attr_name] = new_attr
            #Figure it out which counter will be incremented
            if is_diff_announcement:
                self.new_ann_after_wd[self.bin] += 1
                # self.prefix_nada.add(prefix)
                # self.print_classification(m, 'NEW ANN. AFTER WITHDRAW', prefix)
            else:
                self.flap_announcements[self.bin] += 1
                # self.prefix_flap.add(prefix)
                # self.print_classification(m, 'FLAP', prefix)
            del current_attr
        else:
            self.ann_after_wd_unknown[self.bin] += 1
            for attr in m.bgp.msg.attr:
                self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[attr.type]] = attr
            # self.print_classification(m, 'ANN. AFTER WITHDRAW - UNKNOWN', prefix)

    def is_equal(self, new_attr, old_attr):
        if BGP_ATTR_T[new_attr.type] == 'ORIGIN':
            if new_attr.origin <> old_attr['ORIGIN'].origin:
                self.diff_counter['ORIGIN'] += 1
            return new_attr.origin == old_attr['ORIGIN'].origin

        elif BGP_ATTR_T[new_attr.type] == 'AS_PATH':
            if new_attr.as_path[0]['val'] <> old_attr['AS_PATH'].as_path[0]['val']:
                self.diff_counter['AS_PATH'] += 1
            return new_attr.as_path[0]['val'] == old_attr['AS_PATH'].as_path[0]['val']

        elif BGP_ATTR_T[new_attr.type] == 'NEXT_HOP':
            if new_attr.next_hop <> old_attr['NEXT_HOP'].next_hop:
                self.diff_counter['NEXT_HOP'] += 1
            return new_attr.next_hop == old_attr['NEXT_HOP'].next_hop

        elif BGP_ATTR_T[new_attr.type] == 'MULTI_EXIT_DISC':
            if new_attr.med <> old_attr['MULTI_EXIT_DISC'].med:
                self.diff_counter['MULTI_EXIT_DISC'] += 1
            return (old_attr['MULTI_EXIT_DISC'] != []) and (new_attr.med == old_attr['MULTI_EXIT_DISC'].med)

        elif BGP_ATTR_T[new_attr.type] == 'ATOMIC_AGGREGATE':
            if not old_attr.has_key('ATOMIC_AGGREGATE'):
                self.diff_counter['ATOMIC_AGGREGATE'] += 1
                return False
            else:
                return True == old_attr['ATOMIC_AGGREGATE']

        elif BGP_ATTR_T[new_attr.type] == 'AGGREGATOR':
            if old_attr['AGGREGATOR'] != [] and new_attr.aggr <> old_attr['AGGREGATOR'].aggr:
                self.diff_counter['AGGREGATOR'] += 1
            return (old_attr['AGGREGATOR'] != []) and (new_attr.aggr == old_attr['AGGREGATOR'].aggr)

        elif BGP_ATTR_T[new_attr.type] == 'COMMUNITY':
            if old_attr['COMMUNITY'] != [] and new_attr.comm <> old_attr['COMMUNITY'].comm:
                self.diff_counter['COMMUNITY'] += 1
            return (old_attr['COMMUNITY'] != []) and (new_attr.comm == old_attr['COMMUNITY'].comm)

    def plot(self):
        for bin, prefix_count in self.upds_prefixes.iteritems():
            self.max_prefix[bin] = np.array(self.upds_prefixes[bin].values()).max()
            self.mean_prefix[bin] = np.array(self.upds_prefixes[bin].values()).mean()

        self.sort_timeseries()
        self.fill_blanks_timeseries()
        self.plot_timeseries()
        # self.print_dicts()

        for prefix in self.prefix_nada:
             for peer in self.prefix_history.keys():
                 qtd_bgp_msgs = len(self.prefix_history[peer][prefix])
                 # if qtd_bgp_msgs > 20:
                     # print peer + ' @ ' +  prefix + '->' + str(qtd_bgp_msgs)
                     # self.print_prefix_history(peer, prefix)
                     # return

        # print self.diff_counter
        # print self.error_counter
        prefix_lookup_size = 0
        c = 0
        for peers, prefixes in self.prefix_lookup.iteritems():
            prefix_lookup_size += sys.getsizeof(peers)
            prefix_lookup_size += sys.getsizeof(prefixes)

            for prefix, attrs in prefixes.iteritems():
                prefix_lookup_size += sys.getsizeof(prefix)
                prefix_lookup_size += sys.getsizeof(attrs)

                for attr_name_, attr_ in attrs.iteritems():
                    # print pdir(attr_)
                    prefix_lookup_size += sys.getsizeof(attr_)
                    prefix_lookup_size += sys.getsizeof(attr_name_)
                    c += 1

        # print 'self.upds_prefixes ->' + str(total_size(self.upds_prefixes)/1024) + 'KB'
        # print 'self.prefix_withdrawals ->' + str(total_size(self.prefix_withdrawals)/1024) + 'KB'
        # print 'self.prefix_lookup ->' + str(total_size(self.prefix_lookup)/1024) + 'KB'
        # print 'self.prefix_lookup2 ->' + str(prefix_lookup_size/1024) + 'KB'

        # print 'self.upds_prefixes ->' + str(len(self.upds_prefixes.keys())) + ' keys'
        # print 'self.prefix_withdrawals ->' + str(len(self.prefix_withdrawals.keys())) + ' keys'

        # prefix_lookup_counter = 0
        # for peers, prefixes in self.prefix_lookup.iteritems():
        #     prefix_lookup_counter += len(prefixes.keys())
        #
        # print 'self.prefix_lookup ->' + str(prefix_lookup_counter) + ' keys'
        # print 'TEST ->' + str(total_size(self.prefix_lookup['3549']['65.202.5.0/24']))
        #
        # # prefix_heavy_hitters = dict(sorted(self.msg_counter.items(), key = operator.itemgetter(1)))
        # prefix_heavy_hitters = sorted(self.msg_counter.items(), key = operator.itemgetter(1), reverse = True)
        # print prefix_heavy_hitters[0:5]
        # print total_size(self.prefix_lookup[])

    def print_classification(self, m, type, prefix):
        if prefix == '' and m.bgp.peer_as == '':
            print '#'*15 + type + '#'*15
            print 'Timestamp: %s' % (dt.datetime.fromtimestamp(m.ts))
            print_bgp4mp(m)

    def print_prefix_history(self, peer, prefix):
        for msg in self.prefix_history[peer][prefix]:
            print 'Timestamp: %s' % (dt.datetime.fromtimestamp(msg.ts))
            print_bgp4mp(msg)
            print '_'*80
            print ''
            print ''

    def print_dicts(self):
        self.print_dict('updates', self.updates)
        # self.print_dict('upds_prefixes', self.upds_prefixes)
        self.print_dict('implicit_withdrawals_dpath', self.implicit_withdrawals_dpath)
        self.print_dict('implicit_withdrawals_spath', self.implicit_withdrawals_spath)
        self.print_dict('dup_announcements', self.dup_announcements)
        self.print_dict('new_announcements', self.new_announcements)
        self.print_dict('new_ann_after_wd', self.new_ann_after_wd)
        self.print_dict('flap_announcements', self.flap_announcements)

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
        # os.system('shotwell ' + output + ' &')

    def sort_dict(self, unsort_dict):
        return defaultdict(int, dict(sorted(unsort_dict.items(), key = operator.itemgetter(0))))

    def print_dict(self, name, dict):
        for k, v in dict.iteritems():
            print name + '  ' + str(dt.datetime.fromtimestamp(self.first_ts + self.bin_size * k)) + ' -> ' + str(v)

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
        self.dup_announcements = self.sort_dict(self.dup_announcements)
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
            self.dup_announcements[i]
            self.new_announcements[i]
            self.new_ann_after_wd[i]
            self.flap_announcements[i]
