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
# from guppy import hpy
import gc

os.environ['TZ'] = 'US'
time.tzset()

def dd():
    return defaultdict(int)
def ddarr():
    return defaultdict(np.zeros(1))
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

def is_table_dump(m):
    return (m.type == MRT_T['TABLE_DUMP'] \
            or m.type == MRT_T['TABLE_DUMP_V2']) \
            and m.td is not None

def is_bgp_open(m):
    return (m.type == MRT_T['BGP4MP'] \
            or m.type == MRT_T['BGP4MP_ET']) \
            and m.subtype in BGPMessageST \
            and m.bgp.msg is not None \
            and m.bgp.msg.type == BGP_MSG_T['OPEN']

def edit_distance(l1, l2):
    rows = len(l1)+1
    cols = len(l2)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if l1[row-1] == l2[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
    return dist[row][col]

class Metrics(object):
    """docstring for Metrics."""
    def __init__(self):
        super(Metrics, self).__init__()

        self.bin_size = 60*5
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

        #AS path features
        self.as_paths = []
        self.num_of_paths_rcvd = defaultdict(int)
        self.unique_as_paths = [] #Ignore prepending
        self.distinct_as_paths = set() #Ignore repeated AS paths
        self.as_paths_distribution = defaultdict(int)
        self.as_path_max_length = defaultdict(int)
        self.as_path_avg_length = defaultdict(int)
        self.unique_as_path_max = defaultdict(int)
        self.unique_as_path_avg = defaultdict(int)
        self.window_end = 0

        self.rare_threshold = 0
        self.rare_ases_iteration = 1
        self.rare_ases_max = defaultdict(int)
        self.rare_ases_avg = defaultdict(int)
        self.number_rare_ases = defaultdict(int)
        self.rare_ases_counter = defaultdict(int)

        self.edit_distance_max = defaultdict(int)
        self.edit_distance_avg = defaultdict(int)
        self.edit_distance_counter = defaultdict(int)
        self.edit_distance_unique_counter = defaultdict(int)
        self.edit_distance_dict = defaultdict(dd)
        self.edit_distance_unique_dict = defaultdict(dd)

        # - Stateless
        #   - [x] Maximum AS-PATH length (any)
        #   - [x] Average AS-PATH length (any)
        #   - [x] Maximum unique AS-PATH length (any)
        #   - [x] Average unique AS-PATH length (any)
        # - Stateful
        #   - [ ] Maximum edit distance (reann) (new aft. wd)
        #   - [ ] Average edit distance  (reann) (new aft. wd)
        #   - [ ] Maximum edit distance equals $n$ ($n = 1,2,...$) (reann) (new aft. wd)
        #   - [ ] Maximum AS-path edit distance equals $n$ ($n = 1,2,...$) (reann) (new aft. wd)
        #   - [x] Observation of rare ASes in the path (any)
        #   - [x] Maximum of rare ASes in the path (any)
        #   - [x] Average of rare ASes in the path (any)
        #   - [ ] Announcement to longer path  (reann) (new aft. wd)
        #   - [ ] Announcement to shorter path  (reann) (new aft. wd)
        #   - [ ] AS-PATH change according to geographic location
        #   - [ ] Prefix origin change (reann)
        #   - [ ] Number of new paths announced after withdrawing an old path (new aft. wd)
        #   - [ ] Interarrival time of different types of events (average)
        #   - [ ] Interarrival time of different types of events (standard deviation)

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
        self.prfx_set = defaultdict(dd)
        self.prefix_found = 0
        self.peer_found = 0
        self.table_exchange_period = False
        self.rib_count = 0
        self.table_exchange_period = dict()

    def init_rib(self, file):
        d = Reader(file)

        prfx_count = defaultdict(int)
        peer_count = defaultdict(int)

        for m in d:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            if is_table_dump(m):
                peer = m.td.peer_as
                prefix = str(m.td.prefix) + '/' + str(m.td.plen)

                self.print_classification(m, 'RIB', prefix)
                self.prfx_set[peer][prefix] += 1
                prfx_count[prefix] += 1
                peer_count[peer] += 1

                for attr in m.td.attr:
                    self.prefix_lookup[peer][prefix][BGP_ATTR_T[attr.type]] = attr

        rolou = False

        for prefix, peers in prfx_count.iteritems():
            if peers > 1:
                rolou = True
                print prefix

        for peer, prefix_count in peer_count.iteritems():
            print str(peer) + '->' + str(prefix_count)

        if not rolou:
            print 'not rolou'

    def add_updates(self, file):
        #init
        self.count_updates = 0
        self.count_announcements = 0
        self.peer_upds = defaultdict(int)
        d = Reader(file)

        if self.first_ts == 0:
            self.first_ts = d.next().mrt.ts

        m = d.next()
        while m:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            if is_bgp_update(m):
                self.count_msgs[BGP_MSG_T[m.bgp.msg.type]] += 1

                # if BGP_MSG_T[m.bgp.msg.type] != 'UPDATE':
                #     print m.bgp.peer_as
                #     print BGP_MSG_T[m.bgp.msg.type]

                self.bin = (m.ts - self.first_ts)/self.bin_size
                #Init
                window = (m.ts - self.first_ts)/self.window_size

                #Total number of annoucements/withdrawals/updates
                self.count_updates += 1
                self.updates[self.bin] += 1
                self.peer_upds[m.bgp.peer_as] += 1

                if m.bgp.msg.nlri is not None:
                    self.classify_announcement(m)
                self.classify_withdrawal(m)

                self.count_origin_attr(m)

            elif is_bgp_open(m):
                print_bgp4mp(m)
                pass

            m = next(d, None)
            # del m

    def classify_as_path(self, m, attr, prefix):
        for as_path in attr.as_path:
            if as_path['type'] == 2:
                unique_as_path = set(as_path['val'])
                # self.num_of_paths_rcvd += 1

                self.as_paths.append(as_path)
                self.unique_as_paths.append(unique_as_path) #Ignore prepending
                self.distinct_as_paths.add(str(as_path['val'])) #Ignore repeated AS paths
                rare_ases = 0
                for asn in unique_as_path:
                    self.as_paths_distribution[asn] += 1
                    if self.as_paths_distribution[asn] < self.rare_threshold:
                        rare_ases += 1
                        print 'rare_ases ->' + str(rare_ases)

                self.rare_ases_iteration += 1
                if self.rare_ases_iteration % 1000 == 0:
                    self.rare_threshold = np.percentile(np.array(self.as_paths_distribution.values()), 20)
                    print 'self.rare_threshold ->' + str(self.rare_threshold)

                if self.rare_ases_iteration > 1000:
                    if type(self.rare_ases_counter[self.bin]) != int:
                        self.rare_ases_counter[self.bin] = np.append(self.rare_ases_counter[self.bin], rare_ases)
                    else:
                        self.rare_ases_counter[self.bin] = np.array(rare_ases)

                    if rare_ases > self.rare_ases_max[bin]:
                        self.rare_ases_max[bin] = rare_ases
                        print 'self.rare_ases_max[bin] ->' + str(self.rare_ases_max[bin])

                if as_path['len'] > self.as_path_max_length[self.bin]:
                    self.as_path_max_length[self.bin] = as_path['len']
                if len(unique_as_path) > self.unique_as_path_max[self.bin]:
                    self.unique_as_path_max[self.bin] = len(unique_as_path)

                # print self.rare_ases_counter

                if type(self.rare_ases_counter[self.bin]) != int:
                    self.rare_ases_avg[self.bin] = self.rare_ases_counter[self.bin].mean()
                    print self.rare_ases_avg[self.bin]

                self.num_of_paths_rcvd[self.bin] += 1
                self.as_path_avg_length[self.bin] = (as_path['len'] * self.num_of_paths_rcvd[self.bin] + self.as_path_avg_length[self.bin])/self.num_of_paths_rcvd[self.bin]
                self.unique_as_path_avg[self.bin] = (len(unique_as_path) * self.num_of_paths_rcvd[self.bin] + self.unique_as_path_max[self.bin])/self.num_of_paths_rcvd[self.bin]

                #   - [x] Maximum AS-PATH length
                #   - [x] Average AS-PATH length
                #   - [x] Maximum unique AS-PATH length
                #   - [x] Average unique AS-PATH length
                #   - [x] Maximum of rare ASes in the path
                #   - [x] Average of rare ASes in the path
                # - Stateful
                #   - [x] Observation of rare ASes in the path
                #   - [ ] Announcement to longer path
                #   - [ ] Announcement to shorter path
                #   - [ ] AS-PATH change according to geographic location
                #   - [ ] Prefix origin change
                #   - [ ] Number of new paths announced after withdrawing an old path
                #   - [ ] Number of new-path announcements
                #   - [ ] Interarrival time of different types of events (average)
                #   - [ ] Interarrival time of different types of events (standard deviation)

    def calc_edit_distance(self, m, new_path, old_path, prefix):
        dist = edit_distance(new_path, old_path)
        dist_unique = edit_distance(list(set(new_path)),list(set(new_path)))

        self.edit_distance_dict[self.bin][dist] += 1
        self.edit_distance_dict[self.bin][dist_unique] += 1

        if dist > self.edit_distance_max[self.bin]:
            self.edit_distance_max[self.bin] = dist

        if type(self.edit_distance_counter[self.bin]) != int:
            self.edit_distance_counter[self.bin] = np.append(self.edit_distance_counter[self.bin], dist)
            print 'self.edit_distance_counter -> '+ str( self.edit_distance_counter[self.bin].mean())
            # self.edit_distance_avg[self.bin] = self.edit_distance_counter[self.bin].mean()
        else:
            self.edit_distance_counter[self.bin] = np.array(dist)

        if type(self.rare_ases_counter[self.bin]) != int:
            print self.rare_ases_avg[self.bin]

        #   - [x] Maximum edit distance
        #   - [x] Average edit distance
        #   - [x] Maximum AS-path edit distance equals $n$ ($n = 1,2,...$)
        #   - [x] Maximum unique AS-path edit distance equals $n$ ($n = 1,2,...$)


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
            self.dbg_prefix(m, prefix)

            #Store history
            # self.prefix_history[m.bgp.peer_as][prefix].append(m)
            self.msg_counter[m.bgp.peer_as + '@' + prefix] += 1

            if self.prefix_lookup[m.bgp.peer_as].has_key(prefix) and not self.prefix_withdrawals[m.bgp.peer_as][prefix]:
                self.classify_reannouncement(m, prefix)
            else:
                self.classify_new_announcement(m, prefix)

    def dbg_prefix(self, m, prefix):
        if self.prfx_set[m.bgp.peer_as][prefix] == 1:
            self.rib_count += 1
            self.prfx_set[m.bgp.peer_as][prefix] = 0
            self.prefix_found = prefix
            self.peer_found = m.bgp.peer_as

            # print_bgp4mp(m)
            # print '*'*50
            # print self.prefix_lookup[m.bgp.peer_as][prefix]
            # os.abort()


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

            if attr_name == 'AS_PATH':
                self.classify_as_path(m, new_attr, prefix)
                self.calc_edit_distance(m, new_attr.as_path[0]['val'], current_attr['AS_PATH'].as_path[0]['val'], prefix)

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

    def classify_new_announcement(self, m, prefix):
        if not self.prefix_withdrawals[m.bgp.peer_as][prefix]:
            self.new_announcements[self.bin] += 1
            if prefix == self.prefix_found and m.bgp.peer_as == self.peer_found:
                print self.prefix_lookup

            for attr in m.bgp.msg.attr:
                self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[attr.type]] = attr
                #Classify AS PATH
                if BGP_ATTR_T[attr.type] == 'AS_PATH':
                    self.classify_as_path(m, attr, prefix)

            self.print_classification(m, 'NEW ANNOUNCEMENT', prefix)

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
                #Classify AS PATH
                if attr_name == 'AS_PATH':
                    self.classify_as_path(m, new_attr, prefix)

                self.prefix_lookup[m.bgp.peer_as][prefix][attr_name] = new_attr
            #Figure it out which counter will be incremented
            if is_diff_announcement:
                self.new_ann_after_wd[self.bin] += 1
                # self.prefix_nada.add(prefix)
                self.print_classification(m, 'NEW ANN. AFTER WITHDRAW', prefix)
            else:
                self.flap_announcements[self.bin] += 1
                # self.prefix_flap.add(prefix)
                self.print_classification(m, 'FLAP', prefix)
            del current_attr
        else:
            self.ann_after_wd_unknown[self.bin] += 1
            for attr in m.bgp.msg.attr:
                self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[attr.type]] = attr
                attr_name = BGP_ATTR_T[attr.type]

                if attr_name == 'AS_PATH':
                    self.classify_as_path(m, attr, prefix)

            self.print_classification(m, 'ANN. AFTER WITHDRAW - UNKNOWN', prefix)

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
        # print self.as_paths_distribution
        # print self.as_path_max_length
        # print self.unique_as_path_max
        # print self.as_path_avg_length
        # print self.unique_as_path_avg
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
        # prefix_lookup_size = 0
        # c = 0
        # for peers, prefixes in self.prefix_lookup.iteritems():
        #     prefix_lookup_size += sys.getsizeof(peers)
        #     prefix_lookup_size += sys.getsizeof(prefixes)
        #
        #     for prefix, attrs in prefixes.iteritems():
        #         prefix_lookup_size += sys.getsizeof(prefix)
        #         prefix_lookup_size += sys.getsizeof(attrs)
        #
        #         for attr_name_, attr_ in attrs.iteritems():
        #             # print pdir(attr_)
        #             prefix_lookup_size += sys.getsizeof(attr_)
        #             prefix_lookup_size += sys.getsizeof(attr_name_)
        #             c += 1
        #
        # print 'self.upds_prefixes ->' + str(total_size(self.upds_prefixes)/1024) + 'KB'
        # print 'self.prefix_withdrawals ->' + str(total_size(self.prefix_withdrawals)/1024) + 'KB'
        # print 'self.prefix_lookup ->' + str(total_size(self.prefix_lookup)/1024) + 'KB'
        # print 'self.prefix_lookup2 ->' + str(prefix_lookup_size/1024) + 'KB'
        #
        # print 'self.upds_prefixes ->' + str(len(self.upds_prefixes.keys())) + ' keys'
        # print 'self.prefix_withdrawals ->' + str(len(self.prefix_withdrawals.keys())) + ' keys'
        #
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
        # if prefix == self.prefix_found and m.bgp.peer_as == self.peer_found:

        if prefix == '2.31.96.0/24':
            if MRT_T[m.type] == 'BGP4MP' or  MRT_T[m.type] == 'BGP4MP_ET':
                peer = m.bgp.peer_as
                if peer == '2686':
                    print '#'*15 + type + '#'*15
                    print 'Timestamp: %s' % (dt.datetime.fromtimestamp(m.ts))
                    print_bgp4mp(m)
            else:
                peer = m.td.peer_as
                if peer == '2686':
                    print '#'*15 + type + '#'*15
                    print 'Timestamp: %s' % (dt.datetime.fromtimestamp(m.ts))
                    print_td(m)

            if type != 'RIB':
                self.print_dicts()
                # os.abort()

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
        return defaultdict(int, dict(sorted(unsort_dict.items(), key = operator.itemgetter(1))))

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
