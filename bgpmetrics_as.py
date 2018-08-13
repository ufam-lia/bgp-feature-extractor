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
import pandas as pd
from operator import add

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

def is_table_dump_v1(m):
    return m.type == MRT_T['TABLE_DUMP'] and m.td is not None

def is_table_dump_v2(m):
    return m.type == MRT_T['TABLE_DUMP_V2'] and\
          (m.peer is not None or\
           m.rib is not None)

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

class Features(object):
    def __init__(self):
        self.withdrawals = 0
        self.imp_wd = 0
        self.imp_wd_spath = 0
        self.imp_wd_dpath = 0
        self.announcements = 0
        self.nlri_ann = 0
        self.news = 0
        self.dups = 0
        self.wd_dups = 0
        self.nadas = 0
        self.flaps = 0
        self.origin = 0
        self.origin_changes = 0
        self.as_path_max = 0
        self.as_path_avg = 0
        self.unique_as_path_max = 0
        self.unique_as_path_avg = 0
        self.rare_ases_max = 0
        self.rare_ases_avg = 0
        self.number_rare_ases = 0
        self.edit_distance_max = 0
        self.edit_distance_avg = 0
        self.edit_distance_dict = 0
        self.edit_distance_unique_dict = 0
        self.ann_to_shorter = 0
        self.ann_to_longer = 0
        self.timestamp = 0
        self.timestamp2 = 0
        self.class_traffic = 0

    def to_dict(self):
        features = dict()

        features['timestamp'] = self.timestamp
        features['timestamp2'] = self.timestamp2
        features['withdrawals'] = self.withdrawals
        features['imp_wd'] = self.imp_wd
        features['wd_dups'] = self.wd_dups
        features['imp_wd_spath'] = self.imp_wd_spath
        features['imp_wd_dpath'] = self.imp_wd_dpath
        features['nlri_ann'] = self.nlri_ann
        features['announcements'] = self.announcements
        features['news'] = self.news
        features['dups'] = self.dups
        features['nadas'] = self.nadas
        features['flaps'] = self.flaps
        features['origin_changes'] = self.origin_changes
        features['as_path_max'] = self.as_path_max
        features['as_path_avg'] = self.as_path_avg
        features['unique_as_path_max'] = self.unique_as_path_max
        features['unique_as_path_avg'] = self.unique_as_path_avg
        features['rare_ases_max'] = self.rare_ases_max
        features['rare_ases_avg'] = self.rare_ases_avg
        features['number_rare_ases'] = self.number_rare_ases
        features['edit_distance_max'] = self.edit_distance_max
        features['edit_distance_avg'] = self.edit_distance_avg
        features['ann_to_shorter'] = self.ann_to_shorter
        features['ann_to_longer'] = self.ann_to_longer
        features['class'] = self.class_traffic

        for k, v in self.origin.iteritems():
            if k < 11 :
                features['origin_' + str(k)] = v

        for k, v in self.edit_distance_dict.iteritems():
            if k < 11 :
                features['edit_distance_dict_' + str(k)] = v

        for k, v in self.edit_distance_dict.iteritems():
            if k < 11 :
                features['edit_distance_unique_dict_' + str(k)] = v

        return features

    def to_csv(self):
        pass

    def to_dataframe(self):
        return pd.DataFrame(self.to_dict())

class Metrics(object):

    def volume_attr_init(self):
        #Volume features init
        self.updates = defaultdict(int)
        self.withdrawals = defaultdict(int)
        self.imp_wd = defaultdict(int) #*feature*
        self.implicit_withdrawals_spath = defaultdict(int) #*feature*
        self.implicit_withdrawals_dpath = defaultdict(int) #*feature*
        self.announcements = defaultdict(int) #*feature*
        self.nlri_ann = defaultdict(int) #*feature*
        self.new_announcements = defaultdict(int) #*feature*
        self.wd_dups = defaultdict(int) #*feature*
        self.dup_announcements = defaultdict(int) #*feature*
        self.new_ann_after_wd = defaultdict(int) #*feature*
        self.flap_announcements = defaultdict(int) #*feature*
        self.ann_after_wd_unknown = defaultdict(int) #*feature?*
        self.attr_count = defaultdict(int)
        self.max_prefix = defaultdict(int)
        self.mean_prefix = defaultdict(int)
        self.count_origin = defaultdict(dd) #*feature*
        self.count_origin_changes = defaultdict(int) #*feature*
        self.count_ts_upds_ases = defaultdict(dd)
        self.upds_prefixes = defaultdict(dd)
        self.first_ts = 0
        self.diff_counter = defaultdict(int)
        self.error_counter = defaultdict(int)
        self.msg_counter = defaultdict(int)
        self.nlri_ann = defaultdict(int)

    def as_path_attr_init(self):
        #AS path features
        self.as_paths = []
        self.num_of_paths_rcvd = defaultdict(int)
        self.unique_as_paths = [] #Ignore prepending
        self.distinct_as_paths = set() #Ignore repeated AS paths
        self.as_paths_distribution = defaultdict(int)
        self.as_path_max_length = defaultdict(int) #*feature*
        self.as_path_avg_length = defaultdict(int) #*feature*
        self.unique_as_path_max = defaultdict(int) #*feature*
        self.unique_as_path_avg = defaultdict(int) #*feature*
        self.window_end = 0

        self.rare_threshold = 0
        self.rare_ases_iteration = 1
        self.rare_ases_max = defaultdict(int) #*feature*
        self.rare_ases_avg = defaultdict(int) #*feature*
        self.number_rare_ases = defaultdict(int) #*feature*
        self.rare_ases_counter = defaultdict(int)

        self.edit_distance_max = defaultdict(int) #*feature*
        self.edit_distance_avg = defaultdict(int) #*feature*
        self.edit_distance_counter = defaultdict(int)
        self.edit_distance_unique_counter = defaultdict(int)
        self.edit_distance_dict = defaultdict(dd) #*feature*
        self.edit_distance_unique_dict = defaultdict(dd) #*feature*

        self.ann_to_shorter = defaultdict(int) #*feature*
        self.ann_to_longer = defaultdict(int) #*feature*

    def rib_attr_init(self):
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

    def __init__(self):
        super(Metrics, self).__init__()
        if len(sys.argv) > 3:
            self.minutes_window = sys.argv[4]
        else:
            print './feature_extractor -rrc -peer -time_bin'
            sys.exit()
        self.bin_size = 60*int(self.minutes_window)
        self.window_size = 60
        self.count_ts = defaultdict(int)
        self.plens = defaultdict(int)

        self.volume_attr_init()
        self.as_path_attr_init()
        self.rib_attr_init()
        self.class_traffic = defaultdict(int)

    def init_rib(self, file, peer_arg):
        if isfile(file + '-' + peer_arg + '-lookup.pkl'):
	    print 'Loading ' + file + '-lookup.pkl'
            self.prefix_lookup = pickle.load(open(file + '-' + peer_arg +'-lookup.pkl', "rb"))
        else:
            d = Reader(file)

            prfx_count = defaultdict(int)
            peer_count = defaultdict(int)
            print file
            for m in d:
                m = m.mrt
                if m.err == MRT_ERR_C['MRT Header Error']:
                    continue

                if is_table_dump_v1(m):
                    peer = m.td.peer_as
                    prefix = str(m.td.prefix) + '/' + str(m.td.plen)

                    self.print_classification(m, 'RIB', prefix)
                    self.prfx_set[peer][prefix] += 1

                    for attr in m.td.attr:
                        self.prefix_lookup[peer][prefix][BGP_ATTR_T[attr.type]] = attr
                elif is_table_dump_v2(m):
                    if m.peer is not None:
                        peer_index = dict()
                        i = 0
                        for p in m.peer.entry:
                            peer_index[i] = p.asn
                            i += 1
                    elif m.rib is not None:
                        prefix = str(m.rib.prefix) + '/' + str(m.rib.plen)
                        for e in m.rib.entry:
                            peer = peer_index[e.peer_index]
                            if peer == peer_arg:
                                for attr in e.attr:
                                    self.prefix_lookup[peer][prefix][BGP_ATTR_T[attr.type]] = attr

            pickle.dump(self.prefix_lookup, open(file + '-' + peer_arg + '-lookup.pkl', "wb"))

    def increment_update_counters(self, m):
        self.count_updates += 1
        self.updates[self.bin] += 1
        self.peer_upds[m.bgp.peer_as] += 1

    def update_time_bin(self, m):
        self.bin = (m.ts - self.first_ts)/self.bin_size

    def add_updates(self, file, peer):
        d = Reader(file)
        m = d.next()
        self.count_updates = 0
        if self.first_ts == 0:
            self.first_ts = m.mrt.ts
        while m:
            m = m.mrt
            if m.err == MRT_ERR_C['MRT Header Error']:
                prerror(m)
                continue

            if is_bgp_update(m):
                if m.bgp.peer_as == peer or peer == '0':
                    self.update_time_bin(m)
                    #Total number of annoucements/withdrawals/updates
                    self.increment_update_counters(m)
                    if m.bgp.msg.nlri is not None:
                        self.classify_announcement(m)
                    self.classify_withdrawal(m)
                    self.count_origin_attr(m)
            try:
                m = next(d, None)
            except:
                break

    def classify_announcement(self, m):
        for nlri in m.bgp.msg.nlri:
            self.announcements[self.bin] += 1
            prefix = nlri.prefix + '/' + str(nlri.plen)
            self.plens[nlri.plen] += 1
            self.upds_prefixes[self.bin][prefix] += 1
            self.nlri_ann[self.bin] = len(self.upds_prefixes[self.bin].keys())
            self.dbg_prefix(m, prefix)

            #Store history
            # self.prefix_history[m.bgp.peer_as][prefix].append(m)
            self.msg_counter[m.bgp.peer_as + '@' + prefix] += 1

            if self.prefix_lookup[m.bgp.peer_as].has_key(prefix) and not self.prefix_withdrawals[m.bgp.peer_as][prefix]:
                self.classify_reannouncement(m, prefix)
            else:
                self.classify_new_announcement(m, prefix)

    def classify_withdrawal(self, m):
        if (m.bgp.msg.wd_len > 0):
            if m.bgp.msg.withdrawn is not None:
                for nlri in m.bgp.msg.withdrawn:
                    self.withdrawals[self.bin] += 1
                    prefix = nlri.prefix + '/' + str(nlri.plen)
                    wd_state = self.prefix_withdrawals[m.bgp.peer_as][prefix]
                    if wd_state != 0 and wd_state == True:
                        self.wd_dups[self.bin] += 1
                    self.prefix_withdrawals[m.bgp.peer_as][prefix] = True
                    self.print_classification(m, 'WITHDRAW', prefix)
                    # self.prefix_history[m.bgp.peer_as][prefix].append(m)
                    self.msg_counter[m.bgp.peer_as + '@' + prefix] += 1

    def dbg_prefix(self, m, prefix):
        if self.prfx_set[m.bgp.peer_as][prefix] == 1:
            self.rib_count += 1
            self.prfx_set[m.bgp.peer_as][prefix] = 0
            self.prefix_found = prefix
            self.peer_found = m.bgp.peer_as

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
                new_path = new_attr.as_path[0]
                curr_path = current_attr['AS_PATH'].as_path[0]
                self.classify_as_path(m, new_attr, prefix)
                self.calc_edit_distance(m, new_path['val'], curr_path['val'], prefix)

                if new_path['len'] > curr_path['len']:
                    self.ann_to_longer[self.bin] += 1
                    # print 'self.ann_to_longer -> ' + str(self.ann_to_longer[self.bin])
                elif new_path['len'] < curr_path['len']:
                    self.ann_to_shorter[self.bin] += 1
                    # print 'self.ann_to_shorter -> ' + str(self.ann_to_shorter[self.bin])

            #Check if there is different attributes
            if not self.is_equal(new_attr, current_attr):
                is_implicit_wd = True
                if attr_name == 'AS_PATH':
                    is_implicit_dpath = True
            self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[new_attr.type]] = new_attr


        #Figure it out which counter will be incremented
        if is_implicit_wd:
            # self.prefix_imp.add(prefix)
            self.imp_wd[self.bin] += 1

            if is_implicit_dpath:
                self.implicit_withdrawals_dpath[self.bin] += 1
                self.print_classification(m, 'IMPLICIT_DIFF_PATH', prefix)
            else:
                self.implicit_withdrawals_spath[self.bin] += 1
                self.print_classification(m, 'IMPLICIT_SAME_PATH', prefix)
        else:
            self.dup_announcements[self.bin] += 1
            self.print_classification(m, 'DUPLICATE', prefix)

    def classify_new_ann_simple(self, m, prefix):
        self.new_announcements[self.bin] += 1
        if prefix == self.prefix_found and m.bgp.peer_as == self.peer_found:
            print self.prefix_lookup

        for attr in m.bgp.msg.attr:
            self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[attr.type]] = attr
            #Classify AS PATH
            if BGP_ATTR_T[attr.type] == 'AS_PATH':
                self.classify_as_path(m, attr, prefix)
        self.print_classification(m, 'NEW ANNOUNCEMENT', prefix)

    def classify_new_ann_after_wd(self, m, prefix):
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
                self.calc_edit_distance(m, new_attr.as_path[0]['val'], current_attr['AS_PATH'].as_path[0]['val'], prefix)

                #Announcement to longer or shorter path?
                new_path = new_attr.as_path[0]
                curr_path = current_attr['AS_PATH'].as_path[0]
                if new_path['len'] > curr_path['len']:
                    self.ann_to_longer[self.bin] += 1
                elif new_path['len'] < curr_path['len']:
                    self.ann_to_shorter[self.bin] += 1
            #Update with new attr
            self.prefix_lookup[m.bgp.peer_as][prefix][attr_name] = new_attr

        #Figure it out which attribute counter will be incremented
        if is_diff_announcement:
            self.new_ann_after_wd[self.bin] += 1
            # self.prefix_nada.add(prefix)
            self.print_classification(m, 'NEW ANN. AFTER WITHDRAW', prefix)
        else:
            self.flap_announcements[self.bin] += 1
            # self.prefix_flap.add(prefix)
            self.print_classification(m, 'FLAP', prefix)

    def classify_new_ann_after_wd_unknown(self, m, prefix):
        self.ann_after_wd_unknown[self.bin] += 1
        for attr in m.bgp.msg.attr:
            self.prefix_lookup[m.bgp.peer_as][prefix][BGP_ATTR_T[attr.type]] = attr
            attr_name = BGP_ATTR_T[attr.type]

            if attr_name == 'AS_PATH':
                self.classify_as_path(m, attr, prefix)
        self.print_classification(m, 'ANN. AFTER WITHDRAW - UNKNOWN', prefix)

    def classify_new_announcement(self, m, prefix):
        if not self.prefix_withdrawals[m.bgp.peer_as][prefix]:
            self.classify_new_ann_simple(m, prefix)
        elif self.prefix_lookup[m.bgp.peer_as][prefix]['ORIGIN'] != []:
            self.classify_new_ann_after_wd(m, prefix)
        else:
            self.classify_new_ann_after_wd_unknown(m, prefix)

    def classify_as_path(self, m, attr, prefix):
        for as_path in attr.as_path:
            if as_path['type'] == 2:
                unique_as_path = set(as_path['val'])

                self.as_paths.append(as_path)
                self.unique_as_paths.append(unique_as_path) #Ignore prepending
                self.distinct_as_paths.add(str(as_path['val'])) #Ignore repeated AS paths
                rare_ases = 0
                for asn in unique_as_path:
                    self.as_paths_distribution[asn] += 1
                    if self.as_paths_distribution[asn] < self.rare_threshold:
                        rare_ases += 1
                        # print 'rare_ases ->' + str(rare_ases)

                #Periodically recalculates threshold
                self.rare_ases_iteration += 1
                if self.rare_ases_iteration % 1000 == 0:
                    self.rare_threshold = np.percentile(np.array(self.as_paths_distribution.values()), 20)
                    # print 'self.rare_threshold ->' + str(self.rare_threshold)

                #Just consider AS-paths above the initial threshold
                if self.rare_ases_iteration > 1000:
                    if type(self.rare_ases_counter[self.bin]) != int:
                        self.rare_ases_counter[self.bin] = np.append(self.rare_ases_counter[self.bin], rare_ases)
                    else:
                        self.rare_ases_counter[self.bin] = np.array(rare_ases)

                    if rare_ases > self.rare_ases_max[self.bin]:
                        self.rare_ases_max[self.bin] = rare_ases

                    # if type(self.rare_ases_counter[self.bin]) != int:
                    self.rare_ases_avg[self.bin] = self.rare_ases_counter[self.bin].mean()
                        # print self.rare_ases_avg[self.bin]

                    #Rare ASes per time bin
                    self.number_rare_ases[self.bin] += rare_ases

                if as_path['len'] > self.as_path_max_length[self.bin]:
                    self.as_path_max_length[self.bin] = as_path['len']
                if len(unique_as_path) > self.unique_as_path_max[self.bin]:
                    self.unique_as_path_max[self.bin] = len(unique_as_path)


                self.num_of_paths_rcvd[self.bin] += 1
                self.as_path_avg_length[self.bin] = (as_path['len'] * self.num_of_paths_rcvd[self.bin] + self.as_path_avg_length[self.bin])/self.num_of_paths_rcvd[self.bin]
                self.unique_as_path_avg[self.bin] = (len(unique_as_path) * self.num_of_paths_rcvd[self.bin] + self.unique_as_path_max[self.bin])/self.num_of_paths_rcvd[self.bin]

    def calc_edit_distance(self, m, new_path, old_path, prefix):
        dist = edit_distance(new_path, old_path)
        dist_unique = edit_distance(list(set(new_path)),list(set(new_path)))

        self.edit_distance_dict[dist][self.bin] += 1
        self.edit_distance_unique_dict[dist_unique][self.bin] += 1

        if dist > self.edit_distance_max[self.bin]:
            self.edit_distance_max[self.bin] = dist

        if type(self.edit_distance_counter[self.bin]) != int:
            self.edit_distance_counter[self.bin] = np.append(self.edit_distance_counter[self.bin], dist)
            # print 'self.edit_distance_counter -> '+ str( self.edit_distance_counter[self.bin].mean())
        else:
            self.edit_distance_counter[self.bin] = np.array(dist)

        if type(self.edit_distance_counter[self.bin]) != int:
            self.edit_distance_avg[self.bin] = self.edit_distance_counter[self.bin].mean()


    def count_origin_attr(self, m):
        if m.bgp.msg.attr is not None:
            for attr in m.bgp.msg.attr:
                if BGP_ATTR_T[attr.type] == 'ORIGIN':
                    self.count_origin[attr.origin][self.bin] += 1


    def is_equal(self, new_attr, old_attr):
        if BGP_ATTR_T[new_attr.type] == 'ORIGIN':
            if new_attr.origin <> old_attr['ORIGIN'].origin:
                self.diff_counter['ORIGIN'] += 1
                self.count_origin_changes[self.bin] += 1
                # print 'self.count_origin_changes ->' + str(self.count_origin_changes[self.bin])
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
            if old_attr['MULTI_EXIT_DISC'] != [] and new_attr.med <> old_attr['MULTI_EXIT_DISC'].med:
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

    def print_peers(self):
        print self.prefix_lookup.keys()

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
        # print output
        # os.system('xviewer ' + output + ' &')

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

    def get_features(self):
        feat = Features()
        self.fill_blanks_timeseries()
        feat.withdrawals = self.withdrawals
        feat.wd_dups = self.wd_dups
        feat.nlri_ann = self.nlri_ann
        feat.imp_wd_spath = self.implicit_withdrawals_spath
        feat.imp_wd_dpath = self.implicit_withdrawals_dpath
        feat.announcements = self.announcements
        feat.news = self.new_announcements
        feat.dups = self.dup_announcements
        feat.nadas = self.new_ann_after_wd
        feat.flaps = self.flap_announcements
        feat.origin = self.count_origin
        feat.origin_changes = self.count_origin_changes
        feat.as_path_max = self.as_path_max_length
        feat.as_path_avg = self.as_path_avg_length
        feat.unique_as_path_max = self.unique_as_path_max
        feat.unique_as_path_avg = self.unique_as_path_avg
        feat.rare_ases_max = self.rare_ases_max
        feat.rare_ases_avg = self.rare_ases_avg
        feat.number_rare_ases = self.number_rare_ases
        feat.edit_distance_max = self.edit_distance_max
        feat.edit_distance_avg = self.edit_distance_avg
        feat.edit_distance_dict = self.edit_distance_dict
        feat.edit_distance_unique_dict = self.edit_distance_unique_dict
        feat.ann_to_shorter = self.ann_to_shorter
        feat.ann_to_longer = self.ann_to_longer
        feat.imp_wd = self.imp_wd
        feat.timestamp = dict(zip(self.announcements.keys(), [dt.datetime.fromtimestamp(ts*self.bin_size + self.first_ts) for ts in self.announcements.keys()]))
        feat.timestamp2 = dict(zip(self.announcements.keys(), [(ts*self.bin_size + self.first_ts) for ts in self.announcements.keys()]))
        feat.class_traffic = self.class_traffic
        return feat

    def fill_blanks_timeseries(self):
        #Filling blanks in the timeseries
        for i in range(self.updates.keys()[-1] + 1):
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
            self.count_origin_changes[i]
            self.as_path_max_length[i]
            self.as_path_avg_length[i]
            self.unique_as_path_max[i]
            self.unique_as_path_avg[i]
            self.nlri_ann[i]
            self.wd_dups[i]
            self.rare_ases_max[i]
            self.rare_ases_avg[i]
            self.number_rare_ases[i]
            self.rare_ases_counter[i]
            self.imp_wd[i]
            self.edit_distance_max[i]
            self.edit_distance_avg[i]
            self.edit_distance_counter[i]
            self.edit_distance_unique_counter[i]

            for dist, counter in self.edit_distance_dict.iteritems():
                counter[i]
            for dist, counter in self.edit_distance_unique_dict.iteritems():
                counter[i]
            for origin, counter in self.count_origin.iteritems():
                counter[i]

            self.ann_to_shorter[i]
            self.ann_to_longer[i]
            self.class_traffic[i]

    def plot_ts(self, dict_ts, name_dict):
        fig = plt.figure(1)
        plt.subplot(1,1,1)
        plt.plot(range(len(dict_ts.keys())), dict_ts.values(), lw=1.15, color = 'black')
        output = name_dict + str(random.randint(1, 1000)) + '.png'
        fig.savefig(output, bboxes_inches = '30', dpi = 400)
        plt.gcf().clear()
        # print output
        # os.system('xviewer ' + output + ' &')

    def plot(self):
        for feat_name, feat in features_dict.iteritems():
        #     # print feat_name + ' -> ' + str(len(feat))
        # print len(feat)
        #     # print feat
            self.plot_ts(feat, feat_name)

        # for bin, prefix_count in self.upds_prefixes.iteritems():
        #     self.max_prefix[bin] = np.array(self.upds_prefixes[bin].values()).max()
        #     self.mean_prefix[bin] = np.array(self.upds_prefixes[bin].values()).mean()
        #
        # self.sort_timeseries()
        # self.fill_blanks_timeseries()
        # self.plot_timeseries()
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
