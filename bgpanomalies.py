
import sys
import os, time
import os.path
from os.path import isfile
from optparse import OptionParser
import datetime as dt
from datetime import datetime
from mrtparse import *
from collections import defaultdict, OrderedDict
import glob
import operator
import pandas as pd
from operator import add

anomalies = dict()
anomalies['code-red'] = ['20010717','20010718','20010719', '20010720', '20010721']
anomalies['nimda'] = ['20010915', '20010916', '20010917', '20010918', '20010919', '20010920', '20010921', '20010922']
anomalies['slammer'] = ['20030123', '20030124', '20030125', '20030126', '20030127']
anomalies['moscow-blackout'] = ['20050523', '20050524', '20050525', '20050526', '20050527']
anomalies['as9121'] = ['20041222', '20041223', '20041224', '20041225', '20041226']
anomalies['as-depeering'] = ['20051003','20051004','20051005', '20051006', '20051007', '20051008']
anomalies['as-3561-filtering'] = ['20010403','20010404','20010405','20010406','20010407','20010408']
anomalies['as-path-error'] = ['20011004','20011005','20011006','20011007','20011008','20011009']
anomalies['malaysian-telecom'] = ['20150610', '20150611', '20150612', '20150613', '20150614']
anomalies['aws-leak'] = ['20160420', '20160421', '20160422', '20160423', '20160424']
anomalies['panix-hijack'] = ['20060120', '20060121', '20060122', '20060123', '20060124']
anomalies['japan-earthquake'] = ['20110309', '20110310', '20110311', '20110312', '20110313']

class BGPAnomaly(object):
    def __init__(self, event_name, rrc, peer):
        self.base_path = '/home/pc/ripe-ris/'
        self.event = event_name
        self.rrc = rrc
        self.peer = peer
        self.start = 0
        self.end = 0
        self.set_period()
        # self.set_files()

    def set_period(self):
        if self.event == 'code_red':
            self.start = 995553071
            self.end = 995591487
        elif self.event == 'nimda':
            self.start = 1000818222
            self.end = 1001030344
        elif self.event == 'slammer':
            self.start = 1043472590
            self.end = 1043540404
        elif self.event == 'moscow':
            self.start = 1116996009
            self.end = 1117006209
            if self.peer == '13237':
                self.start = 1116996009
                self.end = 1117006209
        elif self.event == 'as9121':
            self.start = 1103916000
            self.end = 1103918580
        elif self.event == 'malaysian-telecom':
            self.start = 1434098580
            self.end = 1434109500
        elif self.event == 'as-depeering':
            self.start = 1128715200
            self.end = 1128729660

    def get_files(self):
        days = []
        for day in anomalies[self.event]:
            files_retrieved = sorted(glob.glob(self.base_path + self.event + '/' + self.rrc + '/updates.'+ day + '.*.gz'))
            files_retrieved = files_retrieved + sorted(glob.glob(self.base_path + self.event + '/' + self.rrc + '/updates.'+ day + '.*.bz2'))
            if len(files_retrieved) > 0:
                days.append(files_retrieved)
            else:
                print (self.base_path + self.event + '/' + self.rrc + '/updates.'+ day + '.*.gz')
                print ('#####day ' + self.event + '/' + self.rrc + '/updates.'+ day  + ' not found')
        return days

    def get_rib(self):
        rib = sorted(glob.glob(self.base_path + self.event + '/' + self.rrc + '/bview.*.gz'))
        rib = rib + sorted(glob.glob(self.base_path + self.event + '/' + self.rrc + '/rib.*.bz2'))
        if len(rib) > 0:
            return rib[0]
        else:
            print ('#####rib ' + self.event + '/' + self.rrc + '/bview.* not found')
            return None

class BGPDataset(object):
    def __init__(self, event_name, anomaly = False, base_path='/home/pc/bgp-feature-extractor/datasets/'):
        self.event = event_name
        self.anomaly = anomaly
        self.dataset_path = base_path
        # self.set_files()

    def get_files(self, timebin = 0, peer = '*', multi = False):
        if multi:
            dataset_prefix = '/dataset_multi_'
        else:
            dataset_prefix = '/dataset_'

        if self.anomaly:
            dataset_prefix = '/anomaly_multi_'

        if type(timebin) == list:
            files = []
            for bin in timebin:
                bin = str(bin)
                unique_files = glob.glob(self.dataset_path + dataset_prefix + self.event + '_'+ peer +'_'+ bin +'_*.csv')
                files += [f for f in unique_files if 'rand' not in f]
        else:
            timebin = str(timebin)
            files = glob.glob(self.dataset_path + dataset_prefix + self.event + '_'+ peer +'_'+ timebin +'_*.csv')

        return sorted(files)
