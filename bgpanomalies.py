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

base_path = '/home/pc/ripe-ris/'
anomalies = dict()
anomalies['code-red'] = ['20010717','20010718','20010719', '20010720', '20010721']
anomalies['nimda'] = ['20010916', '20010917', '20010918', '20010919', '20010920', '20010921', '20010922']
anomalies['slammer'] = ['20030123', '20030124', '20030125', '20030126', '20030127']
anomalies['moscow-blackout'] = ['20050523', '20050524', '20050525', '20050526', '20050527']

class BGPAnomaly(object):
    def __init__(self, event_name, rrc):
        self.event = event_name
        self.rrc = rrc
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
            #AS13237
            # self.start = 1116996009
            # self.end = 1117006209

    def get_files(self):
        days = []
        for day in anomalies[self.event]:
            files_retrieved = sorted(glob.glob(base_path + self.event + '/' + self.rrc + '/updates.'+ day + '.*.gz'))
            if len(files_retrieved) > 0:
                days.append(files_retrieved)
            else:
                print '#####day ' + self.event + '/' + self.rrc + '/updates.'+ day  + ' not found'

        return days
