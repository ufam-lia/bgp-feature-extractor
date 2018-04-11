#!/usr/bin/env python
from _pybgpstream import BGPStream, BGPRecord, BGPElem
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import random

## NOTE CODE RED 1
# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()
l = []
c = 0

# Consider RIPE RRC 04 only
stream.add_filter('collector','rrc03')
stream.add_filter('collector','rrc00')
stream.add_filter('record-type','updates')

#17-07-2001 to 21-07-2001
# stream.add_interval_filter(995328000,995760000)
stream.add_interval_filter(995328000,995330880)

# Start the stream
stream.start()

count_updates_rrc00 = 0
count_updates_rrc03 = 0
timeseries_rrc00 = np.array(0)
timeseries_rrc03 = np.array(0)
last_ts = 0

# Get next record
while(stream.get_next_record(rec)):

    if rec.status == "valid":
        if last_ts != rec.dump_time:
            print rec.collector
            print rec.dump_time
            if rec.collector == 'rrc00':
                timeseries_rrc00 = np.append(timeseries_rrc00, count_updates_rrc00)
                count_updates_rrc00 = 0

            if rec.collector == 'rrc03':
                timeseries_rrc03 = np.append(timeseries_rrc03, count_updates_rrc03)
                count_updates_rrc03 = 0
            last_ts = rec.dump_time

        elem = rec.get_next_elem()
        while(elem):
            if rec.collector == 'rrc00':
                if elem.type == 'A':
                    count_updates_rrc00 += 1
                    # if elem.peer_asn == 513:
                    #         count_updates_as513 += 1

            if rec.collector == 'rrc03':
                if elem.type == 'A':
                    count_updates_rrc03 += 1
                    # if elem.peer_asn == 513:
                    #         count_updates_as513 += 1
            elem = rec.get_next_elem()

fig = plt.figure(1)
plt.subplot(1,2,1)
plt.plot(range(len(timeseries_rrc00)), timeseries_rrc00, lw=1, color = 'teal')
plt.plot(range(len(timeseries_rrc03)), timeseries_rrc03, lw=1, linestyle = '--', color = 'orange')
plt.show()
## NOTE NIMDA WORM
# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()
l = []
c = 0

# Consider RIPE RRC 04 only
stream.add_filter('collector','rrc03')
stream.add_filter('collector','rrc00')
stream.add_filter('record-type','updates')

#16-09-2001 to 24-09-2001
#stream.add_interval_filter(1000598400,1001376000)
stream.add_interval_filter(1000598400,1000609400)

# Start the stream
stream.start()

count_updates_rrc00 = 0
count_updates_rrc03 = 0
timeseries_rrc00 = np.array(0)
timeseries_rrc03 = np.array(0)
last_ts = 0

# Get next record
while(stream.get_next_record(rec)):
    if rec.status == "valid":
        if last_ts != rec.dump_time:
            timeseries_rrc00 = np.append(timeseries_rrc00, count_updates_rrc00)
            timeseries_rrc03 = np.append(timeseries_rrc03, count_updates_rrc03)
            count_updates_rrc00 = 0
            count_updates_rrc03 = 0
            last_ts = rec.dump_time

        elem = rec.get_next_elem()
        while(elem):
            if rec.collector == 'rrc00':
                if elem.type == 'A':
                    count_updates_rrc00 += 1
                    # if elem.peer_asn == 513:
                    #         count_updates_as513 += 1

            if rec.collector == 'rrc03':
                if elem.type == 'A':
                    count_updates_rrc03 += 1
                    # if elem.peer_asn == 513:
                    #         count_updates_as513 += 1
            elem = rec.get_next_elem()

plt.subplot(1,2,2)
plt.plot(range(len(timeseries_rrc00)), timeseries_rrc00, lw=1, color = 'teal')
plt.plot(range(len(timeseries_rrc03)), timeseries_rrc03, lw=1, linestyle = '--', color = 'orange')

# fig.savefig(str(random.randint(1, 1000)),bboxes_inches = 'tight',dpi=700)
plt.show()
