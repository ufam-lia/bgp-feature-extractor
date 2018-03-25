#!/usr/bin/env python
from _pybgpstream import BGPStream, BGPRecord, BGPElem
import time

# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()
l = []
c = 0

# Consider RIPE RRC 10 only
stream.add_filter('collector','rrc06')
stream.add_filter('collector','route-views.jinx')
stream.add_filter('record-type','updates')

# Consider this time interval:
# Sat Aug  1 08:20:11 UTC 2015
stream.add_interval_filter(1286705410,1286709071)

# Start the stream
stream.start()

# Get next record
while(stream.get_next_record(rec)):
# Print the record information only if it is not a valid record
    if rec.status == "valid":
        elem = rec.get_next_elem()
        while(elem):
            c += 1
            elem = rec.get_next_elem()

# print 'Read ' + str(c) + ' elems'
