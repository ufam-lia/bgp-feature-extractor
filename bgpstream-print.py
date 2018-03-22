#!/usr/bin/env python
from _pybgpstream import BGPStream, BGPRecord, BGPElem
import time

# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()
l = []
c = 0

# Consider RIPE RRC 10 only
stream.add_filter('collector','rrc11')

# Consider this time interval:
# Sat Aug  1 08:20:11 UTC 2015
stream.add_interval_filter(1438417216,1438418736)

# Start the stream
stream.start()

# Get next record
while(stream.get_next_record(rec)):
# Print the record information only if it is not a valid record
    if rec.status != "valid":
        print 'project: ' + str(rec.project)
        print 'collector: ' + str(rec.collector)
        print 'type: ' + str(rec.type)
        print 'time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rec.time)))
        print 'status: ' + str(rec.status)
        print
    else:
        elem = rec.get_next_elem()
        while(elem):
            # Print record and elem information
            # print 'project: ' + str(rec.project)
            # print 'collector: ' + str(rec.collector)
            # print 'type: ' + str(rec.type)
            # print 'time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rec.time)))
            # print 'status: ' + str(rec.status)
            # print 'type: ' + str(elem.type)
            # print 'peer_address: ' + str(elem.peer_address)
            # print 'peer_asn: ' + str(elem.peer_asn)
            # print 'fields: ' + str(elem.fields)
            # print

            l.append(str(elem.type))
            c += 1
            elem = rec.get_next_elem()
print set(l)
print c
