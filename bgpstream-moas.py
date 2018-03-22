#!/usr/bin/env python

from _pybgpstream import BGPStream, BGPRecord, BGPElem
from collections import defaultdict


# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()
c = 0
# Consider Route Views Singapore only
stream.add_filter('collector','route-views.sg')

# Consider RIBs dumps only
stream.add_filter('record-type','ribs')

# Consider this time interval:
# Sat, 01 Aug 2015 7:50:00 GMT -  08:10:00 GMT
stream.add_interval_filter(1438415400,1438416600)

# Start the stream
print 'stream.start()'
stream.start()
print 'stream.started'
# <prefix, origin-ASns-set > dictionary
prefix_origin = defaultdict(set)

# Get next record
while(stream.get_next_record(rec)):
    elem = rec.get_next_elem()
    while(elem):
        # Get the prefix
        pfx = elem.fields['prefix']
        # Get the list of ASes in the AS path
        ases = elem.fields['as-path'].split(" ")
        if len(ases) > 0:
            # Get the origin ASn (rightmost)
            origin = ases[-1]
            # Insert the origin ASn in the set of
            # origins for the prefix
            prefix_origin[pfx].add(origin)
        elem = rec.get_next_elem()
    c +=  1
    print c

# Print the list of MOAS prefix and their origin ASns
for pfx in prefix_origin:
    if len(prefix_origin[pfx]) > 1:
        print pfx, ",".join(prefix_origin[pfx])
