#!/usr/bin/env python
from _pybgpstream import BGPStream, BGPRecord, BGPElem

# Create a new bgpstream instance and a reusable bgprecord instance
stream = BGPStream()
rec = BGPRecord()

# Consider RIPE RRC 10 only
stream.add_filter('collector','rrc11')

# Consider this time interval:
# Sat Aug  1 08:20:11 UTC 2015
stream.add_interval_filter(1438417216,1438417216)

# Start the stream
stream.start()

# Get next record
while(stream.get_next_record(rec)):
# Print the record information only if it is not a valid record
    if rec.status != "valid":
      print rec.project, rec.collector, rec.type, rec.time, rec.status
    else:
      elem = rec.get_next_elem()
      while(elem):
        # Print record and elem information
        print rec.project, rec.collector, rec.type, rec.time, rec.status,
        print elem.type, elem.peer_address, elem.peer_asn, elem.fields
        elem = rec.get_next_elem()
