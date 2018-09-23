import os
import glob
import time
import numpy as np

times = []

for i in xrange(51):
  start = time.time()
  cmd = 'python bgpstream-count-elems.py'
  os.system(cmd)
  times.append(np.round(time.time() - start, 3))

print '--> Python'
print 'mean: ' + str(np.array(times).mean())
print 'min: ' + str(np.array(times).min())
print 'max: ' + str(np.array(times).max())
print 'std:' + str(np.array(times).std())
print 'array: ' + str(np.round(times, 3))
print ''

times = []

for i in xrange(51):
  start = time.time()
  cmd = './bgp'
  os.system(cmd)
  times.append(np.round(time.time() - start, 3))

print '--> C'
print 'mean: ' + str(np.array(times).mean())
print 'min: ' + str(np.array(times).min())
print 'max: ' + str(np.array(times).max())
print 'std:' + str(np.array(times).std())
print 'array: ' + str(np.round(times, 3))
