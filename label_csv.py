import pandas as pd
import numpy as np
import os, glob

features_path = '/home/pc/bgp-feature-extractor/csv/'

files = sorted(glob.glob(features_path + 'features-nimda-2001*'))

df = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f, index_col=0, delimiter = ';')
    labels = []
    for ts in csv['timestamp2']:
        print ts
        if int(ts) >= 1000818000 and int(ts) <= 1001030400:
            print 1
            labels.append(1)
        else:
            print 0
            labels.append(0)
    csv['class'] = pd.Series(labels)
    print csv
    df = df.append(csv, sort = True)

df.reset_index(drop = True, inplace = True)
df.to_csv(features_path + 'nimda-rrc00.csv')
