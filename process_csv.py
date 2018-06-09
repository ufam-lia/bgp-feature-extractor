import pandas as pd
import os, glob

features_path = '/home/pc/bgp-feature-extractor/csv/'

files = sorted(glob.glob(features_path + 'features-2001*'))
df = pd.DataFrame()

for f in files:
    csv = pd.read_csv(f, index_col=0)
    df = df.append(csv, sort = True)

# df = df.drop(0, 1)
# df = df.drop(1, 1)
df.reset_index(drop = True, inplace = True)
# df.drop(['timestamp', 'class'], 1, inplace = True)
print df
