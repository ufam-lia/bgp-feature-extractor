import pandas as pd
import os, glob

features_path = '/home/pc/bgp-feature-extractor/csv/'

files = sorted(glob.glob(features_path + 'features-2001*'))
df = pd.DataFrame()

for f in files:
    csv = pd.read_csv(f, index_col=0)
    for i in xrange(11, 30):
        col = 'edit_distance_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)
        col = 'edit_distance_unique_dict_' + str(i) + '_'
        if col in csv.keys():
            csv.drop(col, 1, inplace = True)
    # print df.
    df = df.append(csv, sort = True)

# df = df.drop(0, 1)
# df = df.drop(1, 1)
df.reset_index(drop = True, inplace = True)
df.drop(['timestamp', 'class'], 1, inplace = True)
print df
