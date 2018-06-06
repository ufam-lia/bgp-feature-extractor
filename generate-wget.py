import sys
import os

rrc = sys.argv[1]
path = 'ripe-ris/' + rrc
os.system('mkdir ' + path)
os.system('cd ' + path)
file = open(path + '/wget-ripe-' + rrc + '.sh','w')

for col in xrange(5):
    for year in xrange(2001,2004):
        for month in xrange(1, 12):
            file.write('wget -m -e robots=off --cut-dirs=3 --user-agent=Mozilla/5.0 --reject="index.html*" --no-parent --recursive --relative --level=1 --no-directories  http://data.ris.ripe.net/' + rrc + '/' + str(year) + '.' +str(month).zfill(2) + '/\n')
file.close()
os.system('sudo chmod +x wget-ripe-' + rrc + '.sh')
