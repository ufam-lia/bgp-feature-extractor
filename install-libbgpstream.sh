sudo apt-get install libbz2-dev zlibc libcurl4-gnutls-dev sqlite3 zlib1g-dev libbz2-dev 

mkdir ~/src
cd ~/src/
curl -O https://research.wand.net.nz/software/wandio/wandio-1.0.4.tar.gz
tar zxf wandio-1.0.4.tar.gz
cd wandio-1.0.4/
./configure
make
sudo make install

cd ~/src/
curl -O http://bgpstream.caida.org/bundles/caidabgpstreamwebhomepage/dists/bgpstream-1.1.0.tar.gz
tar zxf bgpstream-1.1.0.tar.gz
cd bgpstream-1.1.0
./configure
make
sudo make install
