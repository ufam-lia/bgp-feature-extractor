# BGP Feature Extractor

**BGP Feature Extractor** extracts relevant features from BGP control plane messages along with tools that assist the labelling of the anomaly period. Our tool extracts volume and AS path features most commonly used by anomaly detection techniques, as well as novel distribution features that allow the observation of BGP traffic changes in a straightforward manner.

## Installing
The requirements for this tool might be downloaded using:
```
$ pip install -r requirements.txt
```

## Using

```
usage: feature-extractor.py [-h] -c COLLECTOR -p PEER -a ANOMALY -t TIMESTEPS
                            [-r]
```

Process BGP timeseries

optional arguments:
  - c COLLECTOR, --collector COLLECTOR                        Name of the collector
  - p PEER, --peer PEER  Peer considered (0, if all peer must be considered)
  - a ANOMALY, --anomaly ANOMALY
                        Anomaly event name
  - t TIMESTEPS, --timesteps TIMESTEPS
                        Timestep window that must be considered
  - r, --rib             Disable RIB initialization

### Folder structure
Currently, our feature extractor requires that the MRT are downloaded before extraction. These files must be placed in a folder structured following the pattern <base_path>/<event_name>/\<collector>/\<dump files>. The dump files might be .bz2 or .gz file extension.

## Generate datasets
After the timeseries are generated labelling is performed using the ```label_csv.py``` by through the command: 
```
usage: label_csv.py [collector] [peer] 
```
Peer and timestamps of anomaly start and end must be added to the script.


## Features

Detailed information about features can be found on the [wiki](https://github.com/ufam-lia/bgp-feature-extractor/wiki/Dataset-Features).
