# BGP Feature Extractor

**BGP Feature Extractor** extracts relevant features from BGP control plane messages along with tools that assist the labelling of the anomaly period. Our tool extracts volume and AS path features most commonly used by anomaly detection techniques, as well as novel distribution features that allow the observation of BGP traffic changes in a straightforward manner.

## Installing
The requirements for this tool might be downloaded using:
```
$ pip install -r requirements.txt
```

##Using

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
Currently, our feature extractor requires that the MRT are downloaded before extraction. By default  

## Generate datasets

## Customize labelling

## Adding new events

## Adding new features
