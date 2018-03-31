# BGP feature extractor

The first step is to investigate available datasets and looking glasses.
Found so far:
## Data sources
  - [x] RouteViews
  - [x] BGPMon
    - [x] BGPStream
    - [x] Web Service API
    - [x] Event detection
  - [x] RIPE/RCC
    - [x] Atlas
    - [x] [xIPEstat Data API](https://stat.ripe.net/docs/data_api)
  - [x] GEANT
  - [x] Abilene/Internet2
  - [x] CAIDA
    - [x]  Ark
  - [x] iplane
  - [x] PeeringDB

### BGPStream
Notes:

1. Two interfaces
   1. `libBGPStream`, a C API with core functions
   2. `PyBGPStream`,  Python bindings to the libBGPStream C API.
2. Necessary to install some packages before
3. *Erro*: Python não encontrava a lib `libbgpstream.so.2` . <br>Solução: Buscar a lib no sistema e adicionar o caminho para ela no `.bashrc` com o comando<br>`export LD_LIBRARY_PATH="/usr/local/lib"`
4. *API*: When a filter is set, the `BGPRecord` will traverse all record and its corresponding `BGPElem` will be `None` if the filter doesn't match to that particular record.
