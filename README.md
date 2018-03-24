# BGP feature extractor

The first step is to investigate available datasets and looking glasses.
Found so far:
## Data sources
  - [ ] RouteViews
  - [ ] BGPMon
    - [ ] BGPStream
    - [ ] Web Service API
    - [ ] Event detection
  - [ ] RIPE/RCC
    - [ ] Atlas
    - [ ] [RIPEstat Data API](https://stat.ripe.net/docs/data_api)
  - [ ] GEANT
  - [ ] Abilene/Internet2
  - [ ] CAIDA
    - [ ]  Ark
  - [ ] iplane
  - [ ] PeeringDB

### BGPStream
Notes:

1. Two interfaces
   1. `libBGPStream`, a C API with core functions
   2. `PyBGPStream`,  Python bindings to the libBGPStream C API. 
2. Necessary to install some packages before
3. *Erro*: Python não encontrava a lib `libbgpstream.so.2` . <br>Solução: Buscar a lib no sistema e adicionar o caminho para ela no `.bashrc` com o comando<br>`export LD_LIBRARY_PATH="/usr/local/lib"`
