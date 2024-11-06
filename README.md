# Guugli-map

simple google map clone. Vehicle Routing Problem With Time Windows Solver (using Genetic Algorithm) & shortest path (see https://github.com/lintang-b-s/navigatorx) from scratch.

## Quick Start

```
- git submodule update --init
- docker compose up -d --build
- wait until  container navigatorx up & running
- docker logs navigatorx -f  (wait until contraction hierarchies processing 100% complete)
- python3 leaflet-vrptw/manage.py runserver
- open localhost:8000/home
- do shortest path query (only for yogyakarta, surakarta, klaten maps)
-  or do vehicle routing problem with time windows query

Note: Try to make sure the location where the query is carried out is close to a road, so that road snapping works.
```

## Hasil parameter tuning
```
https://drive.google.com/drive/folders/1J6QutrRxUFfsqJMZiO0kMgfk7khSrC6z?usp=sharing
```