#!/bin/sh

# initiate jupyter notebook
jupyter-notebook --ip=0.0.0.0 --port=8080 --allow-root \
  usr/src/app/urbanmetabolism/examples/start.ipynb
