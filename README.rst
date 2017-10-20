Spatial Microsimulation Urban Metabolism Model
==============================================

.. image:: https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg
    :target: https://cloud.docker.com/app/emunozh/repository/docker/emunozh/urbanmetabolism/general

.. image:: https://travis-ci.org/emunozh/um.svg
    :target: http://travis-ci.org/emunozh/um

This is the main repository for the developing of the Spatial Microsimulation
Urban Metabolism (SMUM) model.

The model is currently under development.

Check back for an update on deployment.

If you want to play around with the code please install the docker image.

Using the Docker image
----------------------

Currently the docker images are build automatically with each new github push.

.. code:: bash

   # Make suredocker is runing
   systemctl status docker
   # Satrt docker service
   systemctl start docker

   # Pull docker image
   # Be patient the image is big
   docker pull emunoz/urbanmetabolism

   # Run the model
   docker run -it -p 8080:8080 emunozh/urbanmetabolism /bin/bash ./run.sh

This is the actual content of
`run.sh`.

.. code:: bash

   #!/bin/sh
   # run.sh

   # Initiate jupyter notebook
   jupyterhub --ip=0.0.0.0\
     -f ./jupyterhub_config.py\
     ./urbanmetabolism/examples/Welcome.ipynb

You can build the docher image localy for testing and debuging.

.. code:: bash

   # Build the docker image on your computer
   # Clone github repository
   git clone git@github.com:emunozh/um.git
   # Move to um folder
   cd um
   # Build docker image
   docker build . -t um_test
   # Run docker image
   docker run -it um_test


