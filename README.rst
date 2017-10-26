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

If you want to run the model locally please use the docker image.

Using the Docker image
----------------------

Currently the docker images are build automatically with each new github push.
This means that you need to be careful with the version you are using, make
sure that the main python library code is passing all test on `travis <https://travis-ci.org/emunozh/um>`_.

The model has been developed to run on a server, and therefor developed to run
on linux, below you will find setup instruction for installing and running the
model on a linux server and an *untested* instruction on how to do in on
a windows machine.

Linux
~~~~~

.. code:: bash

   # Install docker if you don't have it via your favorite pakage manager
   sudo pacman -Ss docker

   # Make sure docker is runing
   systemctl status docker
   # Satrt docker service
   systemctl start docker

   # Pull docker image
   # Be patient the image is big
   docker pull emunoz/urbanmetabolism

   # Run the model
   docker run -it -p 8080:8080 emunozh/urbanmetabolism

   # Lunch jupyterhub within docker container
   jupyterhub --ip=0.0.0.0

You can build the docker image locally for testing and debugging.

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

   # Lunch jupyterhub
   jupyterhub --ip=0.0.0.0

Windows
~~~~~~~

.. caution::
  UNTESTED

The installation process should be simple:

1. Install `Docker for Windows <https://www.docker.com/docker-windows>`_

2. Pull the docker image :code:`docker pull emunoz/urbanmetabolism`

3. Run the docker container :code:`docker run -it -p 8080:8080 emunozh/urbanmetabolis`

4. Lunch the Jupyterhub server :code:`jupyterhub --ip=0.0.0.0`

5. Open your browser at `<http://0.0.0.0:8080>`_
