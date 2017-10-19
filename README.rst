Docker for Urban Metabolism Model
=================================

.. image:: https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg
    :target: https://cloud.docker.com/app/emunozh/repository/docker/emunozh/urbanmetabolism/general

.. image:: https://travis-ci.org/emunozh/um.svg
    :target: http://travis-ci.org/emunozh/um

This is the main repository for the developing of the Spatial Microsimulation
Urban Metabolism (SMUM) model.

The model is currently under development.

Check back for an update on deployment.

If you want to play around with the code please install the docker image.

.. code:: bash

   # Make suredocker is runing
   systemctl status docker
   # Satrt docker service
   systemctl start docker

   # Pull docker image
   # Be patient the image is big
   docker pull emunoz/urbanmetabolism

   # Run the model
   docker run -it emunozh/urbanmetabolism /bin/bash /usr/src/app/run.sh

.. code:: bash

   # initiate jupyter notebook
   jupyter-notebook ./urbanmetabolism/examples/start.ipynb

   pip install jupyterhub


