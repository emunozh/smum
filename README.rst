Spatial Microsimulation Urban Metabolism Model (SMUM)
=====================================================

|docker| |travis| |docs|

.. image:: ./docs/_static/images/um_logo.png
   :scale: 100 %
   :alt: SMUM
   :align: center

This is the main repository for the developing of the Spatial Microsimulation
Urban Metabolism (SMUM) model.

.. image:: ./docs/_static/images/UNEnvironment2.png
   :width: 5pt
   :scale: 5 %
   :alt: UNEP
   :align: center

The model is currently under development.

Check back for an update on its deployment.

If you want to run the model locally please use the docker image.

Install
-------

To install the python library only via pip.

.. code-block:: bash

  pip install -e git+https://github.com/emunozh/um.git#egg=urbanmetabolism

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

.. code-block:: bash

   # Install docker if you don't have it via your favorite pakage manager
   sudo pacman -Ss docker

   # Make sure docker is runing
   systemctl status docker
   # Start docker service
   systemctl start docker

   # Pull docker image
   # Be patient the image is big
   docker pull emunozh/urbanmetabolism

   # Run the model
   docker run -it -p 8080:8080 emunozh/urbanmetabolism

   # Lunch jupyterhub within docker container
   jupyterhub --ip=0.0.0.0

You can build the docker image locally for testing and debugging.

.. code-block:: bash

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
  NOT TESTED!

The installation process should be simple:

1. Install `Docker for Windows <https://www.docker.com/docker-windows>`_

2. Pull the docker image :code:`docker pull emunozh/urbanmetabolism`

3. Run the docker container :code:`docker run -it -p 8080:8080 emunozh/urbanmetabolism`

4. Lunch the Jupyterhub server :code:`jupyterhub --ip=0.0.0.0`

5. Open your browser at `<http://0.0.0.0:8080>`_

Contribute
----------

- Issue Tracker: github.com/emunozh/um/issues
- Source Code: github.com/emunozh/um

Support
-------

If you are having issues, please let us know.

License
-------

The project is licensed under the GPL-3.0 license.

.. |docker| image:: https://img.shields.io/docker/automated/jrottenberg/ffmpeg.svg
    :alt: Docker cloud
    :scale: 100%
    :target: https://cloud.docker.com/app/emunozh/repository/docker/emunozh/urbanmetabolism/general

.. |travis| image:: https://travis-ci.org/emunozh/um.svg
    :alt: build status
    :scale: 100%
    :target: http://travis-ci.org/emunozh/um

.. |docs| image:: https://readthedocs.org/projects/smum/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://smum.readthedocs.io/en/latest/?badge=latest
