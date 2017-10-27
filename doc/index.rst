.. highlight:: rst
.. |date| date::

.. urbanmetabolism_doc Masterfile

===============================================
Spatial Microsimulation Urban Metabolism (SMUM)
===============================================

|GIREC| |UNEP|

:Author: Dr. M. Esteban Munoz H. <emunozh@gmail.com>
:Version: 0.1
:Date: |date|

Aim of this Documentation
=========================

This documentations aims to describe the main rationale behind the development
of this python library and to present an overview of the main modules and
functions implemented on the library.

The python library is composed of two main components:

  1. An Urban Metabolism sections that aims to balance all resources flows of
     city systems at an aggregated level (i.e. city-level) and;

  2. A Spatial Microsimulation section. The Spatial Microsimulation modules
     constructs a synthetic city and allocates consumption values to
     micro-level agents.

For a complete exmple please refere to the following
`ipython notebook <http://nbviewer.jupyter.org/github/emunozh/um/blob/master/doc/examples/Welcome.ipynb>`_

.. toctree::
   :maxdepth: 3

   x-readme
   0-intro.rst
   1-um.rst
   2-sm.rst
   x-authors
   x-contributing
   x-history

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. glossary::

.. |GIREC| image:: ./_static/images/GI-REC.png
    :alt: GI-REC
    :scale: 35%
    :target: www.resourceefficientcities.org

.. |UNEP| image:: ./_static/images/UNEnvironment.png
    :alt: UN Environment
    :scale: 20%
    :target: www.unep.org
