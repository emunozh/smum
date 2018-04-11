.. urbanmetabolism_doc Masterfile

===============================================
Spatial Microsimulation Urban Metabolism (SMUM)
===============================================

.. only:: latex

  |logo|

.. only:: html

  |UNEP| |space| |GIREC|

:Author: Dr. M. Esteban Munoz H. <emunozh@gmail.com> or <esteban.munoz@un.org>
:Version: 0.2.0
:Date: |date|

.. note::

  This documentation is generated automatically from the main `github
  repository <www.github.com/emunozh/em>`_. The build of the documentation is not
  always successful and you might end up reading a documentation of an outdated
  version of the model. Please always verify the version of the model you are
  running. This simulation library is under development and we are constantly
  changing the simulation libraries.

This is the main documentation for the Spatial Microsimulation Urban Metabolism
model. This model combines two powerful approaches for the simulation of
resource flows within cities. The first approach is Urban Metabolism (UM).
This approach describes the metabolic performance of cities by quantifying and
balancing all resource inputs and outputs from a predefined city-system. The
second component of the simulation model is the Spatial Microsimulation (SM)
model. This component of the simulation library constructs a synthetic
population for the specific city-system and allocates consumption values to the
individual city agents. The simulation library benchmarks this synthetic sample
to the aggregated consumption values from the UM model.

The aim of this documentation is twofold:

  1. Describe the methodological approach of the simulation model; and
  2. Explain how to use the components of the library and present some simulation examples.

This simulation library is build on top of some well know python libraries as
well as some specific python an R libraries.

+----------------------+-----------------------------------------+
| **Python libraries** |  **Use**                                |
+----------------------+-----------------------------------------+
| pandas               | Data library for python                 |
+----------------------+-----------------------------------------+
| numpy                | Numerical model in python               |
+----------------------+-----------------------------------------+
| scipy                | Scientific python                       |
+----------------------+-----------------------------------------+
| statsmodels          | Statistical models                      |
+----------------------+-----------------------------------------+
| Theano               | Compiled numerical computation library. |
+----------------------+-----------------------------------------+
| jupyterhub           | Used as main UI                         |
+----------------------+-----------------------------------------+
| matplotlib           | De facto python plotting library        |
+----------------------+-----------------------------------------+
| seaborn              | Statistical plots                       |
+----------------------+-----------------------------------------+
| pymc3                | Bayesian Statistics, MCMC               |
+----------------------+-----------------------------------------+
| ipfn                 | Iterative Proportional Fitting          |
+----------------------+-----------------------------------------+
| XlsxWriter           | Create excel files                      |
+----------------------+-----------------------------------------+

+-----------------+----------------------------+
| **R libraries** |  **Use**                   |
+-----------------+----------------------------+
| GREGWT          | Sample reweighting library |
+-----------------+----------------------------+

.. toctree::
   :maxdepth: 1

   0-intro.rst
   1-um.rst
   2-sm.rst
   3-examples.rst
   4-api.rst
   x-authors
   x-contributing
   x-history

.. |GIREC| image:: ./_static/images/GI-REC.png
    :alt: GI-REC
    :scale: 42%
    :target: http://www.resourceefficientcities.org

.. |UNEP| image:: ./_static/images/UNEnvironment.png
    :alt: UN Environment
    :scale: 20%
    :target: http://www.unep.org

.. |space| image:: ./_static/images/na.png
    :alt: _
    :width: 1cm

.. |logo| image:: ./_static/images/UNEnvironment2.png
    :alt: UNEP-GIREC
    :width: 12cm

.. |date| date::
