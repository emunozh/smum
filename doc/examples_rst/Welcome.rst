
Spatial\ :math:`^{*}` Microsimulation Urban Metabolism Model (SMUM)
===================================================================

.. raw:: html

   <div class="image123">

::

    <div class="imgContainer">
        <img src="./logos/UNEnvironment.png" alt="UNEP logo" style="width:200px">
    </div>
    <div class="imgContainer">
        <img src="./logos/GI-REC.png" alt="GI_REC logo" style="width:200px">
    </div>

.. raw:: html

   </div>

`UN Environment <http://www.unep.org/>`__

:math:`^{*}`\ `Under development <https://github.com/emunozh/um>`__

Abstract:
---------

This is a simple implementation example of the developed **Spatial
Microsimulation Urban Metabolism Model (SMUM)**.

The aim of this model is to identify and quantify the impact of
transition pathways to a circular economy.

Two main method are implemented in the model, giving it it’s name:

1. A Spatial Microsimulation creates a synthetic population; and
2. An Urban Metabolism approach is used to benchmark consumption level
   at a city-level or neighborhood level (making it spatial).

(1) Constructing a synthetic population:
----------------------------------------

Aggregate level benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~

`(1.a) Projecting demographic
variables <Aa_ProjectionAggregates.ipynb>`__

Micro level consumption models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`(1.b) Micro-level Income model <Ab_MCMC_income.ipynb>`__

`(1.c) Micro-level Electricity demand
model <Ac_MCMC_electricity.ipynb>`__

`(1.d) Micro-level Water demand model <Ad_MCMC_water.ipynb>`__

`(1.e) Micro-level Non-Residential model <Ae_MCMC_nonres.ipynb>`__

(2) Sampling and reweighting
----------------------------

Dynamic samplic models
----------------------

`(2.a) Dynamic Sampling Model and GREGWT <Ba_GREGWT_Dynamic.ipynb>`__

`(2.b) Non-Residential Model <Bb_GREGWT_NonResidential.ipynb>`__

Model internal validation
-------------------------

`(2.c) Model Internal Validation <Bc_GREGWT_validation_wbias.ipynb>`__

(3) Constructing scenarions:
----------------------------

`(3.a) Define Transition Scenarions <Ca_DefineTransitions.ipynb>`__

`(3.b) Visualize transition scenarios <Cb_VisualizeTransitions.ipynb>`__
