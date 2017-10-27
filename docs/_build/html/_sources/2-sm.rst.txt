.. highlight:: rst

Bottom-Up: Synthetic Populations (Spatial Microsimulation)
==========================================================



Simple example of a simulation model:

- Define model parameters.
- We define a formula for the Electricity model. This model will compute the
  electricity demand based on previously computed income levels.
- We define a python dictionary to tell the fuction `run_calibrated_model` how
  to calibrate it. The order to the models (i.e. dictionary keys) matter, as
  the model will calibrate them in the specified order. In this case we need to
  calibrate the income model first in order to calibrate the electricity model
  because the computation of electricity directly depends on the estimation of
  income.
- We run the model with the defined parameters. The model with iterate until
  all models are calibrated.

.. literalinclude:: ./_static/code/GREGWT.py
   :language: python

.. include:: ./_static/tables/test_inc.rst

.. include:: ./_static/tables/test_elec.rst

.. figure:: ./_static/images/error_total.png
   :align: center
   :scale: 100%

   Simulation error

.. figure:: ./_static/images/Income.png
   :align: center
   :scale: 100%

   Estimated income distribution

.. figure:: ./_static/images/Electricity.png
   :align: center
   :scale: 100%

   Estimated electricity distribution



.. include:: 2.0-population.rst
