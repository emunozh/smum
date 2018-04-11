.. _apium:

API: Top-Down (Urban Metabolism)
=========================================

.. _city:

City
------------

.. figure:: ./_static/images/classes_M_city.png
   :align: center
   :scale: 50%

   City class diagram.


.. autoclass:: urbanmet.city.City
   :members:
   :special-members:
   :private-members:

.. _materials:

Materials
------------

.. figure:: ./_static/images/classes_M_materials.png
   :align: center
   :scale: 50%

   Materials class diagram.


.. autoclass:: urbanmet.materials.MaterialsFlow
   :members:
   :special-members:
   :private-members:

.. autoclass:: urbanmet.materials.MaterialsStock
   :members:
   :special-members:
   :private-members:

.. _water:

Water
------------

.. figure:: ./_static/images/classes_M_water.png
   :align: center
   :scale: 50%

   Water class diagram.


.. autoclass:: urbanmet.water.WaterDemand
   :members:
   :special-members:
   :private-members:

.. autoclass:: urbanmet.water.WaterFlow
   :members:
   :special-members:
   :private-members:

.. autoclass:: urbanmet.water.WaterStock
   :members:
   :special-members:
   :private-members:


.. _energy:

Energy
------------

.. figure:: ./_static/images/classes_M_energy.png
   :align: center
   :scale: 50%

   Energy class diagram.


.. autoclass:: urbanmet.energy.EnergyFlow
   :members:
   :special-members:
   :private-members:

.. autoclass:: urbanmet.energy.EnergyStock
   :members:
   :special-members:
   :private-members:


.. _food:

Food
------------

.. figure:: ./_static/images/classes_M_food.png
   :align: center
   :scale: 50%

   Food class diagram.


.. autoclass:: urbanmet.food.FoodFlow
   :members:
   :special-members:
   :private-members:

.. autoclass:: urbanmet.food.FoodStock
   :members:
   :special-members:
   :private-members:


.. _waste:

Waste
------------

.. figure:: ./_static/images/classes_M_waste.png
   :align: center
   :scale: 50%

   Waste class diagram.


.. autoclass:: urbanmet.waste.WasteFlow
   :members:
   :special-members:
   :private-members:

.. autoclass:: urbanmet.waste.WasteStock
   :members:
   :special-members:
   :private-members:

.. _UML: https://en.wikipedia.org/wiki/Unified_Modeling_Language


.. _apism:

API: Bottom-Up (Spatial Microsimulation)
==========================================================

.. autofunction:: microsim.run.run_calibrated_model

.. autofunction:: microsim.run.run_composite_model

.. autofunction:: microsim.run.transition_rate

.. autofunction:: microsim.run.reduce_consumption

.. autofunction:: microsim.util_plot.cross_tab

.. autofunction:: microsim.util_plot.plot_data_projection

.. autofunction:: microsim.util_plot.plot_error

.. autofunction:: microsim.util_plot.plot_projected_weights

.. autofunction:: microsim.util_plot.plot_transition_rate

.. figure:: ./_static/images/classes_M_aggregates.png
   :align: center
   :scale: 50%

   Aggregates class diagram.

.. autoclass:: microsim.aggregates.Aggregates
   :members:


.. figure:: ./_static/images/classes_M_population.png
   :align: center
   :scale: 50%

   Population class diagram.

.. autoclass:: microsim.population.PopModel
   :members:


.. figure:: ./_static/images/classes_M_table.png
   :align: center
   :scale: 50%

   Table class diagram.

.. autoclass:: microsim.table.TableModel
   :members:
