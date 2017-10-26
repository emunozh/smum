
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

3.b. Visualize transition scenarios
===================================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-26 16:09:43.262808


**Notebook Abstract:**

The following notebook visualze the the simple transition scenarios.

Import libraries
----------------

.. code:: ipython3

    from urbanmetabolism.population.model import plot_data_projection

The visualization is performed with help of the module function
``plot_data_projection()``.

.. code:: ipython3

    iterations = 1000
    typ = 'resampled'
    model_name = 'Sorsogon_Electricity_Water_wbias_projected_dynamic_{}'.format(typ)
    reweighted_survey = 'data/survey_{}_{}'.format(model_name, iterations)

Base scenario
-------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    data = plot_data_projection(
        reweighted_survey, var, "{}, {}".format(iterations, typ),
        benchmark_year=2016 
    )

Base scenario grouped by Urban-Rural households
-----------------------------------------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    groupby = 'i_Urbanity'
    data = plot_data_projection(
        reweighted_survey, var, "{}, {} by {}".format(iterations, typ, groupby),
        benchmark_year = 2016,
        groupby = groupby
    )

Base scenario grouped by family size
------------------------------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    groupby = 'i_Family_Size'
    data = plot_data_projection(
        reweighted_survey, var, "{}, {} by {}".format(iterations, typ, groupby),
        benchmark_year = 2016,
        groupby = groupby
    )

Base scenario grouped by AC ownership
-------------------------------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    groupby = 'e_AC'
    data = plot_data_projection(
        reweighted_survey, var, "{}, {} by {}".format(iterations, typ, groupby),
        benchmark_year = 2016,
        verbose = False,
        groupby = groupby
    )

.. code:: ipython3

    import numpy as np
    pr = [i for i in np.linspace(0.1, 0.6, num=15)]
    pr = [0]*6 + pr
    scenario_name = 'scenario 1'

Scenario 1 compared to base scenario
------------------------------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    data = plot_data_projection(
        reweighted_survey, var, "{}, {}, alt. scenario 1".format(iterations, typ),
        benchmark_year=2016, pr = pr, scenario_name = scenario_name
    )

Scenario 1 grouped by education
-------------------------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    groupby = 'i_Education'
    data = plot_data_projection(
        reweighted_survey, var, "{}, {} by {}, alt. scenario 1".format(iterations, typ, groupby),
        benchmark_year = 2016, pr = pr, scenario_name = scenario_name,
        groupby = groupby
    )

.. code:: ipython3

    import numpy as np
    pr = [i for i in np.linspace(0.1, 0.8, num=15)]
    pr = [0]*6 + pr
    scenario_name = 'scenario 2'

Scenario 2 compared to base scenario
------------------------------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    data = plot_data_projection(
        reweighted_survey, var, "{}, {}, alt. scenario 2".format(iterations, typ),
        benchmark_year=2016, pr = pr, scenario_name = scenario_name
    )

Scenario 2 grouped by education
-------------------------------

.. code:: ipython3

    var = ['Income', 'Water', 'Electricity']
    groupby = 'i_Education'
    data = plot_data_projection(
        reweighted_survey, var, "{}, {} by {}, alt. scenario 2".format(iterations, typ, groupby),
        benchmark_year = 2016, pr = pr, scenario_name = scenario_name,
        groupby = groupby
    )

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

3.b. Visualize transition scenarios
===================================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__
