
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

(2.c) Model Internal Validation
===============================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-26 15:57:50.784378


**Notebook abstract**

Model internal validation.

.. code:: ipython3

    from urbanmetabolism.population.model import plot_error

.. code:: ipython3

    iterations = 1000#00
    benchmark_year = 2016
    census_file = 'data/benchmarks_year_bias.csv'
    typ = 'resampled'
    model_name = 'Sorsogon_Electricity_Water_wbias_projected_dynamic_{}'.format(typ)
    survey_file = 'data/survey_{}_{}_{}.csv'.format(model_name, iterations, benchmark_year)

.. code:: ipython3

    REC = plot_error(
        survey_file,
        census_file,
        "{}, {}".format(iterations, typ),
        year = benchmark_year)



.. image:: FIGURES_rst/Bc_GREGWT_validation_wbias_5_0.png


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

(2.c) Model Internal Validation
===============================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__

`Next <Ca_DefineTransitions.ipynb>`__ (3.a) Define Transition Scenarions
