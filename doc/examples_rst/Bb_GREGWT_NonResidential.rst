
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

(2.b) Non-Residential Model
===========================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-25 14:42:26.286853


**Notebook abstract**

This notebook shows the main sampling and reweighting algorithm for the
non-residential sector.

Import libraries
----------------

.. code:: ipython3

    from urbanmetabolism.population.model import run_calibrated_model
    from urbanmetabolism.population.model import plot_data_projection
    from urbanmetabolism.population.model import TableModel

Global variables
----------------

.. code:: ipython3

    iterations = 1000
    benchmark_year = 2016
    census_file = 'data/benchmarks_nonresidential.csv'
    typ = 'resampled'
    model_name = 'Sorsogon_NonResidentialElectricity_wbias_projected_dynamic_{}'.format(typ)
    verbose = True
    drop_col_survey = ['n_BuildingKwh']

Define model
------------

.. code:: ipython3

    table_model_name = 'data/table_elec_nr.csv'
    estimate_var = 'NonRElectricity'
    tm = TableModel(census_file = census_file, verbose=verbose)
    tm.add_model(table_model_name, estimate_var)
    tm.update_dynamic_model(estimate_var, specific_col = 'BuildingKwh')


.. parsed-literal::

    --> census cols:  Index(['n_BuildingSqm', 'n_BuildingKwh', 'pop', 'NonRElectricity'], dtype='object')
    adding NonRElectricity model as dynamic model.
    	| for all columns:
    	| for specific column BuildingKwh
    	| specific col: 	[]




.. parsed-literal::

    False



.. code:: ipython3

    tm.models[estimate_var]




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>co_mu</th>
          <th>co_sd</th>
          <th>p</th>
          <th>dis</th>
          <th>lb</th>
          <th>ub</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>n_Building_sqm_cat</th>
          <td>719.587128022,312.594751517,1165.99458393,703....</td>
          <td>45.9315047501,27.8888172966,510.591052778,206....</td>
          <td>0.0608498641378,0.091606063041,0.0106072914917...</td>
          <td>Deterministic;n;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>n_Building_kwh_cat</th>
          <td>262,631,592,316,293,233,296,137,243</td>
          <td>135.810529783,649.550998768,344.818792991,124....</td>
          <td>0.0608498641378,0.091606063041,0.0106072914917...</td>
          <td>Deterministic;Building_sqm_cat;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    tm.models[estimate_var]




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>co_mu</th>
          <th>co_sd</th>
          <th>p</th>
          <th>dis</th>
          <th>lb</th>
          <th>ub</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>n_Building_sqm_cat</th>
          <td>719.587128022,312.594751517,1165.99458393,703....</td>
          <td>45.9315047501,27.8888172966,510.591052778,206....</td>
          <td>0.0608498641378,0.091606063041,0.0106072914917...</td>
          <td>Deterministic;n;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>n_Building_kwh_cat</th>
          <td>262,631,592,316,293,233,296,137,243</td>
          <td>135.810529783,649.550998768,344.818792991,124....</td>
          <td>0.0608498641378,0.091606063041,0.0106072914917...</td>
          <td>Deterministic;Building_sqm_cat;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    #formula_nrb = "c_{} * c_{}".format(*tm.models[estimate_var].loc[2010].index)
    #tm.add_formula(formula_nrb, estimate_var)
    #table_model = tm.make_model()
    #tm.to_excel()

.. code:: ipython3

    formula_nrb

Run model
---------

.. code:: ipython3

    fw = run_calibrated_model(
        table_model,
        #verbose = True,
        project = typ,
        census_file = census_file,
        year = benchmark_year,
        population_size = False,
        name = '{}_{}'.format(model_name, iterations),
        iterations = iterations,
        #drop_col_survey = drop_col_survey
    )

Plot results
------------

.. code:: ipython3

    reweighted_survey = 'data/survey_{}_{}'.format(model_name, iterations)

.. code:: ipython3

    data = plot_data_projection(
        reweighted_survey, [estimate_var], "{}, {}".format(iterations, typ),
        benchmark_year=year, unit = "building")

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

(2.b) Non-Residential Model
===========================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__

`Next <Bc_GREGWT_validation_wbias.ipynb>`__ (2.c) Model Internal
Validation
