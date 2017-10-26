
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

(2.a) Dynamic Sampling Model and GREGWT
=======================================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-26 10:06:09.825152


**Notebook abstract**

This notebook shows the main sampling and reweighting algorithm.

Import libraries
----------------

.. code:: ipython3

    from urbanmetabolism.population.model import run_calibrated_model, _project_survey_resample
    from urbanmetabolism.population.model import TableModel

Global variables
----------------

.. code:: ipython3

    iterations = 1000
    benchmark_year = 2016
    census_file = 'data/benchmarks_year_bias.csv'
    typ = 'resampled'
    model_name = 'Sorsogon_Electricity_Water_wbias_projected_dynamic_{}'.format(typ)
    verbose = False
    #The number of chains to run in parallel. 
    njobs = 1

Define Table model
------------------

.. code:: ipython3

    tm = TableModel(census_file = census_file, verbose=verbose)

Income model
~~~~~~~~~~~~

.. code:: ipython3

    tm.add_model('data/table_inc.csv',   'Income')
    tm.update_dynamic_model('Income', specific_col = 'Education')
    tm.update_dynamic_model('Income',
                            specific_col = 'FamilySize',
                            specific_col_as = 'Size',
                            val = 'mu', compute_average =  0)
    tm.update_dynamic_model('Income',
                            specific_col = 'Age',
                            val = 'mu', compute_average =  0)

.. code:: ipython3

    tm.models['Income'].loc[2020]




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
          <th>mu</th>
          <th>sd</th>
          <th>dis</th>
          <th>ub</th>
          <th>lb</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>i_Intercept</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>1147.66</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Deterministic</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>i_Sex</th>
          <td>919.012059036333</td>
          <td>161.50344091572538</td>
          <td>0.243795</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>i_Urbanity</th>
          <td>7105.2244566329355</td>
          <td>127.94148635675795</td>
          <td>0.6356</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>i_FamilySize</th>
          <td>1666.846395220964</td>
          <td>29.03482607534048</td>
          <td>NaN</td>
          <td>3.70878</td>
          <td>1.83794</td>
          <td>Poisson</td>
          <td>10</td>
          <td>1</td>
        </tr>
        <tr>
          <th>i_Age</th>
          <td>116.57589770606201</td>
          <td>4.681393204635</td>
          <td>NaN</td>
          <td>52.5153</td>
          <td>12.2451</td>
          <td>Normal</td>
          <td>100</td>
          <td>18</td>
        </tr>
        <tr>
          <th>i_Education</th>
          <td>1.0,6023.86254599,11959.091528,18727.4606703,1...</td>
          <td>1e-10,140.904404522,217.208790314,282.17614554...</td>
          <td>0.243037974684,0.21581625995,0.255409108704,0....</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



Electricity model
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    tm.add_model('data/table_elec.csv',  'Electricity', reference_cat = ['yes'])
    tm.update_dynamic_model('Electricity', specific_col = 'Income', val = 'mu', compute_average = False)

.. code:: ipython3

    tm.models['Electricity'].loc[2016]




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
          <th>mu</th>
          <th>sd</th>
          <th>dis</th>
          <th>ub</th>
          <th>lb</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>e_Intercept</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>3.29998</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Deterministic</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Lighting</th>
          <td>0.825662</td>
          <td>18.6676</td>
          <td>0.946022</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_TV</th>
          <td>18.7899</td>
          <td>1.75962</td>
          <td>0.964932</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Cooking</th>
          <td>28.8862</td>
          <td>1.96894</td>
          <td>0.0142662</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Refrigeration</th>
          <td>59.2432</td>
          <td>1.55605</td>
          <td>0.602102</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_AC</th>
          <td>203.323</td>
          <td>3.13016</td>
          <td>0.256521</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Urban</th>
          <td>24.5935</td>
          <td>1.39104</td>
          <td>1</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Income</th>
          <td>0.00142607</td>
          <td>4.10201e-05</td>
          <td>NaN</td>
          <td>190472</td>
          <td>1904.72</td>
          <td>None</td>
          <td>inf</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    formula_elec = "e_Intercept+"+"+".join(
        ["c_{0} * {0}".format(e) for e in tm.models['Electricity'][benchmark_year].index if\
            (e != 'e_Intercept') &\
            (e != 'e_Income') &\
            (e != 'e_Urban')
        ])
    formula_elec += '+c_e_Urban * i_Urbanity'
    formula_elec += '+c_e_{0} * {0}'.format('Income')

.. code:: ipython3

    tm.add_formula(formula_elec, 'Electricity')

.. code:: ipython3

    tm.print_formula('Electricity')


.. parsed-literal::

    Electricity =
    	 e_Intercept +
    	 c_e_Lighting * e_Lighting +
    	 c_e_TV * e_TV +
    	 c_e_Cooking * e_Cooking +
    	 c_e_Refrigeration * e_Refrigeration +
    	 c_e_AC * e_AC +
    	 c_e_Urban * i_Urban +
    	 c_e_Income * Income +


Water model
~~~~~~~~~~~

.. code:: ipython3

    tm.add_model('data/table_water.csv', 'Water')
    tm.update_dynamic_model('Water', specific_col = 'Education')
    tm.update_dynamic_model('Water',
                            specific_col = 'FamilySize',
                            specific_col_as = 'Size',
                            val = 'mu', compute_average =  0)
    tm.update_dynamic_model('Water',
                            specific_col = 'Age',
                            val = 'mu', compute_average =  0)

.. code:: ipython3

    tm.models['Water'].loc[2020]




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
          <th>mu</th>
          <th>sd</th>
          <th>ub</th>
          <th>lb</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>w_Intercept</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>-601.592</td>
          <td>Deterministic</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Sex</th>
          <td>98.49504620801835</td>
          <td>29.44380722589748</td>
          <td>0.243795</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Urbanity</th>
          <td>1000.9789077676428</td>
          <td>25.415910606032206</td>
          <td>0.6356</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Total_Family_Income</th>
          <td>0.05318701200857999</td>
          <td>0.0009823058551951082</td>
          <td>NaN</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_FamilySize</th>
          <td>49.73935151831777</td>
          <td>5.897790558149098</td>
          <td>NaN</td>
          <td>None</td>
          <td>3.70878</td>
          <td>1.83794</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Age</th>
          <td>6.088941881654669</td>
          <td>0.9127405886772298</td>
          <td>NaN</td>
          <td>None</td>
          <td>52.5153</td>
          <td>12.2451</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Education</th>
          <td>1.0,214.401145313,260.327274277,101.70283943,4...</td>
          <td>1e-10,28.8158024405,40.0574490885,49.995759305...</td>
          <td>0.243037974684,0.21581625995,0.255409108704,0....</td>
          <td>None;i;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    formula_water = "w_Intercept+"+"+".join(
        ["c_{0} * {1}".format(e, "i_"+"_".join(e.split('_')[1:]))\
             for e in tm.models['Water'][benchmark_year].index if \
                                     (e != 'w_Intercept') &\
                                     (e != 'w_Total_Family_Income')   &\
                                     (e != 'w_Education')
        ])
    formula_water += '+c_w_Total_Family_Income*Income'
    formula_water += '+c_w_Education*i_Education'

.. code:: ipython3

    tm.add_formula(formula_water, 'Water')

.. code:: ipython3

    tm.print_formula('Water')


.. parsed-literal::

    Water =
    	 w_Intercept +
    	 c_w_Sex * i_Sex +
    	 c_w_Urbanity * i_Urbanity +
    	 c_w_FamilySize * i_FamilySize +
    	 c_w_Age * i_Age +
    	 c_w_Total_Family_Income*Income +
    	 c_w_Education*i_Education +


Make model and save it to excel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    table_model = tm.make_model()

.. code:: ipython3

    tm.to_excel()


.. parsed-literal::

    creating data/tableModel_Income.xlsx
    creating data/tableModel_Electricity.xlsx
    creating data/tableModel_Water.xlsx


Define model variables
----------------------

.. code:: ipython3

    labels = ['age_0_18', 'age_19_25', 'age_26_35',
              'age_36_45', 'age_46_55', 'age_56_65',
              'age_66_75', 'age_76_85', 'age_86_100']
    cut = [0, 19, 26, 36, 46, 56, 66, 76, 86, 101]
    to_cat = {'i_Age':[cut, labels]}
    drop_col_survey = ['e_Income', 'e_Urban', 'w_Total_Family_Income', 'w_Education']

.. code:: ipython3

    fw = run_calibrated_model(
        table_model,
        project = typ,
        njobs = njobs,
        #rep = {'FamilySize': ['Size']},
        #rep={'urb': ['urban', 'urbanity']},
        census_file = census_file,
        year = benchmark_year,
        population_size = False,
        name = '{}_{}'.format(model_name, iterations),
        to_cat = to_cat,
        iterations = iterations,
        verbose = verbose,
        drop_col_survey = drop_col_survey)


.. parsed-literal::

    loop: 1/4; calibrating: Income; sufix = loop_1
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:42<00:00,  6.22s/it]


.. parsed-literal::

    loop: 2/4; calibrating: Electricity; sufix = loop_2
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:44<00:00,  6.60s/it]


.. parsed-literal::

    loop: 3/4; calibrating: Water; sufix = loop_3
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:42<00:00,  6.35s/it]


.. parsed-literal::

    loop: 4/4; final loop, for variables: Income, Electricity, Water; sufix = loop_4
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:42<00:00,  8.53s/it]


.. parsed-literal::

    Calibration Error:
    	9.1434E-04  Income
    	-3.3974E-05  Electricity
    	9.8670E-01  Water
    Projecting sample survey for 21 steps via resample
    resampling for year 2010
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:41<00:00,  6.22s/it]


.. parsed-literal::

    resampling for year 2011
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:39<00:00,  5.87s/it]


.. parsed-literal::

    resampling for year 2012
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:39<00:00,  5.89s/it]


.. parsed-literal::

    resampling for year 2013
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:39<00:00,  5.92s/it]


.. parsed-literal::

    resampling for year 2014
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:39<00:00,  5.91s/it]


.. parsed-literal::

    resampling for year 2015
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:39<00:00,  5.95s/it]


.. parsed-literal::

    resampling for year 2016
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:39<00:00,  5.83s/it]


.. parsed-literal::

    resampling for year 2017
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:06<00:00,  1.49it/s]


.. parsed-literal::

    resampling for year 2018
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:06<00:00,  1.48it/s]


.. parsed-literal::

    resampling for year 2019
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:06<00:00,  1.47it/s]


.. parsed-literal::

    resampling for year 2020
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 12.31it/s]


.. parsed-literal::

    resampling for year 2021
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 12.11it/s]


.. parsed-literal::

    resampling for year 2022
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 13.23it/s]


.. parsed-literal::

    resampling for year 2023
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 12.71it/s]


.. parsed-literal::

    resampling for year 2024
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 13.10it/s]


.. parsed-literal::

    resampling for year 2025
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 13.52it/s]


.. parsed-literal::

    resampling for year 2026
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 13.57it/s]


.. parsed-literal::

    resampling for year 2027
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 12.17it/s]


.. parsed-literal::

    resampling for year 2028
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 12.99it/s]


.. parsed-literal::

    resampling for year 2029
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 13.16it/s]


.. parsed-literal::

    resampling for year 2030
    Warning: will overwrite total population column on census


.. parsed-literal::

    100%|██████████| 11/11 [00:00<00:00, 12.88it/s]


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

2.a Micro-level Electricity demand model
========================================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__

`Next <Bb_GREGWT_NonResidential.ipynb>`__ (2.b) Non-Residential Model
