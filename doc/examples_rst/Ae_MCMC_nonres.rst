
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

1.e Micro-level Non-Residential model
=====================================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-25 14:40:43.353497


**Notebook abstract**

A simple micro-level building stock model.

Compile building level data
---------------------------

.. code:: ipython3

    from urbanmetabolism._scripts.sqm_data import get_pop_data, get_sqm_data

.. code:: ipython3

    census_file = 'data/benchmarks_year_bias.csv'

.. code:: ipython3

    pop_data_1 = get_pop_data(census_file, sqm_nonres_mean = 800)
    nr_data = get_sqm_data()
    pop_data = get_pop_data(
        census_file,
        sqm_nonres_mean = nr_data.loc[:, 'sqm'].mean(),
    )
    pop_data_3 = get_pop_data(census_file, sqm_nonres_mean = 500)

.. code:: ipython3

    benchmarks = pop_data.loc[:,['sqm_nonres', 'num_nonres']]
    benchmarks.columns = ['n_BuildingSqm', 'pop']

.. code:: ipython3

    kwh_2016 = 6.52 * 1000000

.. code:: ipython3

    pb_kwh = kwh_2016 / benchmarks.loc[2016, 'n_BuildingSqm']
    pb_kwh_sqm = benchmarks.loc[:, 'pop'].mul(pb_kwh)
    benchmarks.insert(1, 'n_BuildingKwh', pb_kwh_sqm)

.. code:: ipython3

    benchmarks.loc[2016, 'NonRElectricity'] = kwh_2016

.. code:: ipython3

    benchmarks




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
          <th>n_BuildingSqm</th>
          <th>n_BuildingKwh</th>
          <th>pop</th>
          <th>NonRElectricity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2010</th>
          <td>68716.843636</td>
          <td>9989.175844</td>
          <td>111.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2011</th>
          <td>69223.535896</td>
          <td>10079.168419</td>
          <td>112.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2012</th>
          <td>69772.846840</td>
          <td>10169.160994</td>
          <td>113.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2013</th>
          <td>70361.528759</td>
          <td>10259.153569</td>
          <td>114.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2014</th>
          <td>71001.761591</td>
          <td>10349.146145</td>
          <td>115.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2015</th>
          <td>71699.539394</td>
          <td>10439.138720</td>
          <td>116.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2016</th>
          <td>72450.421467</td>
          <td>10529.131295</td>
          <td>117.0</td>
          <td>6520000.0</td>
        </tr>
        <tr>
          <th>2017</th>
          <td>73262.865014</td>
          <td>10619.123870</td>
          <td>118.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2018</th>
          <td>74192.780824</td>
          <td>10799.109020</td>
          <td>120.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2019</th>
          <td>75235.646063</td>
          <td>10889.101596</td>
          <td>121.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2020</th>
          <td>76389.214220</td>
          <td>11069.086746</td>
          <td>123.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2021</th>
          <td>77687.299788</td>
          <td>11249.071896</td>
          <td>125.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2022</th>
          <td>79167.682437</td>
          <td>11519.049622</td>
          <td>128.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2023</th>
          <td>80801.910589</td>
          <td>11699.034772</td>
          <td>130.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2024</th>
          <td>82649.717674</td>
          <td>11969.012498</td>
          <td>133.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2025</th>
          <td>84680.858538</td>
          <td>12328.982798</td>
          <td>137.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2026</th>
          <td>86949.276099</td>
          <td>12598.960524</td>
          <td>140.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2027</th>
          <td>89480.862266</td>
          <td>12958.930825</td>
          <td>144.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2028</th>
          <td>92254.166615</td>
          <td>13408.893700</td>
          <td>149.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2029</th>
          <td>95305.598723</td>
          <td>13858.856576</td>
          <td>154.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2030</th>
          <td>98638.177901</td>
          <td>14308.819452</td>
          <td>159.0</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    benchmarks.to_csv('data/benchmarks_nonresidential.csv')

.. code:: ipython3

    from urbanmetabolism._scripts.sqm_data import plot_nr
    plot_nr(pop_data_1, pop_data, pop_data_3, nr_data)



.. image:: FIGURES_rst/Ae_MCMC_nonres_13_0.png


Prior non-residential model
---------------------------

.. code:: ipython3

    from urbanmetabolism._scripts.sqm_data import get_count_data
    import pandas as pd
    count_data = get_count_data()
    count_data = count_data.div(count_data.sum())

.. code:: ipython3

    nrb_elec = pd.DataFrame(columns=['co_mu', 'co_sd', 'p', 'dis', 'lb', 'ub'])
    
    nrb_elec.loc['BuildingSqm', 'co_mu'] = ",".join([str(i) for i in nr_data['sqm']])
    nrb_elec.loc['BuildingSqm', 'co_sd'] = ",".join([str(i) for i in nr_data['sqm_sd']])
    
    nrb_elec.loc['BuildingKwh', 'co_mu'] = ",".join([str(i) for i in nr_data['kwh']])
    nrb_elec.loc['BuildingKwh', 'co_sd'] = ",".join([str(i) for i in nr_data['kwh_sd']])
    
    nrb_elec.loc[:, 'p'] = ",".join([str(i) for i in count_data['counts']])
    nrb_elec.loc['BuildingSqm', 'dis'] = "Deterministic;n;Categorical"
    nrb_elec.loc['BuildingKwh', 'dis'] = "Deterministic;BuildingSqm;Categorical"

.. code:: ipython3

    nrb_elec.to_csv('data/table_elec_nr.csv')

.. code:: ipython3

    nrb_elec.loc['Building_sqm_cat', 'dis'] = "Normal;n;Categorical"
    nrb_elec.to_csv('data/test_elec_nr_normal.csv')

.. code:: ipython3

    nrb_elec




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
          <th>BuildingSqm</th>
          <td>719.587128022,312.594751517,1165.99458393,703....</td>
          <td>45.9315047501,27.8888172966,510.591052778,206....</td>
          <td>0.0608498641378,0.091606063041,0.0106072914917...</td>
          <td>Deterministic;n;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>BuildingKwh</th>
          <td>262,631,592,316,293,233,296,137,243</td>
          <td>135.810529783,649.550998768,344.818792991,124....</td>
          <td>0.0608498641378,0.091606063041,0.0106072914917...</td>
          <td>Deterministic;BuildingSqm;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>Building_sqm_cat</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Normal;n;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



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

1.e Micro-level Non-Residential model
=====================================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__

`Next <Ba_GREGWT_Dynamic.ipynb>`__ (2.a) Dynamic Sampling Model and
GREGWT
