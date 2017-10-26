
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

1.d Micro-level Water demand model
==================================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-25 14:39:45.067213


**Notebook abstract**

A simple micro-level water demand model.

Prior water demand model
------------------------

.. code:: ipython3

    import statsmodels.api as sm
    import pandas as pd
    import numpy as np
    from urbanmetabolism._scripts.micro import compute_categories, change_index


.. parsed-literal::

    /usr/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools


.. code:: ipython3

    water_data = pd.read_csv('data/water.csv', index_col=0)
    formula = "Water_expenditure ~ Total_Family_Income + Family_Size + C(HH_head_Sex)\
    + HH_head_Age + C(Education) + C(Urbanity)"

.. code:: ipython3

    water_data.head()




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
          <th>Family_Size</th>
          <th>HH_head_Sex</th>
          <th>HH_head_Age</th>
          <th>Education</th>
          <th>Electricity_expenditure</th>
          <th>Water_expenditure</th>
          <th>Total_Family_Income</th>
          <th>Urbanity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>3</th>
          <td>2.0</td>
          <td>2</td>
          <td>51</td>
          <td>1.0</td>
          <td>900</td>
          <td>2190</td>
          <td>9932.333333</td>
          <td>0</td>
        </tr>
        <tr>
          <th>9</th>
          <td>3.5</td>
          <td>1</td>
          <td>41</td>
          <td>4.0</td>
          <td>17202</td>
          <td>300</td>
          <td>47233.833333</td>
          <td>0</td>
        </tr>
        <tr>
          <th>11</th>
          <td>3.0</td>
          <td>2</td>
          <td>75</td>
          <td>1.0</td>
          <td>4212</td>
          <td>3024</td>
          <td>16521.333333</td>
          <td>0</td>
        </tr>
        <tr>
          <th>12</th>
          <td>4.5</td>
          <td>1</td>
          <td>74</td>
          <td>1.0</td>
          <td>6210</td>
          <td>4086</td>
          <td>20254.333333</td>
          <td>0</td>
        </tr>
        <tr>
          <th>13</th>
          <td>6.5</td>
          <td>2</td>
          <td>55</td>
          <td>2.0</td>
          <td>3900</td>
          <td>2940</td>
          <td>16368.000000</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    model_water = sm.WLS.from_formula(formula, water_data)
    model_results_water = model_water.fit()

.. code:: ipython3

    model_results_water.summary()




.. raw:: html

    <table class="simpletable">
    <caption>WLS Regression Results</caption>
    <tr>
      <th>Dep. Variable:</th>    <td>Water_expenditure</td> <th>  R-squared:         </th>  <td>   0.344</td>  
    </tr>
    <tr>
      <th>Model:</th>                   <td>WLS</td>        <th>  Adj. R-squared:    </th>  <td>   0.344</td>  
    </tr>
    <tr>
      <th>Method:</th>             <td>Least Squares</td>   <th>  F-statistic:       </th>  <td>   925.6</td>  
    </tr>
    <tr>
      <th>Date:</th>             <td>Mon, 23 Oct 2017</td>  <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
    </tr>
    <tr>
      <th>Time:</th>                 <td>18:02:49</td>      <th>  Log-Likelihood:    </th> <td>-1.3853e+05</td>
    </tr>
    <tr>
      <th>No. Observations:</th>      <td> 15904</td>       <th>  AIC:               </th>  <td>2.771e+05</td> 
    </tr>
    <tr>
      <th>Df Residuals:</th>          <td> 15894</td>       <th>  BIC:               </th>  <td>2.771e+05</td> 
    </tr>
    <tr>
      <th>Df Model:</th>              <td>     9</td>       <th>                     </th>      <td> </td>     
    </tr>
    <tr>
      <th>Covariance Type:</th>      <td>nonrobust</td>     <th>                     </th>      <td> </td>     
    </tr>
    </table>
    <table class="simpletable">
    <tr>
               <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
    </tr>
    <tr>
      <th>Intercept</th>           <td> -601.5920</td> <td>   62.632</td> <td>   -9.605</td> <td> 0.000</td> <td> -724.358</td> <td> -478.826</td>
    </tr>
    <tr>
      <th>C(HH_head_Sex)[T.2]</th> <td>   98.4950</td> <td>   29.444</td> <td>    3.345</td> <td> 0.001</td> <td>   40.782</td> <td>  156.208</td>
    </tr>
    <tr>
      <th>C(Education)[T.2.0]</th> <td>  214.4011</td> <td>   28.816</td> <td>    7.440</td> <td> 0.000</td> <td>  157.919</td> <td>  270.883</td>
    </tr>
    <tr>
      <th>C(Education)[T.3.0]</th> <td>  260.3273</td> <td>   40.057</td> <td>    6.499</td> <td> 0.000</td> <td>  181.810</td> <td>  338.844</td>
    </tr>
    <tr>
      <th>C(Education)[T.4.0]</th> <td>  101.7028</td> <td>   49.996</td> <td>    2.034</td> <td> 0.042</td> <td>    3.705</td> <td>  199.700</td>
    </tr>
    <tr>
      <th>C(Education)[T.5.0]</th> <td>   40.1879</td> <td>  119.681</td> <td>    0.336</td> <td> 0.737</td> <td> -194.400</td> <td>  274.775</td>
    </tr>
    <tr>
      <th>C(Urbanity)[T.1]</th>    <td> 1000.9789</td> <td>   25.416</td> <td>   39.384</td> <td> 0.000</td> <td>  951.161</td> <td> 1050.797</td>
    </tr>
    <tr>
      <th>Total_Family_Income</th> <td>    0.0532</td> <td>    0.001</td> <td>   54.145</td> <td> 0.000</td> <td>    0.051</td> <td>    0.055</td>
    </tr>
    <tr>
      <th>Family_Size</th>         <td>   49.7394</td> <td>    5.898</td> <td>    8.434</td> <td> 0.000</td> <td>   38.179</td> <td>   61.300</td>
    </tr>
    <tr>
      <th>HH_head_Age</th>         <td>    6.0889</td> <td>    0.913</td> <td>    6.671</td> <td> 0.000</td> <td>    4.300</td> <td>    7.878</td>
    </tr>
    </table>
    <table class="simpletable">
    <tr>
      <th>Omnibus:</th>       <td>3226.790</td> <th>  Durbin-Watson:     </th> <td>   1.399</td>
    </tr>
    <tr>
      <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7597.976</td>
    </tr>
    <tr>
      <th>Skew:</th>           <td> 1.142</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
    </tr>
    <tr>
      <th>Kurtosis:</th>       <td> 5.499</td>  <th>  Cond. No.          </th> <td>3.10e+05</td>
    </tr>
    </table>



.. code:: ipython3

    params_water = change_index(model_results_water.params)
    bse_water = change_index(model_results_water.bse)
    water = pd.concat([params_water, bse_water], axis=1)
    water.columns = ['co_mu', 'co_sd']
    water = compute_categories(water)

.. code:: ipython3

    water.loc['Urbanity', 'p'] = (water_data.Urbanity == 1).sum() / water_data.shape[0]
    water.loc['Sex', 'p'] = (water_data.HH_head_Sex == 2).sum() / water_data.shape[0]

.. code:: ipython3

    water.loc[:,'dis'] = 'None'
    water.loc['Education', 'dis'] = 'None;i;Categorical'
    water.loc['Intercept', 'dis'] = 'Deterministic'

.. code:: ipython3

    water.loc[:, 'mu'] = np.nan
    water.loc[:, 'sd'] = np.nan
    water.loc['Intercept', 'p'] = water.loc['Intercept', 'co_mu']
    water.loc['Intercept', ['co_mu', 'co_sd']] = np.nan

.. code:: ipython3

    water.loc[:,'ub'] = np.nan
    water.loc[:,'lb'] = np.nan

.. code:: ipython3

    water.index = ['w_'+i for i in water.index]

.. code:: ipython3

    water.to_csv('data/table_water.csv')

.. code:: ipython3

    water




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
          <td>-601.591950</td>
          <td>Deterministic</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Sex</th>
          <td>98.495</td>
          <td>29.4438</td>
          <td>0.224786</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Urbanity</th>
          <td>1000.98</td>
          <td>25.4159</td>
          <td>0.593939</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Total_Family_Income</th>
          <td>0.053187</td>
          <td>0.000982306</td>
          <td>NaN</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_FamilySize</th>
          <td>49.7394</td>
          <td>5.89779</td>
          <td>NaN</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Age</th>
          <td>6.08894</td>
          <td>0.912741</td>
          <td>NaN</td>
          <td>None</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>w_Education</th>
          <td>1.0,214.401145313,260.327274277,101.70283943,4...</td>
          <td>1e-10,28.8158024405,40.0574490885,49.995759305...</td>
          <td>NaN</td>
          <td>None;i;Categorical</td>
          <td>NaN</td>
          <td>NaN</td>
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

1.d Micro-level Water demand model
==================================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__

`Next <Ae_MCMC_nonres.ipynb>`__ (1.e) Micro-level Non-Residential model
