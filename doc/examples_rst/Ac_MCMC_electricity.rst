
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

1.c Micro-level Electricity demand model
========================================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-25 14:39:00.570024


**Notebook Abstract:**

A simple micro-level electricity deman model.

Prior electricity demand model
------------------------------

.. code:: ipython3

    import statsmodels.api as sm
    import pandas as pd
    import numpy as np
    from urbanmetabolism._scripts.micro import compute_categories, change_index

.. code:: ipython3

    electricity_data = pd.read_csv('data/electricity.csv', index_col=0)
    formula = "Electricity ~ C(Lighting) + C(TV) + C(Cooking) + C(Refrigeration) + C(AC) + C(Urban) + Income"

.. code:: ipython3

    electricity_data.head()




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
          <th>Lighting</th>
          <th>TV</th>
          <th>Cooking</th>
          <th>Refrigeration</th>
          <th>AC</th>
          <th>Urban</th>
          <th>Income</th>
          <th>Electricity</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>16000.0</td>
          <td>110.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>4000.0</td>
          <td>80.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>6000.0</td>
          <td>47.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>6300.0</td>
          <td>17.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1</td>
          <td>1</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>5000.0</td>
          <td>17.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    model_elec = sm.WLS.from_formula(formula, electricity_data)
    model_results_elec = model_elec.fit()

.. code:: ipython3

    model_results_elec.summary()




.. raw:: html

    <table class="simpletable">
    <caption>WLS Regression Results</caption>
    <tr>
      <th>Dep. Variable:</th>       <td>Electricity</td>   <th>  R-squared:         </th> <td>   0.516</td> 
    </tr>
    <tr>
      <th>Model:</th>                   <td>WLS</td>       <th>  Adj. R-squared:    </th> <td>   0.516</td> 
    </tr>
    <tr>
      <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2519.</td> 
    </tr>
    <tr>
      <th>Date:</th>             <td>Tue, 24 Oct 2017</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
    </tr>
    <tr>
      <th>Time:</th>                 <td>12:05:46</td>     <th>  Log-Likelihood:    </th> <td> -96932.</td> 
    </tr>
    <tr>
      <th>No. Observations:</th>      <td> 16522</td>      <th>  AIC:               </th> <td>1.939e+05</td>
    </tr>
    <tr>
      <th>Df Residuals:</th>          <td> 16514</td>      <th>  BIC:               </th> <td>1.939e+05</td>
    </tr>
    <tr>
      <th>Df Model:</th>              <td>     7</td>      <th>                     </th>     <td> </td>    
    </tr>
    <tr>
      <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
    </tr>
    </table>
    <table class="simpletable">
    <tr>
                <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
    </tr>
    <tr>
      <th>Intercept</th>             <td>    3.3000</td> <td>   18.699</td> <td>    0.176</td> <td> 0.860</td> <td>  -33.351</td> <td>   39.951</td>
    </tr>
    <tr>
      <th>C(Lighting)[T.1]</th>      <td>    0.8257</td> <td>   18.668</td> <td>    0.044</td> <td> 0.965</td> <td>  -35.765</td> <td>   37.416</td>
    </tr>
    <tr>
      <th>C(TV)[T.1]</th>            <td>   18.7899</td> <td>    1.760</td> <td>   10.678</td> <td> 0.000</td> <td>   15.341</td> <td>   22.239</td>
    </tr>
    <tr>
      <th>C(Cooking)[T.1]</th>       <td>   28.8862</td> <td>    1.969</td> <td>   14.671</td> <td> 0.000</td> <td>   25.027</td> <td>   32.746</td>
    </tr>
    <tr>
      <th>C(Refrigeration)[T.1]</th> <td>   59.2432</td> <td>    1.556</td> <td>   38.073</td> <td> 0.000</td> <td>   56.193</td> <td>   62.293</td>
    </tr>
    <tr>
      <th>C(AC)[T.1]</th>            <td>  203.3226</td> <td>    3.130</td> <td>   64.956</td> <td> 0.000</td> <td>  197.187</td> <td>  209.458</td>
    </tr>
    <tr>
      <th>C(Urban)[T.1]</th>         <td>   24.5935</td> <td>    1.391</td> <td>   17.680</td> <td> 0.000</td> <td>   21.867</td> <td>   27.320</td>
    </tr>
    <tr>
      <th>Income</th>                <td>    0.0014</td> <td>  4.1e-05</td> <td>   34.765</td> <td> 0.000</td> <td>    0.001</td> <td>    0.002</td>
    </tr>
    </table>
    <table class="simpletable">
    <tr>
      <th>Omnibus:</th>       <td>20742.858</td> <th>  Durbin-Watson:     </th>  <td>   1.789</td>  
    </tr>
    <tr>
      <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>9719176.595</td>
    </tr>
    <tr>
      <th>Skew:</th>           <td> 6.463</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
    </tr>
    <tr>
      <th>Kurtosis:</th>       <td>121.115</td>  <th>  Cond. No.          </th>  <td>8.75e+05</td>  
    </tr>
    </table>



.. code:: ipython3

    params_elec = change_index(model_results_elec.params)
    bse_elec = change_index(model_results_elec.bse)
    elec = pd.concat([params_elec, bse_elec], axis=1)
    elec.columns = ['co_mu', 'co_sd']

.. code:: ipython3

    elec.loc['Lighting', 'p'] = (electricity_data.Lighting == 1).sum() / electricity_data.shape[0]
    elec.loc['TV', 'p'] = (electricity_data.TV == 1).sum() / electricity_data.shape[0]
    elec.loc['Cooking', 'p'] = (electricity_data.Cooking == 1).sum() / electricity_data.shape[0]
    elec.loc['Refrigeration', 'p'] = (electricity_data.Refrigeration == 1).sum() / electricity_data.shape[0]
    elec.loc['AC', 'p'] = (electricity_data.AC == 1).sum() / electricity_data.shape[0]
    elec.loc['Urban', 'p'] = (electricity_data.Urban == 1).sum() / electricity_data.shape[0]

.. code:: ipython3

    elec.loc[:, 'mu'] = np.nan
    elec.loc[:, 'sd'] = np.nan
    elec.loc['Intercept', 'p'] = elec.loc['Intercept', 'co_mu']
    elec.loc['Intercept', ['co_mu', 'co_sd']] = np.nan

.. code:: ipython3

    elec.loc[:, 'dis'] = 'Bernoulli'
    elec.loc['Income', 'dis'] = 'None'
    elec.loc['Intercept', 'dis'] = 'Deterministic'

.. code:: ipython3

    elec.loc[:, 'ub'] = np.nan
    elec.loc[:, 'lb'] = np.nan
    elec.loc['Income', 'ub'] = np.inf
    elec.loc['Income', 'lb'] = 0

.. code:: ipython3

    elec.index = ['e_'+i for i in elec.index]

.. code:: ipython3

    elec.to_csv('data/table_elec.csv')

.. code:: ipython3

    elec




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
          <td>3.299984</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Deterministic</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Lighting</th>
          <td>0.825662</td>
          <td>18.667601</td>
          <td>0.998729</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_TV</th>
          <td>18.789909</td>
          <td>1.759621</td>
          <td>0.782774</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Cooking</th>
          <td>28.886242</td>
          <td>1.968938</td>
          <td>0.167474</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Refrigeration</th>
          <td>59.243236</td>
          <td>1.556048</td>
          <td>0.436812</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_AC</th>
          <td>203.322615</td>
          <td>3.130158</td>
          <td>0.059375</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Urban</th>
          <td>24.593500</td>
          <td>1.391044</td>
          <td>0.550236</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>e_Income</th>
          <td>0.001426</td>
          <td>0.000041</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>None</td>
          <td>inf</td>
          <td>0.0</td>
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

1.c Micro-level Electricity demand model
========================================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__

`Next <Ad_MCMC_water.ipynb>`__ (1.d) Micro-level Water demand model
