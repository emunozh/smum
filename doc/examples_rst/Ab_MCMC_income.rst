
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

1.b Micro-level Income model
============================

`UN Environment <http://www.unep.org/>`__

.. code:: ipython3

    import datetime; print(datetime.datetime.now())


.. parsed-literal::

    2017-10-25 14:37:54.542583


**Notebook Abstract:**

A simple micro-level income model.

Prior income model
------------------

.. code:: ipython3

    import statsmodels.api as sm
    import pandas as pd
    import numpy as np
    from urbanmetabolism._scripts.micro import compute_categories, change_index

.. code:: ipython3

    income_data = pd.read_csv('data/income.csv', index_col=0)
    formula = "Total_Family_Income ~\
    Family_Size + C(HH_head_Sex) + HH_head_Age + C(Education) + C(Urbanity)"

.. code:: ipython3

    income_data.head()




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
          <th>0</th>
          <td>5.5</td>
          <td>1</td>
          <td>52</td>
          <td>2.0</td>
          <td>1500</td>
          <td>0</td>
          <td>23939.666667</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>7.5</td>
          <td>1</td>
          <td>70</td>
          <td>1.0</td>
          <td>1608</td>
          <td>0</td>
          <td>16078.166667</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.0</td>
          <td>1</td>
          <td>49</td>
          <td>2.0</td>
          <td>8880</td>
          <td>0</td>
          <td>20925.000000</td>
          <td>0</td>
        </tr>
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
          <th>4</th>
          <td>6.0</td>
          <td>1</td>
          <td>36</td>
          <td>1.0</td>
          <td>3360</td>
          <td>0</td>
          <td>13589.500000</td>
          <td>0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    model_inc = sm.WLS.from_formula(formula, income_data)
    model_results_inc = model_inc.fit()

.. code:: ipython3

    model_results_inc.summary()




.. raw:: html

    <table class="simpletable">
    <caption>WLS Regression Results</caption>
    <tr>
      <th>Dep. Variable:</th>    <td>Total_Family_Income</td> <th>  R-squared:         </th>  <td>   0.315</td>  
    </tr>
    <tr>
      <th>Model:</th>                    <td>WLS</td>         <th>  Adj. R-squared:    </th>  <td>   0.315</td>  
    </tr>
    <tr>
      <th>Method:</th>              <td>Least Squares</td>    <th>  F-statistic:       </th>  <td>   1908.</td>  
    </tr>
    <tr>
      <th>Date:</th>              <td>Mon, 23 Oct 2017</td>   <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
    </tr>
    <tr>
      <th>Time:</th>                  <td>16:37:05</td>       <th>  Log-Likelihood:    </th> <td>-3.5601e+05</td>
    </tr>
    <tr>
      <th>No. Observations:</th>       <td> 33208</td>        <th>  AIC:               </th>  <td>7.120e+05</td> 
    </tr>
    <tr>
      <th>Df Residuals:</th>           <td> 33199</td>        <th>  BIC:               </th>  <td>7.121e+05</td> 
    </tr>
    <tr>
      <th>Df Model:</th>               <td>     8</td>        <th>                     </th>      <td> </td>     
    </tr>
    <tr>
      <th>Covariance Type:</th>       <td>nonrobust</td>      <th>                     </th>      <td> </td>     
    </tr>
    </table>
    <table class="simpletable">
    <tr>
               <td></td>              <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
    </tr>
    <tr>
      <th>Intercept</th>           <td> 1147.6640</td> <td>  313.997</td> <td>    3.655</td> <td> 0.000</td> <td>  532.218</td> <td> 1763.110</td>
    </tr>
    <tr>
      <th>C(HH_head_Sex)[T.2]</th> <td>  919.0121</td> <td>  161.503</td> <td>    5.690</td> <td> 0.000</td> <td>  602.460</td> <td> 1235.565</td>
    </tr>
    <tr>
      <th>C(Education)[T.2.0]</th> <td> 6023.8625</td> <td>  140.904</td> <td>   42.751</td> <td> 0.000</td> <td> 5747.685</td> <td> 6300.040</td>
    </tr>
    <tr>
      <th>C(Education)[T.3.0]</th> <td> 1.196e+04</td> <td>  217.209</td> <td>   55.058</td> <td> 0.000</td> <td> 1.15e+04</td> <td> 1.24e+04</td>
    </tr>
    <tr>
      <th>C(Education)[T.4.0]</th> <td> 1.873e+04</td> <td>  282.176</td> <td>   66.368</td> <td> 0.000</td> <td> 1.82e+04</td> <td> 1.93e+04</td>
    </tr>
    <tr>
      <th>C(Education)[T.5.0]</th> <td> 1.679e+04</td> <td>  742.048</td> <td>   22.624</td> <td> 0.000</td> <td> 1.53e+04</td> <td> 1.82e+04</td>
    </tr>
    <tr>
      <th>C(Urbanity)[T.1]</th>    <td> 7105.2245</td> <td>  127.941</td> <td>   55.535</td> <td> 0.000</td> <td> 6854.455</td> <td> 7355.994</td>
    </tr>
    <tr>
      <th>Family_Size</th>         <td> 1666.8464</td> <td>   29.035</td> <td>   57.409</td> <td> 0.000</td> <td> 1609.937</td> <td> 1723.756</td>
    </tr>
    <tr>
      <th>HH_head_Age</th>         <td>  116.5759</td> <td>    4.681</td> <td>   24.902</td> <td> 0.000</td> <td>  107.400</td> <td>  125.752</td>
    </tr>
    </table>
    <table class="simpletable">
    <tr>
      <th>Omnibus:</th>       <td>3597.783</td> <th>  Durbin-Watson:     </th> <td>   1.606</td>
    </tr>
    <tr>
      <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>4994.573</td>
    </tr>
    <tr>
      <th>Skew:</th>           <td> 0.865</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
    </tr>
    <tr>
      <th>Kurtosis:</th>       <td> 3.786</td>  <th>  Cond. No.          </th> <td>    642.</td>
    </tr>
    </table>



.. code:: ipython3

    params_inc = change_index(model_results_inc.params)
    bse_inc = change_index(model_results_inc.bse)
    inc = pd.concat([params_inc, bse_inc], axis=1)
    inc.columns = ['co_mu', 'co_sd']
    inc = compute_categories(inc)

.. code:: ipython3

    inc.loc['Urbanity', 'p'] = (income_data.Urbanity == 1).sum() / income_data.shape[0]
    inc.loc['Sex', 'p'] = (income_data.HH_head_Sex == 2).sum() / income_data.shape[0]

.. code:: ipython3

    inc.loc[:, 'mu'] = np.nan
    inc.loc[:, 'sd'] = np.nan
    inc.loc['Intercept', 'p'] = inc.loc['Intercept', 'co_mu']
    inc.loc['Intercept', ['co_mu', 'co_sd']] = np.nan

.. code:: ipython3

    inc.loc['Education','dis'] = 'Categorical'
    inc.loc['Urbanity', 'dis'] = 'Bernoulli'
    inc.loc['Sex', 'dis'] = 'Bernoulli'
    inc.loc['FamilySize', 'dis'] = 'Poisson'
    inc.loc['Intercept', 'dis'] = 'Deterministic'
    inc.loc['Age', 'dis'] = 'Normal'

.. code:: ipython3

    inc.loc[:,'ub'] = np.nan
    inc.loc[:,'lb'] = np.nan
    inc.loc['FamilySize', 'lb'] = 1
    inc.loc['FamilySize', 'ub'] = 10
    inc.loc['Age', 'ub'] = 100
    inc.loc['Age', 'lb'] = 18

.. code:: ipython3

    inc.index = ['i_'+i for i in inc.index]

.. code:: ipython3

    inc.to_csv('data/table_inc.csv')

.. code:: ipython3

    inc




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
          <td>1147.663992</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Deterministic</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>i_Sex</th>
          <td>919.012</td>
          <td>161.503</td>
          <td>0.193718</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>i_Urbanity</th>
          <td>7105.22</td>
          <td>127.941</td>
          <td>0.403005</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Bernoulli</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>i_FamilySize</th>
          <td>1666.85</td>
          <td>29.0348</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Poisson</td>
          <td>10.0</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>i_Age</th>
          <td>116.576</td>
          <td>4.68139</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Normal</td>
          <td>100.0</td>
          <td>18.0</td>
        </tr>
        <tr>
          <th>i_Education</th>
          <td>1.0,6023.86254599,11959.091528,18727.4606703,1...</td>
          <td>1e-10,140.904404522,217.208790314,282.17614554...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>Categorical</td>
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

1.b Micro-level Income model
============================

`UN Environment <http://www.unep.org/>`__

`Home <Welcome.ipynb>`__

`Next <Ac_MCMC_electricity.ipynb>`__ (1.c) Micro-level Electricity
demand model
