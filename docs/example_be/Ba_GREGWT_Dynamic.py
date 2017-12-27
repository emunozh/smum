
# coding: utf-8

# # Spatial$^{*}$ Microsimulation Urban Metabolism Model (SMUM)
# 
# <div class="image123">
#     <div class="imgContainer">
#         <img src="./logos/UNEnvironment.png" alt="UNEP logo" style="width:200px">
#     </div>
#     <div class="imgContainer">
#         <img src="./logos/GI-REC.png" alt="GI_REC logo" style="width:200px">
#     </div>
# </div>
# 
# # 2.a Dynamic Sampling Model  and GREGWT
# 
# [UN Environment](http://www.unep.org/)

# In[1]:


import datetime; print(datetime.datetime.now())


# **Notebook abstract**
# 
# This notebook shows the main sampling and reweighting algorithm.

# ## Import libraries

# In[2]:


from urbanmetabolism.population.model import run_calibrated_model
from urbanmetabolism.population.model import TableModel


# ## Global variables

# In[3]:


iterations = 1000
year = 2016
census_file = 'data/benchmarks_be_year_bias3_climate.csv'
typ = 'resampled'
model_name = 'Brussels_Electricity_Water_projected_dynamic_{}_bias'.format(typ)
verbose = False
#The number of chains to run in parallel. 
njobs = 2


# ## Define Table model

# In[4]:


tm = TableModel(census_file = census_file, verbose=verbose)


# ### Water model

# In[13]:


tm.add_model('data/table_water.csv', 'Water')

tm.update_dynamic_model(
    'Water', specific_col = 'ConstructionType', select = 1)
tm.update_dynamic_model(
    'Water', specific_col = 'Age', val = 'mu', compute_average = 0)
tm.update_dynamic_model(
    'Water', specific_col = 'ConstructionYear', val = 'mu')
tm.update_dynamic_model(
    'Water', specific_col = 'HHSize', val = 'mu')
tm.update_dynamic_model(
    'Water', specific_col = 'Income', val = 'mu',
    compute_average = False)


# In[14]:


tm.models['Water'].loc[2020]


# In[15]:


formula_water = "+".join(
    ["c_{0}*{0}".format(e) for e in tm.models['Water'][year].index if\
     (e not in  ['w_Intercept'])
    ])
tm.add_formula(formula_water, 'Water')


# In[16]:


tm.add_formula(formula_water, 'Water')


# In[17]:


tm.print_formula('Water')


# ### Electricity model

# In[5]:


tm.add_model('data/table_elec.csv',  'Electricity',
            skip_cols = [
                'ConstructionType',
                'Income',
                'HHSize',
                'ConstructionYear',
                'ELWARM',
                'ELWATER',
                'ELFOOD'])
tm.update_dynamic_model(
   'Electricity', specific_col = 'sqm', val = 'mu',
    compute_average = False)
tm.update_dynamic_model(
    'Electricity', specific_col = 'CDD',
    static = True,
    compute_average = False)
tm.update_dynamic_model(
    'Electricity', specific_col = 'HDD',
    static = True,
    compute_average = False)


# In[6]:


tm.models['Electricity'].loc[2016]


# In[10]:


skip_elec = [
    'e_Intercept', 'e_ConstructionType', 'e_Income', 'e_HHSize', 'e_ConstructionYear',
    'e_CDD', 'e_HDD',
]
formula_elec = "+".join(
    ["c_{0}*{0}".format(e) for e in tm.models['Electricity'][year].index \
     if (e not in skip_elec)
    ])
formula_elec += '+c_e_ConstructionType*w_ConstructionType +c_e_Income*w_Income +c_e_HHSize*w_HHSize +c_e_ConstructionYear*w_ConstructionYear+e_CDD +e_HDD'


# In[11]:


tm.add_formula(formula_elec, 'Electricity')


# In[12]:


tm.print_formula('Electricity')


# ### Make model and save it to excel

# In[18]:


table_model = tm.make_model()


# In[19]:


tm.to_excel(sufix = "_be")


# ## Define model variables

# In[21]:


labels_age = [
    'Age_24', 'Age_29', 'Age_39',#3
    'Age_54', 'Age_64', 'Age_79',#6
    'Age_120']
cut_age = [17,
       24, 29, 39,
       54, 64, 79,
       120]
  
labels_cy = [
    'ConstructionYear_1900', 'ConstructionYear_1918',
    'ConstructionYear_1945', 'ConstructionYear_1961',
    'ConstructionYear_1970', 'ConstructionYear_1981',
    'ConstructionYear_1991', 'ConstructionYear_2001',
    'ConstructionYear_2011', 'ConstructionYear_2016',
    'ConstructionYear_2020', 'ConstructionYear_2030',
    'ConstructionYear_2035']
cut_cy = [0,
          1900, 1918,
          1945, 1961,
          1970, 1981,
          1991, 2001,
          2011, 2016,
          2020, 2030,
          2100]

to_cat = {
    'w_Age':[cut_age, labels_age],
    'w_ConstructionYear':[cut_cy, labels_cy],
         }

drop_col_survey = [
    'e_ConstructionType', 'e_Income', 'e_HHSize', 'e_ConstructionYear',
    'e_HDD', 'e_CDD'
]


# In[20]:


fw = run_calibrated_model(
    table_model,
    verbose = verbose,
    project = typ,
    njobs = njobs,
    census_file = census_file,
    year = year,
    name = '{}_{}'.format(model_name, iterations),
    to_cat = to_cat,
    iterations = iterations,
    drop_col_survey = drop_col_survey)


# <div class="image123">
#     <div class="imgContainer">
#         <img src="./logos/UNEnvironment.png" alt="UNEP logo" style="width:200px">
#     </div>
#     <div class="imgContainer">
#         <img src="./logos/GI-REC.png" alt="GI_REC logo" style="width:200px">
#     </div>
# </div>
# 
# # 2.a Micro-level Electricity demand model
# 
# [UN Environment](http://www.unep.org/)
# 
# [Home](Welcome.ipynb)
# 
# [Next](Bc_GREGWT_validation_wbias.ipynb) (2.c) Model Internal Validation
