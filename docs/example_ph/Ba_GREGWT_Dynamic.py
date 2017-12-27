
# coding: utf-8

# <div class="image123">
#     <div class="imgContainer">
#         <img src="./logos/UNEnvironment.png" alt="UNEP logo" style="width:200px">
#     </div>
#     <div class="imgContainer">
#         <img src="./logos/GI-REC.png" alt="GI_REC logo" style="width:200px">
#     </div>
# </div>
# 
# # (2.a) Dynamic Sampling Model  and GREGWT
# 
# [UN Environment](http://www.unep.org/)

# In[2]:


import datetime; print(datetime.datetime.now())


# **Notebook abstract**
# 
# This notebook shows the main sampling and reweighting algorithm.

# ## Import libraries

# In[3]:


from urbanmetabolism.population.model import run_calibrated_model, _project_survey_resample
from urbanmetabolism.population.model import TableModel


# ## Global variables

# In[4]:


iterations = 1000
benchmark_year = 2016
census_file = 'data/benchmarks_year_bias.csv'
typ = 'resampled'
model_name = 'Sorsogon_Electricity_Water_wbias_projected_dynamic_{}'.format(typ)
verbose = False
#The number of chains to run in parallel. 
njobs = 1


# ## Define Table model

# In[5]:


tm = TableModel(census_file = census_file, verbose=verbose)


# ### Income model

# In[22]:


tm.add_model('data/table_inc.csv',   'Income')
tm.update_dynamic_model('Income', specific_col = 'Education')
tm.update_dynamic_model('Income',
                        specific_col = 'FamilySize',
                        specific_col_as = 'Size',
                        val = 'mu', compute_average =  0)
tm.update_dynamic_model('Income',
                        specific_col = 'Age',
                        val = 'mu', compute_average =  0)


# In[23]:


tm.models['Income'].loc[2020]


# ### Electricity model

# In[8]:


tm.add_model('data/table_elec.csv',  'Electricity', reference_cat = ['yes'])
tm.update_dynamic_model('Electricity', specific_col = 'Income', val = 'mu', compute_average = False)


# In[9]:


tm.models['Electricity'].loc[2016]


# In[35]:


formula_elec = "e_Intercept+"+"+".join(
    ["c_{0} * {0}".format(e) for e in tm.models['Electricity'][benchmark_year].index if\
        (e != 'e_Intercept') &\
        (e != 'e_Income') &\
        (e != 'e_Urban')
    ])
formula_elec += '+c_e_Urban * i_Urbanity'
formula_elec += '+c_e_{0} * {0}'.format('Income')


# In[36]:


tm.add_formula(formula_elec, 'Electricity')


# In[37]:


tm.print_formula('Electricity')


# ### Water model

# In[10]:


tm.add_model('data/table_water.csv', 'Water')
tm.update_dynamic_model('Water', specific_col = 'Education')
tm.update_dynamic_model('Water',
                        specific_col = 'FamilySize',
                        specific_col_as = 'Size',
                        val = 'mu', compute_average =  0)
tm.update_dynamic_model('Water',
                        specific_col = 'Age',
                        val = 'mu', compute_average =  0)


# In[11]:


tm.models['Water'].loc[2020]


# In[38]:


formula_water = "w_Intercept+"+"+".join(
    ["c_{0} * {1}".format(e, "i_"+"_".join(e.split('_')[1:]))\
         for e in tm.models['Water'][benchmark_year].index if \
                                 (e != 'w_Intercept') &\
                                 (e != 'w_Total_Family_Income')   &\
                                 (e != 'w_Education')
    ])
formula_water += '+c_w_Total_Family_Income*Income'
formula_water += '+c_w_Education*i_Education'


# In[39]:


tm.add_formula(formula_water, 'Water')


# In[40]:


tm.print_formula('Water')


# ### Make model and save it to excel

# In[18]:


table_model = tm.make_model()


# In[19]:


tm.to_excel()


# ## Define model variables

# In[19]:


labels = ['age_0_18', 'age_19_25', 'age_26_35',
          'age_36_45', 'age_46_55', 'age_56_65',
          'age_66_75', 'age_76_85', 'age_86_100']
cut = [0, 19, 26, 36, 46, 56, 66, 76, 86, 101]
to_cat = {'i_Age':[cut, labels]}
drop_col_survey = ['e_Income', 'e_Urban', 'w_Total_Family_Income', 'w_Education']


# In[20]:


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
# [Next](Bb_GREGWT_NonResidential.ipynb) (2.b) Non-Residential Model
