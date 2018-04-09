
# coding: utf-8

# ## Brussels. Step 2.b Dynamic Sampling Model and GREGWT, Non-Residential Model

# In[1]:


import datetime; print(datetime.datetime.now())


# **Notebook abstract**
# 
# This notebook shows the main sampling and reweighting algorithm for the non-residential sector.

# ### Import libraries

# In[2]:


from urbanmetabolism.population.model import run_calibrated_model
from urbanmetabolism.population.model import plot_data_projection
from urbanmetabolism.population.model import TableModel


# In[3]:


#from urbanmetabolism._scripts.material_btyp import get_den


# In[4]:


#year = 2030
#sqm = 60
#construction = "SFH"
#d = get_den(year, construction, sqm)


# ### Global variables

# In[5]:


iterations = 10000
benchmark_year = 2016
census_file = 'data/benchmarks_nonresidential.csv'
typ = 'resampled'
model_name = 'Brussels_NonResidentialElectricity_wbias_projected_dynamic_{}'.format(typ)
verbose = False
drop_col_survey = ['h_BuildingHeat', 'c_BuldingCool', 'e_BuildingElec',
                   'h_BuildingSqm', 'c_BuildingSqm']


# ### Define model

# In[6]:


tm = TableModel(census_file = census_file, verbose=verbose)
table_model_name = 'data/table_elec_nr_{}.csv'


# #### Electricity

# In[7]:


tm.add_model(table_model_name.format('e'), 'elec', static = True)
tm.add_model(table_model_name.format('h'), 'heat', static = True)
tm.add_model(table_model_name.format('c'), 'cool', static = True)
#tm.update_dynamic_model(estimate_var, specific_col = 'BuildingKwh', static=True)


# In[8]:


tm.models['cool'].loc[2020]


# In[9]:


formula_nrb = "c_e_BuildingElec * e_BuildingElec * c_e_BuildingSqm * e_BuildingSqm"
tm.add_formula(formula_nrb, 'elec')


# In[10]:


formula_nrb = "c_h_BuildingHeat * h_BuildingHeat * c_e_BuildingSqm * e_BuildingSqm"
tm.add_formula(formula_nrb, 'heat')


# In[11]:


formula_nrb = "c_c_BuildingCool * c_BuildingCool * c_e_BuildingSqm * e_BuildingSqm"
tm.add_formula(formula_nrb, 'cool')


# In[12]:


table_model = tm.make_model()
tm.to_excel()


# ### Run model

# In[ ]:


fw = run_calibrated_model(
    table_model,
    verbose = verbose,
    project = typ,
    census_file = census_file,
    year = benchmark_year,
    population_size = False,
    name = '{}_{}'.format(model_name, iterations),
    iterations = iterations,
    align_census = False,
    drop_col_survey = drop_col_survey
)


# ### Plot results

# In[14]:


reweighted_survey = 'data/survey_{}_{}'.format(model_name, iterations)


# In[22]:


data = plot_data_projection(
    reweighted_survey, ['elec','heat','cool'], "{}, {}".format(iterations, typ),
    start_year = 2016, end_year = 2025, aspect_ratio = 4,
    benchmark_year = False, unit = "building")

