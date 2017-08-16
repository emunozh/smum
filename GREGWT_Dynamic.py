
# coding: utf-8

# # Dynamic Model (GREGWT)
# 
# UN Environment

# In[1]:

import datetime; print(datetime.datetime.now())


# In[2]:

from urbanmetabolism.population.model import run_calibrated_model
from urbanmetabolism.population.model import TableModel
from urbanmetabolism.population.model import _make_flat_model


# ## Global variables

# In[3]:

iterations = 100000
year = 2016
census_file = 'data/benchmarks_projected_wbias.csv'
model_name = 'Sorsogon_Electricity_Water_wbias_projected_dynamic'


# ## Define model

# In[4]:

tm = TableModel(census_file = census_file, verbose=False)


# In[5]:

# order matters
tm.add_model('data/test_inc.csv',   'Income')
tm.add_model('data/test_elec.csv',  'Electricity')
tm.add_model('data/test_water.csv', 'Water')


# In[6]:

tm.update_dynamic_model('Income', specific_col = 'Education')
tm.update_dynamic_model('Income', specific_col = 'Size', val = 'mu', compute_average =  0)
tm.update_dynamic_model('Income', specific_col = 'Age',  val = 'mu', compute_average = -4)


# In[7]:

formula_elec = "e_Intercept+"+"+".join(
    ["c_{0}*{0}".format(e) for e in tm.models['Electricity'][year].index if (e != 'e_Intercept')&\
                                                  (e != 'e_Income')])
formula_elec += '+c_e_{0}*{0}'.format('Income')


# In[8]:

tm.add_formula(formula_elec, 'Electricity')


# In[9]:

formula_water = "w_Intercept+"+"+".join(
    ["c_{0}*{1}".format(e, "i_"+"_".join(e.split('_')[1:]))\
         for e in tm.models['Water'][year].index if (e != 'w_Intercept')&\
                                 (e != 'w_Total_Family_Income')   &\
                                 (e != 'w_Education_cat')])
formula_water += '+c_w_Total_Family_Income*Income'
formula_water += '+c_w_Education_cat'


# In[10]:

tm.add_formula(formula_water, 'Water')


# In[11]:

table_model = tm.make_model()


# In[13]:

table_model['Electricity']['table_model'].loc[[2015, 2020, 2025, 2030], :, ['mu', 'p']].to_frame()


# In[14]:

tm.print_formula('Electricity')


# ## Define model variables

# In[15]:

labels = ['age_0_18', 'age_19_25', 'age_26_35',
          'age_36_45', 'age_46_55', 'age_56_65',
          'age_66_75', 'age_76_85', 'age_86_100']
cut = [0, 19, 26, 36, 46, 56, 66, 76, 86, 101]
to_cat = {'i_HH_head_Age':[cut, labels]}
drop_col_survey = ['e_Income', 'i_Urbanity']


# In[ ]:

fw = run_calibrated_model(
    table_model,
    #verbose = True,
    project = 'resample',
    census_file = census_file,
    year = year,
    population_size = False,
    name = '{}_{}'.format(model_name, iterations),
    to_cat = to_cat,
    iterations = iterations,
    drop_col_survey = drop_col_survey)

