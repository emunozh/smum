
# coding: utf-8

# # Dynamic Model (GREGWT)
#
# UN Environment

# In[53]:

import datetime; print(datetime.datetime.now())


# In[54]:

from urbanmetabolism.population.model import run_calibrated_model, _project_survey_resample
from urbanmetabolism.population.model import TableModel


# ## Global variables

# In[55]:

iterations = 1000#00
year = 2016
state = 'bra'
#state = 'rec'
census_file = 'data/benchmarks_br_{}_year.csv'.format(state)
typ = 'reweighted'
#typ = 'resampled'
model_name = '{}_Electricity_Water_projected_dynamic_{}'.format(state.upper(), typ)
verbose = True


# ## Define model

# In[56]:

tm = TableModel(census_file = census_file, verbose=verbose)


# In[57]:

# order matters
tm.add_model('data/test_water_br_{}.csv'.format(state), 'Water')
tm.add_model('data/test_elec_br_{}.csv'.format(state),  'Electricity')


# In[58]:

#tm.update_dynamic_model('Electricity', specific_col = 'edu')
#tm.update_dynamic_model('Income', specific_col = 'Size', val = 'mu', compute_average =  0)
#tm.update_dynamic_model('Income', specific_col = 'Age',  val = 'mu', compute_average = -4)


# In[59]:

formula_elec = "e_Intercept+"+"+".join(
    ["c_{0}*{0}".format(e) for e in tm.models['Electricity'][year].index \
     if (e not in ['e_Intercept', 'e_cdd', 'e_dutyp', 'e_urban', 'e_sex'])
    ])
#formula_elec += "*e_cdd"
formula_elec += '+c_e_dutyp*w_dutyp + c_e_urban*w_urban + c_e_sex*w_sex'
tm.add_formula(formula_elec, 'Electricity')


# In[67]:

formula_water = "w_Intercept+"+"+".join(
    ["c_{0}*{0}".format(e) for e in tm.models['Water'][year].index if\
     (e not in  ['w_Intercept'])
    ])
tm.add_formula(formula_water, 'Water')


# In[68]:

formula_water


# In[22]:

table_model = tm.make_model()


# In[23]:

#table_model['Water']['table_model'].loc[[2015], :, ['mu', 'p', 'dis', 'co_mu']].to_frame()


# In[24]:

#tm.print_formula('Electricity')


# ## Define model variables

# In[51]:

labels_age = [
    'age_20a24', 'age_25a29', 'age_30a34',
    'age_35a39', 'age_40a44', 'age_45a49',
    'age_50a54', 'age_55a59', 'age_60a64',
    'age_65a69', 'age_70a74', 'age_75a79',
    'age_80anosou'
          ]
cut_age = [19,
       24, 29, 34,
       39, 44, 49,
       54, 59, 64,
       69, 74, 79,
       120]

labels_hh = ['hhsize_{}'.format(i) for i in range(1, 8)]
cut_hh = [0,1.55,2.55,3.55,4.55,5.55,6.55,20]
to_cat = {'w_age':[cut_age, labels_age], 'e_hhsize':[cut_hh, labels_hh]}
#drop_col_survey = ['e_Income', 'i_Urbanity']


# In[ ]:

fw = run_calibrated_model(
    table_model,
    verbose = verbose,
    project = typ,
    from_script = True,
    #resample_years = [2019, 2020],
    census_file = census_file,
    year = year,
    population_size = False,
    name = '{}_{}'.format(model_name, iterations),
    to_cat = to_cat,
    iterations = iterations,
    #drop_col_survey = drop_col_survey
)


# In[18]:

#import json
#with open('temp/kfactors.json', 'r') as kfile:
#    k_iter = json.loads(kfile.read())


# In[19]:

#import pandas as pd
#census = pd.read_csv(census_file, index_col=0)


# In[32]:

#_project_survey_resample(
#    census,
#    table_model,
#    'wf',
#    k_iter,
#    population_size = False,
#    name = '{}_{}'.format(model_name, iterations),
#    to_cat = to_cat,
#    iterations = iterations,
#    census_file = census_file,
#    drop_col_survey = drop_col_survey)

