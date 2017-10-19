
# coding: utf-8

# # Dynamic Model (GREGWT)
#
# UN Environment

# In[ ]:

import datetime; print(datetime.datetime.now())


# In[ ]:

from urbanmetabolism.population.model import run_calibrated_model, _project_survey_resample
from urbanmetabolism.population.model import TableModel


# ## Global variables

# In[ ]:

iterations = 1000#00
year = 2016
# state = 'bra'
state = 'rec'
census_file = 'data/benchmarks_br_{}_year.csv'.format(state)
typ = 'reweighted'
#typ = 'resampled'
model_name = '{}_Electricity_Water_projected_dynamic_{}'.format(state.upper(), typ)
verbose = False


# ## Define model

# In[ ]:

tm = TableModel(census_file = census_file, verbose=verbose)

# order matters
tm.add_model('data/test_water_br_{}.csv'.format(state), 'Water')
tm.add_model('data/test_elec_br_{}.csv'.format(state),  'Electricity')

formula_elec = "e_Intercept+"+"+".join(
    ["c_{0}*{0}".format(e) for e in tm.models['Electricity'][year].index \
     if (e not in ['e_Intercept', 'e_cdd', 'e_dutyp', 'e_urban', 'e_sex'])
    ])
formula_elec += '+c_e_dutyp*w_dutyp + c_e_urban*w_urban + c_e_sex*w_sex'

tm.add_formula(formula_elec, 'Electricity')
formula_water = "w_Intercept+"+"+".join(
    ["c_{0}*{0}".format(e) for e in tm.models['Water'][year].index if\
     (e not in  ['w_Intercept'])
    ])
tm.add_formula(formula_water, 'Water')

table_model = tm.make_model()


# ## Define model variables

# In[ ]:

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
    #rep = False,
    #from_script = True,
    #resample_years = [2019, 2020],
    census_file = census_file,
    year = year,
    population_size = False,
    name = '{}_{}'.format(model_name, iterations),
    to_cat = to_cat,
    iterations = iterations,
    #drop_col_survey = drop_col_survey
)

