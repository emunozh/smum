# load libraries
import pandas as pd
from urbanmetabolism.population.model import run_calibrated_model

# load model coefficients
elec  = pd.read_csv('data/test_elec.csv',   index_col=0)
inc   = pd.read_csv('data/test_inc.csv',    index_col=0)
water = pd.read_csv('data/test_water.csv',  index_col=0)

# define model, order matters
model = {"Income":      {'table_model': inc },
         "Water":       {'table_model': water},
         "Electricity": {'table_model': elec}}

# run simulation
run_calibrated_model(
    model,
    census_file = 'data/benchmarks_projected.csv',
    year = 2016)
