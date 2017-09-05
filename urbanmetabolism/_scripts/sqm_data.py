#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Mon 28 Aug 2017 03:42:29 PM CEST

"""
import os
import pandas as pd
import numpy as np

SKIP = ['Total', 'Residential Condominium']
SKIPF = ['Table 1_1.xls', 'Table 1_0.xls', 'Table 1_2.xls']
FILE_LIST = [i for i in os.listdir('data') if i.split('.')[-1] == 'xls' and i not in SKIPF]

BUILDING_TYPES = {
    "Education":   ['School'],
    "Food retail": ['Other Commercial'],
    # [ 'Agricultural',
    # [ 'Other Agricultural',
    # [ 'Grain/Rice Mill',
    # [ 'Barn/Poultry House/etc.',
    # ['Slaugther House',
    "Health care":      ['Hospital/Other Similar Structures'],
    "Lodging":          ['Hotel/Motel/etc.'],
    "Office building":  ['Condominium/Office Building'],
    "Non-food retail":  ['Commercial'],
    "Public building": ['Institutional', 'Other Institutional'],
    # ['Factory',
    # ['Refinery',
    # ['Other Industrial',
    # ['Repair/Machine Shop',
    # ['Industrial'],
    # ['Apartment/Accessoria',
    # ['Duplex/Quadruplex',
    # ['Single',
    # ['Other Residential',
    "Religious buildings": ['Church/Other Religious Structures'],
    "Services": ['Banks', 'Store', 'Welfare/Charitable Structures'],
    # ['Other Non-Residential',
    # ['Printing Press',
     }

def get_pop_data(census_file,
                 start_sqm_cap = 40,
                 sqm_demand_gr = 1.03,
                 sqm_demand_nonres = 0.01,
                 sqm_nonres_mean = 800):

    census = pd.read_csv(census_file, index_col=0)

    hh_size = [int(i.split("_")[-1]) for i in census.columns if 'Size' in i]

    inx = [i for i in census.columns if 'Size' in i]

    start_year = 2010; end_year  = 2016
    start_pop = census.loc[start_year, 'pop']
    end_pop = census.loc[end_year, 'pop']

    growth_rate = (end_pop / start_pop)**(1/(end_year-start_year))
    pop = start_pop / growth_rate
    pop_dic = dict()
    for i in range(2010, 2031):
        pop =  int(np.round(pop * growth_rate))
        hh = np.average(hh_size, weights=census.loc[i, inx])
        pop_cap = pop * hh
        sqm_tot = pop_cap * start_sqm_cap
        sqm_nonres = sqm_tot * sqm_demand_nonres
        num_nonres = int(np.round(sqm_nonres / sqm_nonres_mean))
        pop_dic[i] =[pop, hh, pop_cap, start_sqm_cap, sqm_tot, sqm_nonres, num_nonres]
        start_sqm_cap *= sqm_demand_gr

    pop_data = pd.DataFrame(pop_dic, index=['pop', 'hh_size', 'cap',
                                            'sqm/cap', 'sqm', 'sqm_nonres',
                                            'num_nonres']).T

    return(pop_data)

def get_count_data():

    sqm_data = get_sqm(grouped = False, counts = True)
    sqm_data = sqm_data.groupby('group').sum()
    count_data = dict()
    for key, btyp in BUILDING_TYPES.items():
        count = sqm_data.loc[btyp].sum()
        count_data[key] = count
    count_data = pd.DataFrame(count_data).T

    return(count_data)

def get_sqm_data():

    kwh_data = pd.read_csv('data/Commercial.data.csv', index_col=0)
    kwh_data.columns= [i.strip() for i in kwh_data.columns]
    kwh_data.loc[:, 'std'] = np.sqrt(np.sum(np.square(
        kwh_data.loc[:,['Min', 'Max']].sub(kwh_data.loc[:,'Benchmark kWh/m2'],
                                           axis=0)), axis=1) / 2)

    sqm_data = get_sqm()

    nr_data = pd.DataFrame(columns=['sqm', 'sqm_sd', 'kwh', 'kwh_sd'])
    for typ_kwh, typ_sqm in BUILDING_TYPES.items():
        sqm_mean = sqm_data.loc[BUILDING_TYPES[typ_kwh], 'mean'].mean()
        sqm_std = sqm_data.loc[BUILDING_TYPES[typ_kwh], 'std'].max()
        kwh_mean = kwh_data.loc[typ_kwh, 'Benchmark kWh/m2']
        kwh_std  = kwh_data.loc[typ_kwh, 'std']
        nr_data.loc[typ_kwh, :] = [sqm_mean, sqm_std, kwh_mean, kwh_std]

    return(nr_data)

def _get_table(i, s, verbose = False):
    table_data = pd.read_excel(
        os.path.join('data', i),
        index_col = 0,
        skiprows = 3,
        header = [0,1,2,3],
        sheetname = s
    )
    if verbose: print('got table')
    table_data = table_data.loc[table_data.index.notnull()]
    table_data.columns = table_data.columns.droplevel([2,3])
    table_data.columns.names = ['Building type', 'Variable']
    table_data.index.name = "Region"

    table_year = pd.read_excel(
        os.path.join('data', i),
        sheetname = s
    ).columns[0].split(',')[-1]
    table_year = table_year.replace(' - continued', '')
    table_year = table_year.strip()

    return(table_year, table_data)

def _get_tables(i, verbose = False):
    tables = list()
    for s in range(2):
        table_year, table_data = _get_table(i, s, verbose = verbose)
        tables.append(table_data)
    table_data = pd.concat(tables, axis=1)
    table_data = table_data.dropna(how='all')
    return(table_data, table_year)

def _compute_sqm(i, verbose = False, counts = False):
    table_data, year = _get_tables(i, verbose = verbose)
    idx = pd.IndexSlice
    sqm = list()
    for typ in table_data.columns.get_level_values(0).unique():
        if verbose: print("{:<25}".format(typ), end='\t')
        if typ not in SKIP:
            sqm_vector_count = table_data.loc[:, idx[typ, 'Number']]
            if counts:
                sqm_vector = sqm_vector_count
            else:
                sqm_vector = table_data.loc[:, idx[typ, 'Floor Area']].div(sqm_vector_count)
            sqm_vector.name = "{}_{}".format(year, typ)
            sqm.append(sqm_vector)
            if verbose: print('OK')
        else:
            if verbose: print("-")
    data = pd.concat(sqm, axis=1)
    return(data)

def get_sqm(grouped = True, verbose = False, counts = False):
    sqm_data_list = list()
    for e, i in enumerate(FILE_LIST):
        sqm_data_list.append(_compute_sqm(i, counts = counts, verbose = verbose).mean())
    sqm_data = pd.concat(sqm_data_list)
    sqm_data = pd.DataFrame(sqm_data)
    if counts:
        sqm_data.columns = ['counts']
    else:
        sqm_data.columns = ['sqm']
    sqm_data.loc[:, 'year'] = [i.split('_')[0] for i in sqm_data.index]
    sqm_data.loc[:, 'group']  = [i.split('_')[1] for i in sqm_data.index]
    if grouped:
        sqm_data = sqm_data.groupby('group').describe()
        sqm_data.columns = sqm_data.columns.droplevel(0)
    return(sqm_data)

def main():
    pass

if __name__ == "__main__":
    main()
