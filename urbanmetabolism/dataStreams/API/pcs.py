#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 22 Mar 2017 11:52:31 AM CET

"""
import pandas as pd
import numpy as np
import requests
from pandas.io.json import json_normalize

URL = "http://indicadores.cidadessustentaveis.org.br/download-{}s/?content-type=application/json"
INDICATORS_URL = "http://indicadores.cidadessustentaveis.org.br/indicators/?content-type=application/json"
response = requests.get(INDICATORS_URL)
data = response.json()
indicators_table = json_normalize(data['indicators'])
inx = ['id', 'name', 'variable_type']
indicators_table = indicators_table.loc[:, inx]
indicators_table = indicators_table.set_index('id')
name_col = 'name'

variable_file_name = 'http://indicadores.cidadessustentaveis.org.br/variaveis.csv'
variables_table = pd.read_csv(
    variable_file_name,
    usecols = [2, 8],
    index_col=0)
variables_table = variables_table.drop_duplicates()
name_col_var = 'Nome'

def search(term):
    inx = [i for i in indicators_table.index if term in indicators_table.loc[i, name_col]]
    found = indicators_table.loc[inx]
    print('Found in Indicators:')
    print(found)
    inx = [i for i in variables_table.index if term in variables_table.loc[i, name_col_var]]
    found = variables_table.loc[inx]
    print('#'*30)
    print('Found in Variables:')
    print(found)

def _get_col_name(table_in, indicator_id, endpoint):
    if endpoint == 'indicator':
        table = indicators_table
        col_name = name_col
    elif endpoint == 'variable':
        table = variables_table
        col_name = name_col_var

    new_col_names = table.loc[indicator_id, col_name]
    table_in.columns = [new_col_names]
    return(table_in)

def _catch_error(data):
    try:
        error_msg = data['error']
        error = True
        print('\t| error:', error_msg)
    except:
        error = False
    return(error)

def _getIndicator(indicator_id, endpoint = 'indicator',
                 compute_mean = False,
                 return_raw=False, return_raw_table=False, pos=0):
    is_str = False
    usecols = ['city_id', 'valid_from', 'value', 'variation_order']
    url = URL.format(endpoint)
    if endpoint == 'indicator':
        payload = {'indicator_id': indicator_id}
    elif endpoint == 'variable':
        payload = {'variable_id': indicator_id}
    response = requests.get(url, params=payload)
    data = response.json()
    error = _catch_error(data)
    if error: return(False, is_str)
    if return_raw: return(data, is_str)
    try:
        table = json_normalize(data[pos]['data'])
    except:
        table = json_normalize(data['data'])
    table = table.loc[:, usecols]
    if np.isnan(table.variation_order.sum()):
        table.loc[:, 'variation_order'] = 0
    if return_raw_table: return(table, is_str)
    table = table.dropna()

    table.loc[:, "year"] = pd.to_datetime(table.valid_from).map(lambda x: x.year)
    table = table.drop('valid_from', axis=1)

    # import ipdb; ipdb.set_trace() # BREAKPOINT
    if compute_mean:
        table = table.drop(['year', 'variation_order'], axis=1)
        try:
            table.loc[:, 'value'] = table.loc[:, 'value'].astype(float)
            table = table.loc[table.value > 0]
            table = table.groupby('city_id').mean()
        except:
            is_str = True
            table = table.groupby('city_id')['value'].apply(
                lambda x: "{%s}" % ', '.join(x))
            print("\t| can't convert to float.", type(table))
    else:
        table.index = table.loc[:,['city_id', 'year']].apply(
            lambda row: '-'.join(map(str, row)), axis=1)
        table = table.drop(['year','city_id'], axis=1)
        if len(table.variation_order.unique()) > 0:
            for e, var in enumerate(table.variation_order.unique()):
                if e == 0:
                    left = table.loc[table.variation_order == var, 'value'].to_frame(
                        name="{}-{}".format(indicator_id, var))
                else:
                    right = table.loc[table.variation_order == var, 'value'].to_frame(
                        name="{}-{}".format(indicator_id, var))
                    left = pd.merge(
                        left, right, left_index=True, right_index=True, how='outer')
            table = left

    table = _get_col_name(table, indicator_id, endpoint)

    return(table, is_str)

from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
import os

def get_PCM(table, cut = 0.7, **kwargs):
    if os.path.isfile(table):
        print('input is file')
        table = pd.read_csv(table, index_col = 0)
    if isinstance(table, pd.DataFrame):
        print('input is data frame')
        print('\t| initial components: ', table.shape[1])
        table = table.loc[:, table.sum().notnull()]
        print('\t| without empty components: ', table.shape[1])
        missing_values = table.apply(lambda x: np.sum(np.isnan(x)))
        inx = missing_values >= table.shape[0] * cut
        table = table.loc[:, inx]
        # table = table.dropna()
        print('\t| cut value: ', cut)
        print('\t| with cut components: ', table.shape[1])
        table = table.as_matrix()
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(table)
        table = imp.transform(table)
    else:
        print('error: table is not data frame')
    pca = PCA(**kwargs)
    table = pca.fit_transform(table)
    return(table)


def getTable(indicators, skip = 0, skip_str = True, **kwargs):
    if not isinstance(indicators, list): return(False)
    i = 0; result = False
    for e, indicator in enumerate(indicators):
        print("{}/{}".format(e+1, len(indicators)), end='; ')
        print('retriving indicator: {}'.format(indicator), end='\n')
        if e >= skip:
            tab, is_str = _getIndicator(indicator, **kwargs)
            if not skip_str: is_str = False
            if not is_str and not isinstance(tab, bool):
                if isinstance(tab, pd.DataFrame) or isinstance(tab, pd.Series):
                    if i == 0:
                        result = tab
                    else:
                        result = pd.merge(
                            result, tab,
                            left_index=True, right_index=True, how='outer')
                else:
                    try:
                        result.append(tab)
                    except:
                        result = list()
                        result.append(tab)
                i += 1
    try:
        result = result.astype('float')
    except:
        print("\t| can't convert to float.", type(result))
    return result

def main():
    # elec = getTable(
        # [-677], endpoint='variable', compute_mean = True,
        # return_raw=True, return_raw_table=False, pos=0
    # )
    # print(elec.head())
    # print(elec)
    pop = getTable([101])
    print(pop.head())

    # inx = variables_table.index.tolist()
    # inx = [i for i in inx if int(i) >= 0]
    # all_data = getTable(inx,
                        # endpoint='variable', compute_mean=True#, skip = 472
                        # )
    # all_data.to_csv('all_data.csv')

    # table = get_PCM('all_data.csv', cut = 0.8)#, n_components = 'mle', svd_solver = 'full')
    # print(table)


if __name__ == "__main__":
    main()
