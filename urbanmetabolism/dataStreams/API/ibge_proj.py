#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Tue 05 Sep 2017 11:39:34 AM CEST

"""
import pandas as pd
import numpy as np
import requests
import json
from pandas.io.json import json_normalize

URL = "http://servicodados.ibge.gov.br/api/v1/projecoes/populacao/{}"

def _catch_error(data):
    try:
        error_msg = data['error']
        error = True
        print('\t| error:', error_msg)
    except:
        error = False
    return(error)

def _getData(state):
    try:
        with open('temp/proj.json', 'r') as kfile:
            data_dic = json.loads(kfile.read())
        population = data_dic['pop']
        year = data_dic['year']
    except:
        url = URL.format(state)
        response = requests.get(url)
        data = response.json()
        date = data['horario']
        year = date.split(' ')[0].split('/')[-1]
        projection = json_normalize(data['projecao'])
        population = projection.loc[0, 'populacao']
        error = _catch_error(data)
        if error:
            return(False, year)
        else:
            with open('temp/proj.json', 'w') as kfile:
                kfile.write(json.dumps({'pop': int(population), 'year': int(year)}))

    return(int(population), int(year))

def project_data(state, data,
                 initial_population=False, initial_year=2010, pop_col = 'pop',
                 projection_year=2030, growth_rate_input=False):
    if not initial_population:
        initial_population = data.loc[initial_year, pop_col]
    if not growth_rate_input:
        final_population, final_year = _getData(state)
        growth_rate = (final_population / initial_population)**(1 / (final_year - initial_year))
    else:
        growth_rate = growth_rate_input
    print("growth_rate: ", growth_rate)
    if initial_year <= 2010:
        initial_year = 2010
    for year in range(initial_year+1, projection_year+1):
        data.loc[year] = data.loc[year-1].mul(growth_rate)
    return(data)

def main():
    state = 26
    data = pd.read_csv('../../data/benchmarks_br_rec.csv', index_col=0)

    from urbanmetabolism.dataStreams.API.DATA.getData import makeTable
    TAB = '165'; VAR = '164'; IND = 'C1/0'
    total_pop = makeTable('pop', TAB, VAR, IND, level='STA')

    pop = int(total_pop.loc['26', 'Total'])
    data.loc[:, 'pop'] = pop
    proj_data = project_data(state, data)
    # print(proj_data)
    # proj_data.to_csv('benchmarks_br_rec_year.csv')


if __name__ == "__main__":
    main()
