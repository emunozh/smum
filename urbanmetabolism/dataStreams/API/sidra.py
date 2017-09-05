#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 05 Apr 2017 12:33:00 PM CEST

"""
import pandas as pd
import statsmodels.api as sm
import numpy as np
import os

def getTable(
    cap=np.inf,
    indicators =['water', 'income', 'urban', 'connection', 'ban', 'dutyp'],
    specific = True,
    municipios = list(),
    ):

    script_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_path, "DATA/")

    print(data_dir)

    if 'dutyp'in indicators or 'income' in indicators or 'urban' in indicators or ('water' in indicators and specific):
        dutyp = pd.read_csv('{}MUN_duType.csv'.format(data_dir), usecols=[0,1,2,3], index_col=0, na_values=['-', '...'])
        dutyp = dutyp.loc[dutyp.index != 'Município (Código)']
        # dutyp.loc[:, 'Apartamento, Casa'] = dutyp.ix[:, 0:2].sum(axis=1)
        # dutyp = dutyp.ix[:,2:]
        if specific: dutyp = dutyp.div(dutyp.sum(axis=1), axis=0)
        print('DU typ\t= {:0.0f}\tHouseholds\tTab: 1442'.format(dutyp.sum().sum()))

    if 'income' in indicators:
        income = pd.read_csv(
            '{}MUN_rendimento.csv'.format(data_dir),
            usecols=[0,1,3,4,5,6,7,8,9,10,11,12,14],
            index_col=0, na_values=['-', '...'])
        income = income.loc[income.index != 'Município (Código)']
        hhsize = income.sum().sum() / dutyp.sum().sum()
        print('HH size\t= {:0.2f}*'.format(hhsize))
        income = income.div(hhsize)
        # col = [
            # 'Até 1/4 de salário mínimo',            #  0
            # 'Mais de 1/4 a 1/2 salário mínimo',     #  1
            # 'Mais de 1/2 a 1 salário mínimo',       #  2
            # 'Mais de 1 a 2 salários mínimos',       #  3
            # 'Mais de 2 a 3 salários mínimos',       #  4
            # 'Mais de 3 a 5 salários mínimos',       #  5
            # 'Mais de 5 a 10 salários mínimos',      #  6
            # 'Mais de 10 a 15 salários mínimos',     #  7
            # 'Mais de 15 a 20 salários mínimos',     #  8
            # 'Mais de 20 a 30 salários mínimos',     #  9
            # 'Mais de 30 salários mínimos',          # 10
            # 'Sem rendimento',                       # 11
            #'Até 1 de salário mínimo'              # 12
            #'Mais de 1 a 5 salários mínimos'       # 13
            #'Mais de 5 a 15 salários mínimos'      # 14
            #'Mais de 15 salários mínimos'          # 15
            # ]
        # income = income.loc[:, col]
        # income.loc[:, 'Até 1 de salário mínimo']         = income.ix[:, 0:3 ].sum(axis=1)
        # income.loc[:, 'Mais de 1 a 5 salários mínimos']  = income.ix[:, 3:6 ].sum(axis=1)
        # income.loc[:, 'Mais de 5 a 15 salários mínimos'] = income.ix[:, 6:8 ].sum(axis=1)
        # income.loc[:, 'Mais de 15 salários mínimos']     = income.ix[:, 8:11].sum(axis=1)
        # income = income.ix[:,11:]
        if specific: income = income.div(income.sum(axis=1), axis=0)
        print('Income\t= {:0.0f}\tHouseholds\tTab: 3177'.format(income.sum().sum()))

    if 'urban' in indicators or ('water' in indicators and specific):
        urban = pd.read_csv(
            '{}MUN_urban.csv'.format(data_dir),
            usecols=[0,1,3],
            index_col=0, na_values=['-', '...'])
        urban = urban.loc[urban.index != 'Município (Código)']
        hhsize = urban.sum().sum() / dutyp.sum().sum()
        print('HH size\t= {:0.2f}*'.format(hhsize))
        urban = urban.div(hhsize)
        pop = urban.sum(axis=1)
        if specific: urban = urban.div(urban.sum(axis=1), axis=0)
        print('Urban\t= {:0.0f}\tHouseholds\tTab: 761'.format(urban.sum().sum()))

    if 'connection' in indicators:
        connection = pd.read_csv(
            '{}MUN_water2.csv'.format(data_dir),
            usecols=[0,2,3,4],
            index_col=0, na_values=['-', '...'])
        connection = connection.loc[connection.index != 'Município (Código)']
        # connection.loc[:, 'Outra'] = connection.ix[:,0:2].sum(axis=1)
        # connection = connection.ix[:,2:]
        if specific: connection = connection.div(connection.sum(axis=1), axis=0)
        print('Conn\t= {:0.0f}\tHouseholds\tTab: 1442'.format(connection.sum().sum()))

    if 'ban' in indicators:
        ban = pd.read_csv('{}MUN_banheiros.csv'.format(data_dir), usecols=[0,1,2,3,4,5,6], index_col=0, na_values=['-', '...'])
        ban = ban.loc[ban.index != 'Município (Código)']
        # ban.loc[:, '1-5 banheiros'] = ban.ix[:, 0:5].sum(axis=1)
        # ban = ban.ix[:, 5:]
        if specific: ban = ban.div(ban.sum(axis=1), axis=0)
        print('Toilets\t= {:0.0f}\tHouseholds\tTab: 1450'.format(ban.sum().sum()))

    if 'water' in indicators:
        water = pd.read_csv('{}MUN_water.csv'.format(data_dir), usecols=[0,6], index_col=0, na_values=['-', '...'])
        water = water.loc[water.index != 'Município (Código)']
        if specific: water = water.div(pop, axis=0)
        print('Water\t= {:0.0f}\tm^3\t\tTab: 1773'.format(water.sum().sum()))

    if 'sex' in indicators:
        sex = pd.read_csv('{}MUN_popSinopseSex.csv'.format(data_dir),
                          #usecols=[0,1,2,3,4,5,6],
                          index_col=0, na_values=['-', '...'])
        #sex = sex.loc[sex.index != 'Município (Código)']
        #sex.loc[:, '1-5 banheiros'] = ban.ix[:, 0:5].sum(axis=1)
        sex.index = [str(i) for i in sex.index]
        sex = sex.div(hhsize)
        if specific: sex = sex.div(sex.sum(axis=1), axis=0)
        print('Sex\t= {:0.0f}\tHouseholds\tTab: 3107'.format(sex.sum().sum()))

    if 'age' in indicators:
        age = pd.read_csv('{}MUN_popSinopseAge.csv'.format(data_dir),
                          #usecols=[0,1,2,3,4,5,6],
                          index_col=0, na_values=['-', '...'])
        age.index = [str(i) for i in age.index]
        hhsize = age.sum().sum() / dutyp.sum().sum()
        print('HH size\t= {:0.2f}*'.format(hhsize))
        age = age.div(hhsize)
        if specific: age = age.div(age.sum(axis=1), axis=0)
        print('Age\t= {:0.0f}\tHouseholds\tTab: 3107'.format(age.sum().sum()))

    print("* estimated values")

    #result = pd.concat([water, ban, dutyp, income, connection, waste], axis=1)
    Datasets = []
    groups = dict()
    g_num = 0
    if 'water' in indicators:
        Datasets.append(water)
        groups, g_num = add_groups(groups, 'water', water, g_num)
    if 'ban' in indicators:
        Datasets.append(ban)
        groups, g_num = add_groups(groups, 'ban', ban, g_num)
    if 'dutyp' in indicators:
        Datasets.append(dutyp)
        groups, g_num = add_groups(groups, 'dutyp', dutyp, g_num)
    if 'income' in indicators:
        Datasets.append(income)
        groups, g_num = add_groups(groups, 'income', income, g_num)
    if 'connection' in indicators:
        Datasets.append(connection)
        groups, g_num = add_groups(groups, 'connection', connection, g_num)
    if 'urban' in indicators:
        Datasets.append(urban)
        groups, g_num = add_groups(groups, 'urban', urban, g_num)
    if 'sex' in indicators:
        Datasets.append(sex)
        groups, g_num = add_groups(groups, 'sex', sex, g_num)
    if 'age' in indicators:
        Datasets.append(age)
        groups, g_num = add_groups(groups, 'age', age, g_num)
    result = pd.concat(Datasets, axis=1)

    result = result.fillna(0)
    if 'water' in indicators:
        inx = result.loc[:,'Volume total de água com tratamento'] <= cap
        result = result.loc[inx]

    if len(municipios) >= 1:
        result = result.loc[municipios]

    return(result, groups)

def add_groups(groups, name, df, g_num):
    num = df.shape[1] + g_num
    groups[name] = [i for i in range(g_num, num)]
    g_num = num
    return(groups, g_num)



def main():
    result = getTable()

    X = result.ix[:,1:]
    y = result.ix[:,0]

    model = sm.OLS(y, X)
    model_results = model.fit()
    print(model_results.summary())

    X_obs = [
        0, #'Não tinham',
        1, #'1-5 banheiro',
        1, #'Cômodo',
        0, #'Casa',
        0, #'Sem rendimento',
        0, #'Mais de 0 a 1 salários mínimos',
        0, #'Mais de 1 a 5 salários mínimos',
        0, #'Mais de 5 a 15 salários mínimos',
        1, #'Mais de 15 salários mínimos',
        1, #'Rede geral',
        0, #'Outra forma',
        0, #'Rural',
        1, #'Urban'
        ]
    estimation = model_results.predict(X_obs)

    print("estimated water demand = {:0.2f}".format(estimation[0]))

if __name__ == "__main__":
    main()

