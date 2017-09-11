#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 06 Sep 2017 10:52:06 AM CEST

"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def print_proj_var(col, ax1, data):
    inx = [col]
    inx.extend([str(i) for i in range(2010, 2031)])
    data_var = data.loc[:, inx].groupby(col).sum()
    data_var_sp = data_var.div(data_var.sum()).T
    data_var_sp.plot.bar(stacked=True, ax=ax1);
    years = [str(i) for i in range(2010, 2031, 2)]
    ax1.set_xticks([i for i in range(0, len(data_var_sp.index), 2)])
    ax1.set_xticklabels(years)
    ax1.set_title(col)

def _normalize(data, total_pop):
    data = data.div(data.sum(axis=1), axis=0).mul(total_pop, axis=0)
    return(data)

def print_proj_year(col, ax1, data, total_pop):
    inx = [i for i in data.columns if col == i.split('_')[0]]
    data = data.loc[:, inx]
    if isinstance(total_pop, pd.Series):
        data = _normalize(data, total_pop)
    data.plot.area(ax=ax1);
    ax1.set_title(col)

def print_all(data, sufix,
              skip = list(),
              var=True, title='', rows = 2,
              total_pop = False,
              save_data = False,
              bias = False):
    if bias:
        data, data_cols = _introduce_bias(data, bias, skip = skip, pop_col = total_pop, save_data = save_data)
    else:
        data, data_cols = _get_census_cols(data, skip = skip)
    columns = int(np.ceil(len(data_cols) / rows))
    fig, AX = plt.subplots(rows, columns, figsize=(20, 10), sharex='col', sharey='row')
    fig.suptitle(title, fontsize=16)
    i = 0
    for ax_a in AX:
        for ax in ax_a:
            col = data_cols[i]
            i +=1
            if var:
                print_proj_var(col, ax, data)
            else:
                print_proj_year(col, ax, data, total_pop)
    plt.savefig('FIGURES/proj_dist_all_{}'.format(sufix), dpi=300)
    if save_data:
        data.to_csv(save_data)
    return(data)

def _calibrate_census(census, key, vals, key_o, col, tol=0.1):
    for e, year in enumerate(census.index):
        try:
            val = vals[year]
            old_val = val
        except:
            val = old_val
        sum_year = census.loc[year, col].sum()
        tol_year = round(sum_year * tol)
        if tol_year < 1:
            tol_year = 1
        if e > 0:
            # Bias value
            share *= val # 0.1 -> 0.2
            share_o = census.loc[year, key_o] / sum_year

            new_val = round(sum_year * share)
            if new_val > sum_year:
                new_val = sum_year - tol_year
            census.loc[year, key] = new_val                      # <--- (*)

            # All other values
            share_o = share_o / share_o.sum() * (1 - share)
            new_val_o = round(share_o * sum_year)
            new_val_o[new_val_o < 0] = 0 + tol_year
            census.loc[year, key_o] = new_val_o                  # <--- (*)
        else:
            share = census.loc[year, key] / sum_year
    return(census)

def _get_census_cols(census, skip=list()):
    census = census.loc[:, [i for i in census.columns if i not in skip]]
    census_cols = census.columns
    census_cols = pd.DataFrame({"col":
        [i.split('_')[0] for i in census_cols if 'area' not in i]
    }).drop_duplicates()
    census_cols = census_cols.col.tolist()
    return(census, census_cols)

def _introduce_bias(census, bias_to, skip = list(), pop_col = 'pop', save_data = False):
    if 'pop' not in skip:
        if isinstance(pop_col, str):
            if pop_col in census.columns:
                pop = census.loc[:, pop_col]
            else:
                print('No population column found: ', pop_col)
        else:
            pop = pop_col
    skip_census = census.loc[:, skip]
    census, census_cols = _get_census_cols(census, skip = skip)
    for key, val in bias_to.items():
        e = census_cols.index(key.split('_')[0])
        col = [c for c in census.columns if census_cols[e] in c]
        key_o = [i for i in col if key != i]
        if not isinstance(val, dict):
            val = {2010:val}
        census = _calibrate_census(census, key, val, key_o, col, tol=0.001)
    census = census.join(skip_census)
    if "pop" not in skip:
        census.loc[:, 'pop'] = pop
    if save_data:
        census.to_csv(save_data)
    return(census, census_cols)

def main():
    pass


if __name__ == "__main__":
    main()
