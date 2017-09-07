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

def print_all(data, data_cols, sufix, var=True, title='', rows = 2, total_pop = False):
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

def main():
    pass


if __name__ == "__main__":
    main()
