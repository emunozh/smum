#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Mon 23 Oct 2017 03:37:26 PM CEST

"""

import numpy as np
import pandas as pd
import re

def change_index(df):
    new_index = list()
    for line in df.index:
        line = re.sub('C\(' , '', line)
        line = re.sub('[()]' , '', line)
        line = re.sub('\[.\..\]' , '', line)
        line = re.sub('\[T\.' , '_', line)
        line = re.sub('\..' , '', line)
        line = re.sub('\]' , '', line)
        line = re.sub('HH_head_', '', line)
        line = re.sub('Family_Size', 'FamilySize', line)
        line = "_".join([i for i in line.split("_")])
        new_index.append(line)
    df.index = new_index
    return(df)

def compute_categories(df):
    data_to_add = [pd.DataFrame({'ref':{'co_mu':1, 'co_sd':0}})]
    skip = list(); values = list(); values_out = list()
    row_names = list(); skip_cols = list()
    for e, col in enumerate(df.index):
        old_col = col
        col = col.split('_')[0]
        parameters = [i.split('_')[0] for i in df.index.tolist()]
        n = parameters.count(col)
        val = df.loc[old_col, ['co_mu', 'co_sd']]
        if e not in skip:
            skip = [i for i in range(e, n+e)]
            if n > 1:
                skip_cols.extend([i for i in df.index if col in i])
                row_names.append(col)
                if len(values) > 0:
                    values_df = pd.concat(values, axis=1)
                    values_df.insert(0, 'cat', [1, 1e-10])
                    co_sd = ",".join([str(i) for i in values_df.loc["co_sd"]])
                    co_mu = ",".join([str(i) for i in values_df.loc["co_mu"]])
                    values_out.append([co_sd, co_mu])
                values = list()
        if n > 1:
            values.append(val)
    values_df = pd.concat(values, axis=1)
    values_df.insert(0, 'cat', [1, 1e-10])
    co_sd = ",".join([str(i) for i in values_df.loc["co_sd"]])
    co_mu = ",".join([str(i) for i in values_df.loc["co_mu"]])
    values_out.append([co_sd, co_mu])
    values_out = pd.DataFrame(values_out, index=row_names, columns=['co_sd', 'co_mu'])
    inx = [i for i in df.index if i not in skip_cols]
    df = df.loc[inx].append(values_out)
    return(df)
