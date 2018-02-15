#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Thu 15 Feb 2018 10:07:34 AM CET

"""

import pandas as pd
import random
import os

fn_tabula = os.path.join(os.path.dirname(__file__), 'data/tabula.csv')
tabula = pd.read_csv(fn_tabula, index_col=[0,1,2])
fn_den = os.path.join(os.path.dirname(__file__), 'data/densities.csv')
densities = pd.read_csv(fn_den, index_col=[0,1])

def _clean_year(y):
    years = []
    flip = False
    highest = 0
    for i in y.split("'"):
        if 'post' in i: flip = True
        if len(i) >= 1:
            i = i.replace('-', '')
            i = i.replace('pre', '-inf')
            i = i.replace('post', 'inf')
            i = float(i)
            if i >= 40:
                i += 1900
            else:
                i += 2000
            if i <= highest:
                i += 1
            highest = i
            years.append(i)
    if flip:
        years = years[::-1]
    return years

year_ranges = [_clean_year(i) for i in tabula.index.get_level_values(0).unique()]

def _get_year_lable(year, year_ranges, tabula):
    inx_year = [e for e, i in enumerate(year_ranges) if year >= i[0] and year <= i[1]][0]
    year_lable = tabula.index.get_level_values(0).unique()[inx_year]
    return year_lable

def _get_typ(year, construction, sqm, tabula = tabula):
    if construction == "SFH" or construction == 1:
        construction = "SFH "
    elif construction == "MFH" or construction == 2:
        construction = "MFH "
    con_year = _get_year_lable(2006, year_ranges, tabula)
    temp = tabula.loc[(con_year, construction, slice(None)), "Floor surface area per housing unit (m2)"]
    p = random.random()
    temp_p = 1 - (abs(temp - sqm) / sum(abs(temp - sqm)))
    temp_sel = temp_p.loc[temp_p >= p]
    if temp_sel.shape[0] > 1:
        temp_sel = temp_sel.loc[temp_sel == temp_sel.min()]
    elif temp_sel.shape[0] == 0:
        temp_sel = temp_p.loc[temp_p == temp_p.max()]
    if temp_sel.shape[0] == 0:
        print('error')
    btyp = "{} {}".format(
        temp_sel.index.get_level_values(1)[0].strip(),
        temp_sel.index.get_level_values(2)[0].strip())
    return(btyp)

def _get_den(year, p, densities = densities):
    cyear_inx = densities.index.get_level_values(1).unique()
    year_diff = [abs(i - year) for i in cyear_inx]
    sel_inx = [e for e, i in enumerate(year_diff) if i == min(year_diff)][0]
    den = densities.loc[(p, cyear_inx[sel_inx])]
    return den

def get_den(year, construction, sqm, tabula = tabula, densities = densities):
    """Get material densities based on construction year, canstruction type and sqm."""
    p = _get_typ(year, construction, sqm, tabula = tabula)
    d = _get_den(year, p, densities = densities)
    return d


def main():
    sqm = 210
    year = 1946
    construction = "SFH"
    d = get_den(year, construction, sqm)
    print(d)
    print(d.loc['Minerals'])

def main_2():
    sqm = 210
    year = 1946
    construction = "SFH"
    random.seed(1234)
    print("Tab. Material intensities of selected building typologies{sep}\
          for construction year:\t{}{sep}\
          and construction type:\t{}{sep}\
          with a flor area of:\t\t{} m^2".format(
                  year, construction, sqm, sep='\n'))
    print("-" * 60)
    print("| {:<20} | {:^6} | {:^6} | {:^6} | {:^6} |".format(
        'Building Type', "Metals", "Mine.", "Plas.", "Wood"))
    print("=" * 60)
    for i in range(20):
        p = _get_typ(year, construction, sqm, tabula = tabula)
        d = _get_den(year, p, densities = densities)
        print("| {:<20} | {} |".format(p, " | ".join(["{:0.4f}".format(i) for i in d])))
    print("-" * 60)

if __name__ == "__main__":
    main()


