#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Mon 19 Feb 2018 12:23:47 PM CET

"""

# system libraries
import os
import shutil
import warnings
import copy
import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
import seaborn as sns
import theano.tensor as T
from theano import config, shared
config.warn.round = False
# rpy2 libraries
try:
    from rpy2.robjects.vectors import IntVector
    from rpy2.robjects import DataFrame, pandas2ri
    pandas2ri.activate()
    from rpy2.robjects.packages import importr
    UTILS_PACKAGE = importr("utils")  # import utils package from R
except ImportError:
    print("can't find rpy2 libray! GREGWT won't run!")
    IntVector = 'Null'
    DataFrame = 'Null'
    pandas2ri = 'Null'
    importr = 'Null'
    UTILS_PACKAGE = 'Null'

sns.set_context('notebook')
warnings.filterwarnings('ignore')

GREGWT_ITERATION = 0


def _get_breaks(census_col, verbose=False):
    """Compute census breaks"""
    old_var_name = None
    breaks = list()
    for e, col in enumerate(census_col):
        var_name = col.split('_')[0]
        if var_name != old_var_name:
            if e-1 > 0:
                if verbose:
                    print("adding {} as break = {}".format(var_name, e-1))
                breaks.append(e-1)
            old_var_name = var_name
    breaks = breaks[:-1]
    breaks_r = IntVector(breaks)
    return breaks_r


def _toR_df(toR_df):
    """Convert pandas DataFrame to R data.frame."""
    for e, col in enumerate(toR_df.columns):
        col_dtype = toR_df.iloc[:, e].dtype
        if isinstance(col_dtype, CategoricalDtype):
            toR_df.iloc[:, e] = toR_df.iloc[:, e].astype(str)
    col = toR_df.columns.tolist()
    if 'Unnamed' in col[0]:
        col[0] = "X"
    toR_df.columns = col
    new_index = [i for i in range(1, toR_df.shape[0]+1)]
    toR_df.index = new_index
    # toR_df = toR_df.dropna()
    toR_df = pandas2ri.py2ri(toR_df)
    return(toR_df, col)


def _script_gregwt(survey, weights_file, script):
    """Run GREGWT directly from an R script"""
    res = os.system("Rscript {}".format(script))
    if res != 0:
        raise ValueError("cannot run script: {}".format(script))
    new_weights = pd.read_csv(weights_file, index_col=0)
    return new_weights


def _delete_prefix(toR_survey):
    # delete prefix from colum names
    new_survey_cols = list()
    for sc in toR_survey.columns:
        names = sc.split("_")
        if len(names) > 1:
            if len(names[0]) == 1:
                joint_names = "_".join(names[1:])
            else:
                joint_names = "_".join(names)
        else:
            joint_names = names[0]
        new_survey_cols.append(joint_names)
    toR_survey.columns = new_survey_cols
    return toR_survey


def _align_var(breaks_r, pop_col, n, verbose=False):
    prev_b = -1
    i = 1
    align = dict()
    align_t = [1]
    for e, b in enumerate(breaks_r):
        if prev_b + 1 == b and b < n+1:
            try:
                assert(min(align_t) != max(align_t))
                align[pop_col + '.' + str(i)] = IntVector(
                    (min(align_t), max(align_t)))
            except AssertionError:
                if verbose:
                    print("can't allign {} at {} for {}".format(align_t, e, b))
                pass
            i += 1
            align_t = [e+2]
        else:
            align_t.append(e+2)
        prev_b = b
    if len(align) == 0:
        align[pop_col] = IntVector((1, len(breaks_r)))
    align_r = DataFrame(align)
    return align_r


def _gregwt(
        toR_survey, toR_census,
        save_path='./temp/calibrated_benchmarks_{}.csv',
        pop_col='pop',
        verbose=False,
        log=True,
        complete=False,
        survey_index_col=True,
        census_index_col=True,
        align_census=True,
        survey_weights='w',
        **kwargs):
    """GREGWT."""
    global GREGWT_ITERATION
    GREGWT_ITERATION += 1
    save_path = save_path.format(GREGWT_ITERATION)

    if survey_index_col:
        sic = 2
    else:
        sic = 1
    if census_index_col:
        cic = 2
    else:
        cic = 1
    toR_census, census_col = _toR_df(toR_census)
    if verbose:
        print("census cols: ", census_col)
    if align_census:
        breaks_r_in = _get_breaks(census_col, verbose=verbose)
    else:
        breaks_r_in = list()
    if len(breaks_r_in) == 0:
        breaks_r = False
        convert = False
    else:
        breaks_r = breaks_r_in
        convert = True
    if verbose:
        print("breaks: ", breaks_r)
    toR_survey, survey_col = _toR_df(toR_survey)
    if verbose:
        print("survey cols: ", survey_col)
    if isinstance(breaks_r, bool):
        align_r = False
    else:
        align_r = _align_var(
            breaks_r, pop_col, toR_census.ncol, verbose=verbose)
    if verbose:
        print("align: ", align_r)
    census_cat_r = IntVector(
        [e+1 for e, i in enumerate(census_col)
         if i != pop_col and e+1 >= cic])
    if verbose:
        print("census cat: ", census_cat_r)
        print("census col names: ", toR_census.colnames)
    survey_cat_r = IntVector(
        [e+1 for e, i in enumerate(survey_col)
         if i != survey_weights and e+1 >= sic])
    if verbose:
        print("survey cat: ", survey_cat_r)
        print("survey col names: ", toR_survey.colnames)

    gregwt = importr('GREGWT')

    # (1) prepare data
    simulation_data = gregwt.prepareData(
        toR_census, toR_survey,
        verbose=verbose,
        align=align_r,
        breaks=breaks_r,
        survey_weights=survey_weights,
        convert=convert,
        pop_total_col=pop_col,
        census_categories=census_cat_r,
        survey_categories=survey_cat_r)

    UTILS_PACKAGE.write_csv(simulation_data.rx2('Tx_complete'), save_path)

    # (2) reweight
    new_weights = gregwt.GREGWT(
        data_in=simulation_data,
        use_ginv=True,
        verbose=verbose,
        output_log=log,
        **kwargs)

    fw = new_weights.rx2('final_weights')
    if not complete:
        fw = fw.rx(True, sic)

    return fw


def _delete_files(name, sufix, verbose=False):
    files = "{0}_{1}_{2}.csv|{0}_{1}_{2}.sqlite|{0}_{1}_{2}.txt".format(
        './data/trace', name, sufix)
    for file in files.split("|"):
        if os.path.isfile(file):
            if verbose:
                print("delete: ", file)
            os.remove(file)
        elif os.path.isdir(file):
            if verbose:
                print("delete: ", file)
            shutil.rmtree(file)
        else:
            if verbose:
                print("no file: ", file)


def _merge_data(
        reweighted_survey, inx, v,
        group=False, groupby=False, verbose=False):
    values = dict()
    cap = dict()
    for i in inx:
        file_survey = reweighted_survey + "_{}.csv".format(i)
        survey_temp = pd.read_csv(file_survey, index_col=0)
        if verbose:
            print("\n\t\t| year: ", i)
            print('\t\t| file: ', file_survey)
        cap_i = survey_temp.wf.sum()
        if group:
            if verbose:
                print('\t\t| group: ', group)
            val = survey_temp.loc[:, [v, 'wf']].fillna(group).groupby(v).size()
        else:
            if verbose:
                print('\t\t| no group')
            survey_temp.loc[:, v] = survey_temp.loc[:, v].mul(
                survey_temp.loc[:, 'wf'])
            if not groupby:
                if verbose:
                    print('\t\t| un-grouped')
                val = survey_temp.loc[:, v].sum()
            else:
                if not isinstance(groupby, list):
                    groupby = [groupby]
                inx_g = groupby.copy()
                inx_g.append(v)
                if verbose:
                    print('\t\t| groupby: ', groupby, inx_g)
                val = survey_temp.loc[:, inx_g].groupby(groupby).sum()
                if verbose:
                    print('\t\t| val: ', val)
                val.columns = [i]
        values[i] = val
        cap[i] = cap_i
    if group:
        s = pd.concat([it for k, it in values.items()], axis=1)
        s.columns = values.keys()
        s = s.fillna(1)
    elif groupby:
        values_cat = list()
        for key, df in values.items():
            values_cat.append(df)
        s = pd.concat(values_cat, axis=1)
        s = s.T
    else:
        s = pd.Series(values, name=v)
        s.index.name = 'year'
    if any(s.isnull()):
        s = s.interpolate(method='linear', axis=0)
    c = pd.Series(cap, name='pop')
    c.index.name = 'year'
    return(s, c)


def _replace(j, rules):
    j = j.lower()
    for k, r in rules.items():
        j = j.replace(r[0], r[1])
    return j


def _make_flat_model(model, year):
    """make model flat."""
    model_out = copy.deepcopy(model)
    for mod in model_out:
        model_out[mod]['table_model'] = model_out[mod]['table_model'].loc[year]
    return model_out


def _make_theano(co_mu_list):
    co_mu_list = shared(
        np.asarray(co_mu_list, dtype=config.floatX),
        borrow=True)
    co_mu_list = co_mu_list.flatten()
    co_mu_list = T.cast(co_mu_list, 'float64')
    return co_mu_list


def _make_theano_var(variable, var_typ):
    if var_typ == 'float64':
        variable = np.float64(variable)
    var_out = T.cast(variable, var_typ)
    return var_out


def _index_model(inx, co_mu_list):
    res = co_mu_list[inx]
    return res


def _to_str(x):
    if isinstance(x, str):
        y = x.split(",")
        y = [np.round(float(i), 2) for i in y]
        y_out = "{}, ..., {}".format(y[0], y[-1])
        return y_out
    elif isinstance(x, float):
        return np.round(x, 2)
    elif isinstance(x, int):
        return int(x)
    else:
        return np.nan


def _format_table_model(tm, year, var, verbose=False):
    df = tm.models[var].loc[[year], :, :].to_frame().unstack()
    df.columns = df.columns.droplevel(0)
    df.columns.name = ''
    df.index.name = ''

    skip = [i for i in df.loc[:, 'dis']
            if i is not None and 'Categorical' in i]
    select_a = [i not in skip for i in df.loc[:, 'dis']]
    select_b = [i in skip for i in df.loc[:, 'dis']]
    inx = [c for c in df.columns if c != 'dis']
    a = df.loc[select_a, inx].astype(float)
    b = df.loc[select_b, inx]
    c = df.loc[:, ['dis']]

    for i in b.index:
        g = b.loc[i].apply(lambda x: _to_str(x))
        a = a.append(g)

    if a.shape[1] < 5:
        a = a.join(c)
    else:
        a.insert(5, 'dis', c)

    return a


def _to_excel(df, year, var, writer, **kwargs):

    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name=str(year), **kwargs)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[str(year)]

    # Add some cell formats.
    format_name = workbook.add_format({'num_format': '@'})
    format_variable = workbook.add_format({'num_format': '#,##0.00'})
    format_variable_2 = workbook.add_format({'num_format': '#,##0'})
    format_dis = workbook.add_format({'num_format': '@'})

    # Set the column width and format.
    worksheet.set_column('A:A', 20, format_name)
    worksheet.set_column('B:F', 10, format_variable)
    worksheet.set_column('G:G', 16, format_dis)
    worksheet.set_column('H:I', 10, format_variable_2)


def _clean_census(census_file, year):
    census = pd.read_csv(census_file, index_col=0)
    census.columns = [i.replace('G.', '') for i in census.columns]
    census.index = [year]
    return census


def main():
    pass


if __name__ == "__main__":
    main()
    # import doctest
    # doctest.testmod()
