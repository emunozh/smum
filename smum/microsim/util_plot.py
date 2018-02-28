#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Mon 19 Feb 2018 02:55:39 PM CET

"""
# system libraries
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# internal
from smum.microsim.util import _merge_data, _clean_census

sns.set_context('notebook')
warnings.filterwarnings('ignore')


def plot_data_projection(reweighted_survey, var, iterations='n.a.',
                         groupby=False, pr=False, scenario_name='scenario 1',
                         verbose=False, cut_data=False,
                         col_num=1, col_width=10, aspect_ratio=4,
                         unit='household',
                         start_year=2010, end_year=2030, benchmark_year=False):
    """
    Plot projected data as total sum and as per capita.

    Args:
        reweighted_survey (str): Base survey name.
            This is the reweighted survey resulting from the simulation.
        var (list): Python list containing the name of the variables to be
            included in the plot.
        iterations (:obj:`str`, optional): Number of simulation iterations,
            this variable is only used in the title of the plot.
            Default to `'n.a.'`.
        groupby (:obj:`str`, :obj:`list`, optional):
        pr (:obj:`str`, optional):
        scenario_name (:obj:`str`, optional):
        verbose (:obj:`bool`, optional): Be verbose, Default to `False`.
        cut_data (:obj:`bool`, optional): Make quintiles, Default to `False`.
        col_num (:obj:`int`, optional): Number of columns to plot.
            Default to `1`.
        col_width (:obj:`int`, optional): Width of plot column.
            Default to `10`.
        aspect_ratio (:obj:`int`, optional): Plot aspect ratio.
            Default to `4`.
        unit (:obj:`str`, optional): y-label plot. Default to `household`.
        start_year (:obj:`int`): Plot start year, defines start of x-axis.
            Default to `2010`.
        end_year (:obj:`int`): Plot end year, defines end of x-axis.
            Default to `2030`.
        bechmark_year (:obj:`int`): Defined benchmark year,
            plots a red dotted line on benchmark year. Default to `False`.

    """
    reweighted_survey_in = reweighted_survey
    inx = [str(i) for i in range(start_year, end_year+1)]
    if not isinstance(var, list):
        raise TypeError('expected type {} for variable var, got {}'.format(
            type(list), type(var)))
    if not isinstance(pr, bool):
        pr = [i for i in pr]
    if isinstance(pr, list):
        inx_pr = ["{}_{}_{:0.2f}".format(year, scenario_name, p)
                  for year, p in zip(inx, pr)]
    else:
        inx_pr = False
    plot_num = int(len(var) / col_num)
    row_height = int(col_width / aspect_ratio * plot_num)
    fig, ax = plt.subplots(
        plot_num, col_num, figsize=(col_width, row_height), sharex=True)
    if isinstance(ax, np.ndarray):
        AX = ax
    else:
        AX = [ax]
    for v, ax in zip(var, AX):
        if verbose:
            print(v)
        if isinstance(pr, list):
            data_pr, _ = _merge_data(
                reweighted_survey_in, inx_pr, v,
                groupby=groupby, verbose=verbose)
        else:
            data_pr = False
        if isinstance(reweighted_survey, str):
            if verbose:
                print('\t| is str')
            if os.path.isfile(reweighted_survey+".csv"):
                if verbose:
                    print('\t| opening *.csv: ', reweighted_survey)
                reweighted_survey += ".csv"
                reweighted_survey = pd.read_csv(reweighted_survey, index_col=0)
            elif os.path.isfile(reweighted_survey):
                if verbose:
                    print('\t| opening: ', reweighted_survey)
                reweighted_survey = pd.read_csv(reweighted_survey, index_col=0)
        if isinstance(reweighted_survey, pd.DataFrame):
            if verbose:
                print('\t| as data frame')
            if groupby:
                data = reweighted_survey.loc[:, inx].mul(
                    reweighted_survey.loc[:, v], axis=0)
                data = data.join(reweighted_survey.loc[:, groupby])
                data = data.groupby(groupby).sum().T
            else:
                data = reweighted_survey.loc[:, inx].mul(
                    reweighted_survey.loc[:, v], axis=0).sum()
            cap = reweighted_survey.loc[:, inx].sum()
        else:
            if verbose:
                print('\t| merging data')
            data, cap = _merge_data(
                reweighted_survey, inx, v,
                groupby=groupby, verbose=verbose)
        _plot_data_projection_single(
            ax, data, v, cap, benchmark_year, iterations, groupby,
            unit=unit, cut_data=cut_data,
            data_pr=data_pr, scenario_name=scenario_name)
    var_names = "-".join(var)
    plt.tight_layout()
    plt.savefig('FIGURES/projected_{}_{}.png'.format(
        var_names,
        iterations), dpi=300)
    return(data)


def _plot_data_projection_single(ax1, data, var, cap, benchmark_year,
                                 iterations, groupby,
                                 data_pr=False, scenario_name='scenario 1',
                                 cut_data=False,
                                 unit='household'):
    """Plot projected data."""
    if groupby:
        kind = 'area'
        alpha = 0.6
    else:
        kind = 'line'
        alpha = 1
    if isinstance(data_pr, bool):
        data.plot(ax=ax1, kind=kind, alpha=alpha, label=var)
    else:
        if not groupby:
            data.plot(ax=ax1, kind=kind, alpha=alpha, label='baseline')
        data_pr.plot(ax=ax1, kind=kind, alpha=alpha, label=scenario_name)
    ax1.set_xlabel('simulation year')
    ax1.set_ylabel('Total {}'.format(var))
    if benchmark_year:
        if groupby:
            min = data.min().min()
            y = data.loc[str(benchmark_year)].sum()
            y += min
        else:
            min = data.min()
            y = data.loc[str(benchmark_year)]
            y += y - min
        x = data.index.tolist().index(str(benchmark_year))
        ax1.vlines(x, min, y, color='r')
        ax1.text(
            x+0.2, y,
            'benchmark', color='r')
    ax1.legend(loc=2)

    ax2 = ax1.twinx()

    per_husehold = data.div(cap, axis=0)
    if groupby:
        per_husehold = per_husehold.sum(axis=1)
    per_husehold.plot(style='k--', ax=ax2, alpha=0.2,
                      label='per {}\nsd={:0.2f}'.format(
                          unit, per_husehold.std()))
    ax2.set_title('{} projection (n = {})'.format(var, iterations))
    ylabel_unit = unit[0].upper() + unit[1:] + 's'
    ax2.set_ylabel('{}\n{}'.format(var, ylabel_unit))
    ax2.legend(loc=1)


# TODO Internalized into PopModel class.
def _plot_single_error(
        survey_var, census_key, survey, census, pop,
        weight='wf',
        verbose=False,
        is_categorical=True,
        save_all=False, year=2010, raw=False):
    """Plot error distrinution for single variable"""
    if survey.loc[:, survey_var].dtype == 'float64':
        Rec_s = survey.loc[:, survey_var].mul(survey.loc[:, str(weight)]).sum()
        Rec_s = pd.DataFrame({survey_var: Rec_s}, index=[year])
        # is_categorical = False
    else:
        Rec_s = survey.loc[:, [survey_var, str(weight)]].groupby(
            survey_var).sum()
    Rec_c = census.loc[[year], [c for c in census.columns if census_key in c]]
    if is_categorical:
        Rec = pd.concat([Rec_c.T, Rec_s], axis=1)
    else:
        Rec = Rec_c.join(Rec_s)
    if raw:
        return(Rec)
    if is_categorical:
        Rec_0 = Rec.loc[Rec.loc[:, str(weight)].isnull()]
        diff_0 = (Rec_0.loc[:, year]).div(pop).mul(-100)
        diff = (Rec.loc[:, str(weight)] - Rec.loc[:, year]).div(pop).mul(100)
    else:
        diff_0 = pd.DataFrame({census_key: False}, index=[year])
        diff = abs(Rec.iloc[0, 0] - Rec.iloc[0, 1])
        diff = pd.DataFrame({census_key: diff}, index=[year])
    if save_all:
        fig, ax = plt.subplots()
        diff.plot(kind='bar', ax=ax)
        if is_categorical:
            ax.set_ylabel("Error [%]")
        else:
            ax.set_ylabel("Error")
        plt.tight_layout()
        plt.savefig("FIGURES/error_{}.png".format(census_key), dpi=300)
        plt.cla()
    return(diff, diff_0)


def _get_plot_var(trace, census, skip_cols, fit_cols):
    """Get variables to plot"""
    plot_variables = dict()
    for c in trace.columns:
        if c not in skip_cols and c not in fit_cols:
            for cc in census.columns:
                for c_split in c.split('_')[1:]:
                    if c_split in cc or c_split.lower() in cc:
                        plot_variables[c] = cc.split('_')[0]
    return(plot_variables)


def plot_error(trace_in, census_in, iterations,
               pop=False,
               skip=list(),
               fit_col=list(),
               weight='wf',
               fit_cols=['Income', 'Electricity', 'Water'],
               add_cols=False,
               verbose=False,
               plot_name=False,
               is_categorical=True,
               wbins=50, wspace=0.2, hspace=0.9, top=0.91,
               year=2010, save_all=False):
    """Plot modeling errro distribution"""
    if isinstance(trace_in, str) and os.path.isfile(trace_in):
        trace = pd.read_csv(trace_in, index_col=0)
    elif isinstance(trace_in, pd.DataFrame):
        trace = trace_in
    else:
        print(trace_in)
        raise TypeError(
            'trace must either be a valid file on disc or a pandas DataFrame')
    trace = trace.loc[trace.loc[:, weight].notnull()]

    if isinstance(census_in, str) and os.path.isfile(census_in):
        if 'temp/' in census_in:
            census = _clean_census(census_in, year)
        else:
            census = pd.read_csv(census_in, index_col=0)
    elif isinstance(census_in, pd.DataFrame):
        census = census_in
    else:
        raise TypeError(
            'census must either be a valid file on disc or a pandas DataFrame')

    if isinstance(add_cols, pd.DataFrame) or isinstance(add_cols, pd.Series):
        for inx in add_cols.index:
            census.loc[year, inx] = add_cols.loc[inx]

    skip_cols = ['w', 'wf', 'level_0', 'index']
    skip_cols.extend(skip)
    skip_cols.append(year)

    # Colors
    sn_blue = sns.color_palette()[0]      # blue
    # sn_orange = sns.color_palette()[1]  # orange
    # color = sns.color_palette()[2]      # green
    sn_red = sns.color_palette()[3]       # red
    # sn_grey = sns.color_palette()[5]    # grey
    # color = sns.color_palette()[6]      # pink
    sn_grey = sns.color_palette()[7]      # grey
    # color = sns.color_palette()[8]      # yellow
    # color = sns.color_palette()[9]      # cyan
    #
    if not pop:
        pop = census.loc[year, 'pop']

    plot_variables = _get_plot_var(trace, census, skip_cols, fit_cols)
    Diff = list()
    Diff_0 = list()
    for pv in plot_variables:
        Rec, Rec_0 = _plot_single_error(
            pv, plot_variables[pv], trace, census, pop,
            verbose=verbose,
            is_categorical=is_categorical,
            save_all=save_all, year=year, weight=weight)
        Diff.append(Rec)
        Diff_0.append(Rec_0)
    if is_categorical:
        Diff = pd.concat(Diff)
        Diff_0 = pd.concat(Diff_0)
    else:
        Diff = pd.concat(Diff, axis=1).T
        Diff_0 = pd.concat(Diff_0, axis=1).T
    fit_error = list()
    fit_error_abs = list()
    for col in fit_cols:
        Rec_s = trace.loc[:, col].mul(trace.loc[:, weight]).sum()
        Rec_c = census.loc[year, col]
        Rec = (Rec_s - Rec_c) / Rec_c * 100
        fit_error.append(Rec)
        fit_error_abs.append(abs(Rec_s - Rec_c))

    fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Sampling error (year = {}, n = {})".format(
            year, iterations),
        fontsize="x-large")
    x = Diff.index.get_indexer_for(Diff_0.index)
    Diff.plot(kind='bar', label="PSAE", ax=ax, color=sn_blue)
    if Diff_0.shape[0] >= 1 and not any(Diff_0 == False):
        print('Diff_0')
        print(Diff_0)
        ax.bar(x, Diff_0, color=sn_grey, label='missing parameters', width=0.6)
        b_labels = ["{:0.2f}%".format(i) for i in Diff_0]

        for x, y, l in zip(x, Diff_0, b_labels):
            # height = rect.get_height()
            if y < 0:
                va_pos = 'top'
            else:
                va_pos = 'bottom'
            ax.text(x, y, l,
                    ha='center', va=va_pos, color='grey', alpha=0.7)

    if is_categorical:
        ax.plot(
            (0, Diff.shape[0]),
            (Diff[Diff >= 0].mean(),
             Diff[Diff >= 0].mean()),
            '--', color=sn_red,
            label='(+)PSAE = {:0.2f}% Overestimate'.format(
                Diff[Diff >= 0].mean()),
            alpha=0.4)
        ax.plot(
            (0, Diff.shape[0]),
            (Diff[Diff < 0].mean(),
             Diff[Diff < 0].mean()),
            '--', color=sn_red,
            label='(-)PSAE = {:0.2f}% Underestimate'.format(
                Diff[Diff < 0].mean()),
            alpha=0.4)
        ax.set_ylabel("PSAE [%]")
        ax.set_title("Percentage Standardized Absolute Error (PSAE)")
    else:
        ax.set_ylabel("Absolute error")
        ax.set_title("Absolute Error")
    ax.legend()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.tight_layout()
    x1 = [i for i in range(len(fit_cols))]
    ax1.bar(x1, fit_error, label='Error on fitted values', color=sn_blue)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(fit_cols, rotation='90')
    ax1.set_title("Percentage Error of Estimated Variables")
    ax1.set_ylabel("Error [%]")
    rects = ax1.patches
    labels = ["{:0.2E}\n{:0.2E}%".format(i, j)
              for i, j in zip(fit_error_abs, fit_error)]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        if not np.isnan(height):
            ax1.text(rect.get_x() + rect.get_width() / 2,
                     height / 2,
                     label,
                     ha='center', va='top', color=sn_red)
    ax1.legend(loc=4)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    rec = list()
    for pv in plot_variables:
        rec.append(
            _plot_single_error(pv, plot_variables[pv],
                               trace, census, pop, raw=True, year=year,
                               is_categorical=is_categorical,
                               verbose=verbose,
                               weight=weight))
    if is_categorical:
        REC = pd.concat(rec)
        REC.columns = ['Observed', 'Simulated']
    else:
        k = list()
        for i in rec:
            i.index = [i.columns[0]]
            i.columns = ['Observed', 'Simulated']
            k.append(i)
        REC = pd.concat(k)
    TAE = abs((REC.Observed - REC.Simulated).mean())
    PSAE = TAE / pop * 100
    rec_corr = REC.corr().iloc[0, 1]
    value_text = """
PearsonR: {:0.2f}

TAE: {:0.2E}, PSAE: {:0.2E}%
    """.format(rec_corr, TAE, PSAE)
    sns.regplot(y="Observed", x="Simulated", data=REC, ax=ax2, color=sn_blue)
    ax2.text(
        REC.Simulated.min(), REC.Observed.max(),
        value_text,
        color=sn_red,
        va='top')
    ax2.set_title("Simulated and Observed marginal sums")
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

    val, bins = np.histogram(trace.loc[:, weight], bins=wbins)
    sns.distplot(
        trace.loc[:, weight],
        bins=bins, kde=False,
        label='Distribution of estimated new weights',
        color=sn_blue,
        hist_kws={"alpha": 1},
        ax=ax3)
    ax3.set_xlabel('Estimated weights')
    ax3.plot(
        (trace.w.mean(), trace.w.mean()),
        (0, val.max()), '--', color=sn_red, alpha=1,
        label='Original sample weight')
    ax3.set_ylim(0, val.max())
    ax3.set_xlim(trace.loc[:, weight].min(), trace.loc[:, weight].max())
    ax3.text(
        trace.w.mean(), val.max() / 2,
        " <-- $d = {:0.2f}$".format(trace.w.mean()),
        color=sn_red,
        va='top')
    ax3.legend()
    ax3.set_title("Distribution of estimated new sample weights")
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    fig.subplots_adjust(wspace=wspace)
    fig.subplots_adjust(hspace=hspace)
    fig.subplots_adjust(top=top)

    if not plot_name:
        plot_name = 'sampling_error_{}_{}'.format(iterations, year)

    plt.savefig("FIGURES/{}.png".format(plot_name), dpi=300)

    return(REC)


def plot_transition_rate(
        variables, name,
        benchmark_year=2016, start_year=2010, end_year=2030,
        title="Technology transition rates for {}",
        title_p="Technology penetration rate for {}",
        ylab="Transition rate",
        ylab_p="Penetration rate",
        ylim_a=(0, 1),
        ylim_b=(0, 1)):
    """Plot growth rates."""
    p_key = ['penetration' in k for k in variables.keys()]
    p_name_keys = [k for k in variables.keys() if 'penetration' in k]
    v_name_keys = [k for k in variables.keys() if 'penetration' not in k]

    if any(p_key):
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, axl = plt.subplots(1)

    years = [int(i) for i in range(start_year, end_year + 1)]
    for k in v_name_keys:
        v = variables[k]
        v = [1 - i for i in v]
        axl.plot(years, v, label=k)

    axl.vlines(
        benchmark_year, ylim_a[0], ylim_a[1], 'r',
        linestyles='dashed', alpha=0.4, label='Benchmark year')
    axl.set_xticks(years)
    axl.set_xticklabels(years, rotation=90)
    axl.set_ylim(ylim_a)
    axl.set_title(title.format(name))
    axl.set_ylabel(ylab)
    axl.legend()

    if any(p_key):
        for r in p_name_keys:
            v = variables[r]
            axr.plot(years, v, label=r)

        axr.vlines(
            benchmark_year, ylim_b[0], ylim_b[1], 'r',
            linestyles='dashed', alpha=0.4, label='Benchmark year')
        axr.set_xticks(years)
        axr.set_xticklabels(years, rotation=90)
        axr.set_ylim(ylim_b)
        axr.set_title(title_p.format(name))
        axr.set_ylabel(ylab_p)
        axr.legend()

    plt.savefig('FIGURES/transition_rates_{}.png'.format(name), dpi=300)


def plot_projected_weights(trace_out, iterations):
    """Plot projected weights."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in trace_out.columns:
        try:
            year = int(c)
            if year >= 1900 and year < 3000:
                if c == '2016':
                    sc = 150
                    zo = 4
                else:
                    sc = 50
                    zo = 2
                ax.scatter(
                    trace_out.loc[:, c], trace_out.wf,
                    label=c, s=sc, zorder=zo)
        except TypeError:
            pass
    ax.scatter(trace_out.w, trace_out.wf, label='w', s=150, zorder=3)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
    ax.set_title(
        'Weights for each simulation year $(n = {})$'.format(iterations))
    ax.set_ylabel('Estimated final weights')
    ax.set_xlabel('Benchmarked weights to consumption $wf$')
    plt.axis('equal')
    newax = fig.add_axes([0.7, 0.7, 0.2, 0.2], zorder=10)
    inx = [str(i) for i in range(2010, 2031)]
    trace_out.loc[:, inx].mean().plot(ax=newax)
    newax.axis('off')
    newax.set_ylabel('mean weight')
    plt.tight_layout()
    plt.savefig('FIGURES/weight_mov_{}.png'.format(iterations),
                dpi=300, format='png', bbox_extra_artists=(lgd,),
                bbox_inches='tight')


def cross_tab(
        a, b, year, file_name,
        split_a=False, split_b=False, print_tab=False):
    r"""
    Print cross tabulation data.

    Args:
        a (str):
        b (str):
        year (int):
        file_name (str):
        split_a (:obj:`bool`, optional). Default `False`.
        split_b (:obj:`bool`, optional). Default `False`.
        print_tab (:obj:`bool`, optional). Default `False`.

    Returns:
        cross tabulation table.

    """
    data = pd.read_csv(file_name.format(year), index_col=0)

    lables_cat = [
        'Low',
        'mid-Low',
        'Middle',
        'mid-High',
        'High']

    if b.split('_')[-1] == 'Level' or split_b:
        data.loc[:, b] = pd.qcut(
            data.loc[:, b.split('_')[0]], 5, labels=lables_cat)

    if a.split('_')[-1] == 'Level' or split_a:
        data.loc[:, a] = pd.qcut(
            data.loc[:, a.split('_')[0]], 5, labels=lables_cat)

    data_cross = pd.crosstab(
        data.loc[:, a], data.loc[:, b], data.wf, aggfunc=sum)
    data_cross_per = data_cross.div(data_cross.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    data_cross_per.plot.bar(stacked=True, ax=ax)
    ax.set_ylabel('Share [%]')
    ax.set_title('Share of <{}> by <{}> for year {}'.format(a, b, year))
    plt.tight_layout()
    plt.savefig("FIGURES/{}_{}_{}.png".format(a, b, year), dpi=300)

    data_cross = np.round(data_cross, 2)
    if print_tab:
        print(data_cross)
    excel_file = 'data/{}_{}_{}.xlsx'.format(a, b, year)
    writer = pd.ExcelWriter(excel_file)
    data_cross.to_excel(writer, 'Sheet1')
    writer.save()
    print("data saved as: {}".format(excel_file))
    return data_cross


def main():
    pass


if __name__ == "__main__":
    main()
    # import doctest
    # doctest.testmod()
