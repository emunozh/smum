#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Thu 22 Feb 2018 11:34:10 AM CET

"""
import json
import pandas as pd
import numpy as np
import pymc3 as pm
# internal
from smum.microsim.population import PopModel
from smum.microsim.util import _make_flat_model, _delete_files


def transition_rate(
        start_rate, final_rate,
        as_array=True,
        start=False, end=False,
        default_start=2010,
        default_end=2030):
    r"""Construct growth rates.

    Args:
        model_in (dict): Model defines as a dictionary, with specified
            variables as keys, containing the `table_model` for each variable
            and an optional formula.
        err (:obj:`str`): Weight to use for error optimization.
            Defaults to `'wf'`.
        log_level (:obj:`int`, optional): Logging level passed to pymc3
            log-level.

    Returns:
        Numpy vector with transitions rates computed as a linear function.

    Example:
        >>> Elec = transition_rate(
        >>>     0, 0.4, default_start=2016, default_end=2025)

    """
    if isinstance(start, bool):
        start = default_start
    if isinstance(end, bool):
        end = default_end

    i = start - default_start
    if i < 0:
        i = 0
    rate_years = end - start + 1
    j = default_end - start - end - start

    transition_rate_v = np.linspace(start_rate, final_rate, num=rate_years)
    transition_rate_v = [0]*i + [i for i in transition_rate_v] + [final_rate]*j
    if as_array:
        transition_rate_v = np.asarray(transition_rate_v)

    return transition_rate_v


def run_calibrated_model(
        model_in,
        log_level=0,
        err='wf',
        project='reweight',
        resample_years=list(),
        rep=dict(),
        **kwargs):
    r"""Run and calibrate model with all required iterations.

    Args:
        model_in (dict): Model defines as a dictionary, with specified
            variables as keys, containing the `table_model` for each variable
            and an optional formula.
        err (:obj:`str`): Weight to use for error optimization.
            Defaults to `'wf'`.
        log_level (:obj:`int`, optional): Logging level passed to pymc3
            log-level.
        project (:obj:`str`): Method used for the projection of the sample
            survey. Defaults to `'reweight'`, this method will reweight the
            synthetic sample survey to match aggregates from the census file.
            This method is fast but might contain large errors on the resulting
            marginal sums (i.e. TAE). An alternative method is define as
            `'resample'`. This method will construct a new sample for each
            iteration and reweight it to the know aggregates on the census
            file, this method is more time consuming as the samples are created
            on each iteration via MCMC. If the variable is set to `False` the
            method will create a sample for a single year.
        rep (:obj:`dict`): Dictionary containing rules for replacing names on
            sample survey. Defaults to `dict()` i.e no modifications, empty
            dictionary.
        **kwargs: Keyword arguments passed to 'run_composite_model'.

    Returns:
        reweighted_survey (pandas.DataFrame): calibrated reweighted survey.

    Example:
        >>> elec = pd.read_csv('data/test_elec.csv', index_col=0)
        >>> inc =  pd.read_csv('data/test_inc.csv',  index_col=0)
        >>> model = {"Income":      {'table_model': inc },
                     "Electricity": {'table_model': elec}
        >>> reweighted_survey = run_calibrated_model(
                model,
                name = 'Sorsogon_Electricity',
                population_size = 32694,
                iterations = 100000)

    """

    pm._log.setLevel(log_level)
    if 'name' in kwargs:
        model_name = kwargs['name']
    else:
        print("No model name provided!")
        model_name = 'noname'

    if 'year' in kwargs:
        year_in = kwargs.pop('year')
    else:
        year_in = 2010
        print("Warning: using default year <{}> as benchmark year".format(
            year_in))

    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
        if verbose:
            print('Being verbose')
            print(year_in)
    else:
        verbose = False

    if 'align_census' in kwargs:
        align_census = kwargs['align_census']
    else:
        align_census = True

    if any([isinstance(model_in[i]['table_model'], pd.Panel)
            for i in model_in]):
        if verbose:
            print("Model define as dynamic.")
        model = _make_flat_model(model_in, year_in)
    else:
        model = model_in

    if verbose:
        for mod in model:
            print("#"*30)
            print(mod)
            print("#"*30)
            print(model[mod]['table_model'])

    k_iter = {i: 1 for i in model}
    n_models = len(model.keys()) + 1
    for e, variable in enumerate(model):
        sufix = "loop_{}".format(e+1)
        print('loop: {}/{}; calibrating: {}; sufix = {}'.format(
            e+1, n_models, variable, sufix))
        k_out, _ = run_composite_model(
            model, sufix,
            year=year_in,
            err=err,
            k=k_iter,
            **kwargs)
        k_iter[variable] = k_out[variable]

    sufix = "loop_{}".format(e+2)
    print('loop: {}/{}; final loop, for variables: {}; sufix = {}'.format(
        e+2, n_models, ", ".join([v for v in model]), sufix))
    _, reweighted_survey = run_composite_model(
        model, sufix,
        year=year_in,
        err=err,
        k=k_iter,
        **kwargs)

    print("Calibration Error:")
    for md in k_out:
        print('\t{1:0.4E}  {0:}'.format(md, 1 - k_out[md]))

    with open('temp/kfactors.json', 'w') as kfile:
        kfile.write(json.dumps(k_iter))

    census_file = kwargs['census_file']
    census = pd.read_csv(census_file, index_col=0)
    if census.shape[0] > 1 and (
            project == 'reweight' or project == 'reweighted'):
        print("Projecting sample survey for {} steps via reweight".format(
            census.shape[0]))
        out_reweighted_survey = _project_survey_reweight(
            reweighted_survey, census, model, err, rep=rep, verbose=verbose,
            align_census=align_census)
        out_reweighted_survey = out_reweighted_survey.set_index('index')
        out_reweighted_survey.to_csv("./data/survey_{}.csv".format(model_name))
    elif census.shape[0] > 1 and (
            project == 'resample' or project == 'resampled'):
        print("Projecting sample survey for {} steps via resample".format(
            census.shape[0]))
        out_reweighted_survey = _project_survey_resample(
            census, model_in, err,
            k_iter,
            resample_years=resample_years,
            **kwargs)
    else:
        out_reweighted_survey = reweighted_survey

    return out_reweighted_survey


def _project_survey_resample(
        census, model_in, err,
        k_iter,
        resample_years=list(),
        **kwargs):
    """Project reweighted survey."""
    if 'name' in kwargs:
        model_name = kwargs['name']
    else:
        print("No model name provided!")
        model_name = 'noname'
    trace_dic = dict()
    if len(resample_years) == 0:
        resample_years = census.index.tolist()
    for year in resample_years:
        print("resampling for year {}".format(year))
        sufix = "resample_{}".format(year)
        model = _make_flat_model(model_in, year)
        _, reweighted_survey = run_composite_model(
            model, sufix,
            year=year,
            err=err,
            k=k_iter,
            **kwargs)
        trace_dic[year] = reweighted_survey
        reweighted_survey.to_csv(
            "./data/survey_{}_{}.csv".format(model_name, year))
    return trace_dic


def _project_survey_reweight(
        trace, census, model_i, err,
        max_iter=100,
        verbose=False,
        align_census=True,
        rep={'urb': ['urban', 'urbanity']}):
    """Project reweighted survey."""
    census.insert(0, 'area', census.index)

    drop_survey = [i for i in model_i]
    drop_survey.append(err)

    census_cols = [i.split('_')[0].lower() for i in census.columns]
    survey_cols = list()
    skip_cols = ['i', 'e', 'HH', 'head', 'cat', 'index', 'c', 'w']
    for i in trace.columns:
        survey_cols.extend(
            [_replace(j, rep) for j in i.split('_') if j not in skip_cols])
    drop_census = [census.columns[e] for e, i in enumerate(census_cols)
                   if (i not in survey_cols or i.split('_')[0]
                       not in survey_cols) and i not in ['area', 'pop']]

    survey_in = trace.loc[
        :, [i for i in trace.columns if i not in drop_survey]]
    census = census.loc[
        :, [i for i in census.columns
            if i not in drop_census and i not in drop_survey]]

    survey_in = _delete_prefix(survey_in)
    fw = _gregwt(
        survey_in, census,
        save_path='./temp/calibrated_benchmarks_proj_{}.csv',
        complete=True, area_code='internal',
        align_census=align_census,
        max_iter=max_iter, verbose=verbose)

    a = pd.DataFrame(index=trace.index)
    for e, i in enumerate([i for i in fw.colnames if 'id' not in i]):
        a.loc[:, i] = fw.rx(True, i)

    trace_out = trace.join(a)

    return trace_out


def run_composite_model(
        model, sufix,
        population_size=False,
        err='wf',
        iterations=100,
        align_census=True,
        name='noname',
        census_file='data/benchmarks.csv',
        drop_col_survey=False,
        verbose=False,
        from_script=False,
        k=dict(),
        year=2010,
        sigma_mu=False,
        sigma_sd=0.1,
        njobs=2,
        to_cat=False,
        to_cat_census=False,
        reweight=True):
    """Run and calibrate a single composite model.

    Args:
        model (dict): Dictionary containing model parameters.
        sufix (str): model `name` sufix.
        name (:obj:`str`, optional). Model name. Defaults to `'noname'`.
        population_size (:obj:`int`, optional). Total population size.
            Defaults to 1000.
        err (:obj:`str`): Weight to use for error optimization.
            Defaults to `'wf'`.
        iterations (:obj:`int`, optional): Number of sample iterations on
            MCMC model.
            Defaults to `100`.
        census_file (:obj:`str`, optional): Define census file with aggregated
            benchmarks. Defaults to `'data/benchmarks.csv'`.
        drop_col_survey (:obj:`list`, optional): Columns to drop from survey.
            Defaults to `False`.
        verbose (:obj:`bool`, optional): Be verbose. Defaults to `False`.
        from_script (:obj:`bool`, optional): Run reweighting algorithm from
            file.
            Defaults to `Fasle`.
        k (:obj:`dict`, optional): Correction k-factor. Default 1.
        year (:obj:`int`, optional): year in `census_file` (i.e. index) to
            use for the model calibration `k` factor.
        to_cat (:obj:`bool`, optional): Convert survey variables to categorical
            variables. Default to `False`.
        to_cat_census (:obj:`bool`, optional): Convert census variables to
            categorical variables. Default to `False`.
        reweight(:obj:`bool`, optional): Reweight sample. Default to `True`.

    Returns:
        result (:obj:`list` of :obj:`objects`): Returns a list containing the
            estimated k-factors as `model.PopModel.aggregates.k` and the
            reweighted survey as `model.PopModel.aggregates.survey`.

    Examples:

        >>> k_out, reweighted_survey = run_composite_model(model, sufix)


    """
    if verbose:
        print("#"*50)
        print("Start composite model: ", name, sufix, year)
        print("#"*50)

    _delete_files(name, sufix, verbose=verbose)

    for mod in model:
        if verbose:
            print('k for {:<15}'.format(mod), end='\t')
        try:
            k[mod] = float(k[mod])
        except:
            k[mod] = 1.0
        if verbose:
            print(k[mod])
    if not all([isinstance(i, float) for i in k.values()]):
        raise TypeError('k values must be float values.')

    if verbose:
        print("Call population model")

    pop_mod = PopModel('{}_{}'.format(name, sufix), verbose=verbose)

    for mod in model:
        print("Computing model: ", mod)
        try:
            formula = model[mod]['formula']
        except:
            if verbose:
                print('no defined formula for: ', mod)
            formula = False
        pop_mod.add_consumption_model(
            mod, model[mod]['table_model'],
            formula=formula,
            sigma_mu=sigma_mu,
            sigma_sd=sigma_sd,
            prefix=mod[0].lower(),
            k_factor=k[mod])

    # define Aggregates
    pop_mod.aggregates.set_table_model(
        [pop_mod._table_model[i] for i in pop_mod._table_model])
    pop_mod.aggregates._set_census_from_file(
        census_file,
        to_cat=to_cat_census,
        total_pop=population_size,
        index_col=0)
    population_size_census = pop_mod.aggregates.census.loc[
        year, pop_mod.aggregates.pop_col]
    # run the MCMC model
    pop_mod.run_model(
        iterations=iterations,
        population=population_size_census,
        njobs=njobs)
    # define survey for reweight from MCMC
    if verbose:
        print("columns of df_trace:")
        print(pop_mod.df_trace.columns)
    pop_mod.aggregates._set_survey_from_frame(
        pop_mod.df_trace,
        drop=drop_col_survey,
        to_cat=to_cat)

    if verbose:
        print("columns of pop_mod.aggregates.survey:")
        print(pop_mod.aggregates.survey.columns)
        for mod in model:
            print("Error for mod: ", mod)
            pop_mod.aggregates.print_error(mod, "w", year=year)
    drop_cols = [i for i in model]
    pop_mod.aggregates.reweight(
        drop_cols,
        from_script=from_script,
        year=year,
        align_census=align_census)
    for mod in model:
        pop_mod.aggregates.compute_k(year=year, var=mod, weight=err)
    if verbose:
        print("columns of pop_mod.aggregates.survey after reweight:")
        print(pop_mod.aggregates.survey.columns)
        for mod in model:
            pop_mod.aggregates.print_error(mod, err, year=year)
    csvfile = pop_mod.tracefile.split('.')[0]
    csvfile += '_df.csv'
    pop_mod.aggregates.survey.to_csv(csvfile)

    return(pop_mod.aggregates.k, pop_mod.aggregates.survey)


def reduce_consumption(
        file_name, year, penetration_rate, sampling_rules, reduction,
        atol=10, verbose=False, scenario_name="scenario 1"):
    r"""
    Reduce consumption levels given a penetration rate and sampling rules.

    Args:
        file_name (str):
        year (int):
        penetration_rate (dict):
        sampling_rules (dict):
        reduction (int):
        atol (:obj:`int`, optional): Toleration level.
        verbose (:obj:`bool`, optional): Be verbose.
        scenario_name (:obj:`str`, optional): Name of the scenario. Default to:
            'scenario 1'.

    Returns:
        Pandas Data Frame with the reduced consumption levels based on
        sampling rules and penetration rates.

    Example:
        >>> sampling_rules = {
        >>>     "w_ConstructionType == 'ConstructionType_appt'": 10,
        >>>     "e_sqm < 70": 10,
        >>>     "w_Income > 13650": 10,
        >>> }
        >>> Elec = transition_rate(
        >>>     0, 0.4, default_start=2016, default_end=2025)
        >>> Water = transition_rate(
        >>>     0, 0.2, default_start=2016, default_end=2025)
        >>> pr = transition_rate(
        >>>     0, 0.3, default_start=2016, default_end=2025)
        >>> for y, p, elec, water in zip(range(2016, 2026), pr, Elec, Water):
        >>>     reduce_consumption(
        >>>         file_name,
        >>>         y, p, sampling_rules,
        >>>         {'Electricity':elec, 'Water':water},
        >>>         scenario_name = "scenario 1")

    """
    # read data
    data = pd.read_csv(file_name.format(year), index_col=0)
    data = data.loc[data.wf > 0]

    # run only is some reduction
    if all([i == 0 for i in reduction.values()]):
        print("{:05.2f}% {:^15} reduction; efficiency rate {:05.2f}%;\
              year {:04.0f} and penetration rate {:05.2f}".format(
                  0, 'both', 0, year, penetration_rate))

        year_sample = str(year) + "_{}_{:0.2f}".format(
            scenario_name, penetration_rate)
        data.to_csv(file_name.format(year_sample))

        return data

    if verbose:
        print("\tfile with {:0.0f} households".format(data.wf.sum()))
    data.loc[:, 'w'] = 1
    data.loc[:, 'sw'] = 1
    del data['index']
    data.insert(0, 'real_index', data.index)

    # Compute sampling probability weights
    max_value = 1
    old_rule = 'unnamed'
    for rule, value in sampling_rules.items():
        if rule.split('=')[0] != old_rule:
            max_value += value
        old_rule = rule.split('=')[0]
        inx = data.query(rule).index
        data.loc[inx, 'sw'] += value

    if verbose:
        if data.sw.max() == max_value:
            print('weights: OK')
        else:
            print('weights: max design p = {}, max real p = {}'.format(
                max_value, data.sw.max()))

    # Expand data
    temp_row = list()
    for i, row in data.iterrows():
        n_weight = int(np.round(row['wf']))
        for _ in range(n_weight):
            temp_row.append(row)
    temp_exp = pd.DataFrame(temp_row)
    if verbose:
        print("\tfile with {} households".format(temp_exp.w.sum()))

    if verbose:
        if temp_exp.w.sum() == np.round(data.wf).sum():
            print('expand: OK')
        else:
            print('expand: Fail')

    # get sample
    data_sample = temp_exp.sample(
        frac=penetration_rate, replace=False, weights=temp_exp.sw)

    if verbose:
        if np.allclose(
                data_sample.w.sum(),
                data.wf.sum() * penetration_rate,
                atol=atol):
            print('sampling: OK, with absolute tolerance = {}'.format(atol))
        else:
            print('sampling: Fail, with abolute tolerance = {}'.format(atol))

    # reduce consumption values
    for variable, reduction_factor in reduction.items():
        temp_old = data_sample.loc[:, variable].sum()
        data_sample.loc[:, variable] = data_sample.loc[
            :, variable].mul(1 - reduction_factor)
        temp_new = data_sample.loc[:, variable].sum()
        if verbose:
            print("{} = {:0.2f}".format(variable, 1 - (temp_new / temp_old)))

    # sample reduction
    col_group = [i for i in data_sample.columns if i not in ['w', 'sw', 'wf']]
    del data_sample['sw']
    del data_sample['wf']
    new_weights = data_sample.groupby(col_group).sum()

    new_index = new_weights.index
    for col in col_group:
        if col != 'real_index':
            new_weights.loc[:, col] = new_index.get_level_values(1).tolist()
            new_index = new_index.droplevel(1)
    new_weights.index = new_index

    if verbose:
        if np.allclose(
                new_weights.w.sum(),
                data.wf.sum() * penetration_rate,
                atol=atol):
            print('reduction: OK, with absolute tolerance = {}'.format(atol))
        else:
            print('reduction: Fail')

    if verbose:
        print("\tfile with {:0.0f} selected households".format(
            new_weights.w.sum()))

    old_values = dict()
    for variable, _ in reduction.items():
        old_val = data.loc[:, variable].mul(data.wf).sum()
        old_values[variable] = old_val

    data.loc[:, 'wf'] = data.loc[:, 'wf'].sub(
        new_weights.loc[:, 'w'], fill_value=0)
    data = data.loc[data.wf > 0]
    new_weights.loc[:, 'wf'] = new_weights.loc[:, 'w']
    data_out = pd.concat([data, new_weights])

    # reduce consumption values
    for variable, reduction_factor in reduction.items():
        old_val = old_values[variable]
        new_val = data_out.loc[:, variable].mul(data_out.wf).sum()
        print("{:05.2f}% {:^15} reduction; efficiency rate {:05.2f}%;\
              year {:04.0f} and penetration rate {:05.2f}".format(
                  (1 - (new_val / old_val)) * 100,
                  variable, reduction_factor * 100, year, penetration_rate))

    if verbose:
        print("\tfile with {:0.0f} households".format(
            data_out.wf.sum()))

    year_sample = str(year) + "_{}_{:0.2f}".format(
        scenario_name, penetration_rate)
    data_out.to_csv(file_name.format(year_sample))

    return data_out


def main():
    """ main test function."""
    pass


if __name__ == "__main__":
    main()
    # import doctest
    # doctest.testmod()
