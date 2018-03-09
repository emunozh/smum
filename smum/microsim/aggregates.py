#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Mon 19 Feb 2018 12:10:41 PM CET

"""

# system libraries
import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import newton
# internal
from smum.microsim.util import _delete_prefix, _gregwt, _script_gregwt

warnings.filterwarnings('ignore')
CATEGORICAL_LIST = ['Categorical', 'Poisson', 'Bernoulli']


class Aggregates():
    """Class containing the aggregated data.

    Args:
        pop_col (:obj:`str`, optional): Defaults to `'pop'`.
        verbose (:obj:`bool`, optional): Defaults to `False`.

    Attributes:
        pop_col (str): Population column on census data.
        verbose (bool): Be verbose.
        k (dict): Dictionary containing the k-factors.
        inverse (list): List of categories to invert.
        drop_cols (list): List of columns to drop.

    """

    def __init__(self, verbose=False, pop_col='pop'):
        """Class initiator"""
        self.pop_col = pop_col
        self.verbose = verbose
        self.k = dict()
        self.inverse = list()
        self.drop_cols = list()

    def compute_k(
            self, init_val=False,
            inter='Intercept', prefix='e_',
            year=2010, weight='wf',
            var='Electricity'):
        """Compute the k factor to minimize the error
        using the Newton-Raphson algorithm.

        Args:
            init_val (:obj:`float`, optional): Estimated initial value passed
                to the Newton-Raphson algorithm for optimizing the error value.
                Defaults to `False`. If no value is given the function will try
                get the intercept value from the `table_model`. If the function
                does not find the intercept value on the `table_model` it will
                set it to 1.
            inter (:obj:`str`, optional): Intercept name on `table_model`.
                Defaults to `'Intercept'`.
            prefix (:obj:`str`, optional): Model prefix. Defaults to `'e_'`.
            year (:obj:`int`, optional): year to be used from aggregates.
                Defaults to `2010`.
            weight (:obj:`str`, optional): Weight value to use for
                optimizations.
                Defaults to `wf`.
            var (:obj:`str`, optional): Variable to minimize the error for.
                Defaults to `'Electricity'`.

        """
        if not init_val:
            try:
                for inx in self.table_model.index.tolist():
                    if inter in inx and prefix in inx:
                        init_val = float(self.table_model.loc[inx, 'p'])
                        break
            except IndexError:
                init_val = 1
        k = newton(self._compute_error, init_val, args=(var, weight, year))
        if self.verbose:
            print("k = ", k)
        self.k[var] = k

    def _compute_error(self, k, var, weight, year):
        """compute error on marginal sums."""
        if np.isnan(self.census.loc[year, var]):
            if self.verbose:
                print("Warning: can't compute error for year {}".format(year))
                print("Error will be set to 0 for the Newton-Raphson\
                      optimization algorithm to converge")
            return 0
        var_a = self.survey.loc[:, var].mul(
            self.survey.loc[:, weight]).mul(k).sum()
        var_b = self.census.loc[year, var]
        error = (var_a - var_b)
        return error

    def print_error(self, var, weight, year=2010, lim=1e-6):
        """Print computed error to command line.

        Args:
            var (str): Variable name.
            weight (str): Weight variable to use for the estimation of error.
            year (:obj:`int`, optional): Year to compute the error for.
                Defaults to `2010`.
            lim (:obj:`float`, optional): Limit for model to converge.
                Defaults to `1e-6`.

        Returns:
            error (:obj:`float`):
                Computed error as:
                :math:`\sum_i X_{i, var} * w_i * k_{var} - Tx_{var, year}`

                Where:

                  - :math:`Tx` are the known marginal sums for variable
                    :math:`var`.
                  - :math:`X` is the generated sample survey. For each
                    :math:`i` record on the sample of variable :math:`var`.
                  - :math:`w` are the estimated new weights.
                  - :math:`k` estimated correction factor for variable
                    :math:`var`.

        """
        try:
            k = self.k[var]
        except IndexError:
            k = 1
        error = self._compute_error(k, var, weight, year)
        print("error: {:0.2E} for {}".format(error, var), end='\t')
        if error > 0:
            print("Overestimate!")
        else:
            print("Underestimate!")
        if abs(error) <= lim:
            print("Model Converged, error lover than {}".format(lim))
        return error

    def reweight(
            self, drop_cols, from_script=False, year=2010,
            max_iter=100,
            weights_file='temp/new_weights.csv', script="reweight.R",
            align_census=True,
            **kwargs):
        """Reweight survey using GREGWT.

        Args:
            drop_cols (list): list of columns to drop previous to the reweight.
            from_script (:obj:`bool`, optional): runs the reweight from a
                script.
            script (:obj:`str`, optional): script to run for the reweighting of
                the sample survey. `from_script` needs to be set to `True`.
                Defaults to `'reweight.R'`
            weights_file (:obj:`str`, optional) file to store the new weights.
                Only required if reweight is run from script.
                Defaults to `'temp/new_weights.csv'`

        """
        self.survey = self.survey.reset_index()
        self.survey = self.survey.loc[
            :, [i for i in self.survey.columns if i not in 'level_0']]

        if self.verbose:
            print('--> survey cols: ', self.survey.columns)
        inx_cols_survey = [
            col for col in self.survey.columns if
            col not in drop_cols
            and col not in 'level_0']
        toR_survey = self.survey.loc[:, inx_cols_survey]
        toR_survey = _delete_prefix(toR_survey)
        toR_survey.to_csv('temp/toR_survey.csv')
        if self.verbose:
            print('--> survey cols: ', self.survey.columns)

        if self.verbose:
            print('--> census cols: ', self.census.columns)
        inx_cols_census = [col for col in self.census.columns if
                           col not in drop_cols]
        toR_census = self.census.loc[[year], inx_cols_census]
        toR_census.insert(0, 'area', toR_census.index)
        toR_census.to_csv('temp/toR_census.csv')
        if self.verbose:
            print('--> census cols: ', toR_census.columns)

        if from_script:
            if self.verbose:
                print("calling gregwt via script")
            new_weights = _script_gregwt(
                toR_survey, toR_census, weights_file, script)
        else:
            if self.verbose:
                print("calling gregwt")
            new_weights = _gregwt(toR_survey, toR_census,
                                  verbose=self.verbose, max_iter=max_iter,
                                  align_census=align_census,
                                  **kwargs)

        if from_script:
            if new_weights.shape[0] == self.survey.shape[0]:
                new_weights = new_weights.reset_index()
                self.survey.loc[:, 'wf'] = new_weights.ix[:, -1]
            else:
                raise ValueError("weights length and sample size differ")
        else:
            if len(new_weights) == self.survey.shape[0]:
                self.survey.loc[:, 'wf'] = new_weights
            else:
                raise ValueError("weights length and sample size differ")

    def _is_inv(self, var):
        """Reverse list for inverse categories."""
        if var in self.inverse:
            inverse_e = -1
        else:
            inverse_e = 1
        return inverse_e

    def _match_keyword(self, var):
        """Match survey and census categories by keywords"""
        labels = list()
        for var_split in var.split("_"):
            if (len(var_split) > 1) & (var_split != 'cat'):
                for ms in self.census.columns:
                    if var_split.lower() in [i.lower() for i in ms.split('_')]:
                        labels.append(ms)
        if len(labels) > 1:
            return labels
        else:
            splitted = re.sub('(?!^)([A-Z][a-z]+)', r' \1', var).split()
            for var_split in splitted:
                if (len(var_split) > 1) & (var_split != 'cat'):
                    for ms in self.census.columns:
                        if var_split.lower() in [m.lower() for m
                                                 in ms.split('_')]:
                            labels.append(ms)
            return(labels)

    def _match_labels(self, labels, cat, inv=1):
        """match labels to census variables."""
        if self.verbose:
            print("\t\tmatching categories: ", cat)
        cat = np.asarray(cat)
        labels = labels[::inv]
        if self.verbose:
            print('\t\tto labels: ', labels)
        try:
            labels_out = [labels[int(i)] for i in cat]
        except IndexError:
            labels_out = list()
            for c in cat:
                this_lab = [l for l in labels if str(c) in l.split('_')]
                if len(this_lab) > 1:
                    print("Error: found more than one label for: ", c)
                else:
                    try:
                        this_lab = this_lab[0]
                    except TypeError:
                        this_lab = this_lab
                labels_out.append(this_lab)
            if self.verbose:
                print('new labels: ', labels_out)
        return labels_out

    def _get_labels(self, var, cat, inv=1):
        """Get labels for survey variables."""
        n_cat = len(cat)
        labels = self._match_keyword(var)
        if n_cat != len(labels):
            if self.verbose:
                print("\t|Warning: {} categories on marginal sums\
                      \n\t\tbut only {} on sample for variable {}".format(
                        len(labels), n_cat, var))
            labels = self._match_labels(labels, cat.tolist(), inv=inv)
            return labels
        else:
            return labels

    def _get_census_cat(self, cut, labels):
        """get census categorical variables."""
        # TODO delete function
        for e_lab in range(len(labels)):
            print(cut[e_lab], cut[e_lab + 1], labels[e_lab])
        return False

    def _survey_to_cat_single(
            self, variable_name, cut_values,
            labels=False, prefix=False, census=False):
        """Transform single variable to categorical."""
        # TODO delete function
        if self.verbose:
            print(variable_name)
        if not labels:
            labels = list()
            if prefix:
                prefix = "{}_".format(prefix)
            else:
                prefix = ''
        if census:
            var_position = [i for i in self.census.columns].index(
                variable_name)
            new_cat = self._get_census_cat(cut_values, labels)
            self.census = self.census[:, [i for i in self.census.columns
                                          if i != variable_name]]
            self.census = pd.concat(
                self.census.loc[:, :var_position],
                new_cat,
                self.census.loc[:, var_position:])
        else:
            self.survey.loc[:, variable_name] = self.survey.loc[
                :, variable_name].astype(float)
            self.survey.loc[:, variable_name] = pd.cut(
                self.survey.loc[:, variable_name],
                cut_values,
                right=False,
                labels=labels)

    def _construct_categories(self, var):
        """Construct survey categories for variable 'var'."""
        if self.verbose:
            print('\t|Distribution defined as categorical')
        inv = self._is_inv(var)
        self.survey.loc[:, var] = self.survey.loc[:, var].astype('category')
        cat = self.survey.loc[:, var].cat.categories
        if len(cat) > 1:
            labels = self._get_labels(var, cat, inv=inv)
            self.survey.loc[:, var] = self.survey.loc[
                :, var].cat.rename_categories(labels)
        else:
            # TODO Allow for variables with single category
            if self.verbose:
                print("\t\t|Single category, won't use variable: ", var)
            self.survey = self.survey.loc[
                :, [i for i in self.survey.columns if i != var]]
            columns_delete = self._match_keyword(var)
            if self.verbose:
                print("\t\t\t|will drop: ", columns_delete)
            self.drop_cols.extend(columns_delete)

    def _construct_new_distribution(self, var, dis):
        """Construct survey with specific distribution."""

        # TODO expand function to compute values given a distribution name
        # (scipy distribution name) and mu and sigma from table model

        if dis == 'Deterministic':
            if self.verbose:
                print('\t|\
                      Computing values deterministically')
            self.survey.loc[:, var] = self.coefficients.loc[:, "c_" + var]
        elif dis == 'Normal':
            if self.verbose:
                print('\t|\
                      Computing normal distributed values')
        else:
            if self.verbose:
                print('\t|\
                      Unimplemented distribution, returning as categorical')
            self._construct_categories(var)

    def _survey_to_cat(self):
        """Convert survey values to categorical values."""
        for var in self.survey.columns:
            if self.verbose:
                print("processing var: ", var, end='\t')
            try:
                dis = self.table_model.loc[var, 'dis']
                if self.verbose:
                    print('OK', dis)
            except IndexError:
                if self.verbose:
                    print('Fail!, no defined distribution.')
                dis = 'Unknown'
            if dis in CATEGORICAL_LIST:
                self._construct_categories(var)
            elif ";" in dis:
                if self.verbose:
                    print("\t|\
                          Warning: distribution <{}> of var <{}> has\
                          multiple distributions".format(dis, var))
                dis_to_use = dis.split(";")[0]
                if dis_to_use != "None" and dis_to_use is not None:
                    if self.verbose:
                        print("\t|\
                              Will use distribution <{}>".format(dis_to_use))
                    self._construct_new_distribution(var, dis_to_use)
            else:
                if self.verbose:
                    print("\t|\
                          Warning: distribution <{}> of var <{}>\
                          not defined as categorical".format(dis, var))

        if self.verbose:
            print('--> census cols: ', self.census.columns)
            print('--> will drop: ', self.drop_cols)
        self.census = self.census.loc[
            :, [i for i in self.census.columns if i not in self.drop_cols]]
        if self.verbose:
            print('--> census cols: ', self.census.columns)

    def set_table_model(self, input_table_model):
        """define table_model.

        Args:
            input_table_model (list, pandas.DataFrame): input `table_model`
                either as a list of `pandas.DataFrame` or as a single
                `pandas.DataFrame`.

        Raises:
            ValueError: if `input_table_model` is not a list of
                `pandas.DataFrame` or a single `pandas.DataFrame`.

        """
        if isinstance(input_table_model, list):
            data_frame = pd.concat(input_table_model)
        elif isinstance(input_table_model, pd.DataFrame):
            data_frame = input_table_model
        else:
            raise ValueError('can convert data type {} to table_model'.format(
                type(input_table_model)))
        self.table_model = data_frame

    def _cut_survey(self, drop=False):
        """drop unwanted columns from survey"""
        inx_data = [c for c in self.survey.columns if
                    ("c_" not in c) and
                    ("index" not in c) and
                    ("sigma" not in c) and
                    ("Intercept" not in c)
                    ]
        if drop:
            for d in drop:
                inx_data = [c for c in inx_data if d != c]
        inx_coef = [c for c in self.survey.columns if ('c_' in c)]
        self.coefficients = self.survey.loc[:, inx_coef]
        if self.verbose:
            print("_cut_survey:")
            print(self.survey.columns)
        self.survey = self.survey.loc[:, inx_data]
        if self.verbose:
            print("_cut_survey:")
            print(self.survey.columns)
        self._survey_to_cat()
        if self.verbose:
            print("_cut_survey after to_cat:")
            print(self.survey.columns)

    def set_survey(self, survey,
                   inverse=False, drop=False, to_cat=False, **kwargs):
        """define survey.

        Args:
            survey (str, pandas.DataFrame): Either survey data as
                `pandas.DataFrame` or name of a file as `str`.
            inverse (:obj:`bool`, optional): Defaults to `False`.
            drop: (:obj:`bool`, optional): Defaults to `False`.
            to_cat: (:obj:`bool`, optional): Defaults to `False`.
            **kwargs: Optional kword arguments for reading data from file,
                only used if `survey` is a file.

        Raises:
            TypeError: If `survey` is neither not a string or a DataFrame.
            ValueError: If `survey` is not a valid file.

        """
        if isinstance(survey, pd.DataFrame):
            self._set_survey_from_frame(
                survey,
                inverse=inverse, drop=drop, to_cat=to_cat)
        elif isinstance(survey, str):
            if not os.path.isfile(survey):
                raise ValueError("Can't find file {} on disk".format(survey))
            self._set_survey_from_file(
                survey,
                inverse=inverse, drop=drop, to_cat=to_cat, **kwargs)
        else:
            raise TypeError(
                "survey must be either a path, formated as str or a pandas\
                DataFrame. Got: {}".format(type(survey)))

    def set_census(self, census, total_pop=False, to_cat=False, **kwargs):
        """define census.

        Args:
            census (str, pandas.DataFrame): Either census data as
                `pandas.DataFrame` or name of a file as `str`.
            total_pop (:obj:`int`, optional): Total population.
                Defaults to `False`.
            **kwargs: Optional kword arguments for reading data from file,
                only used if `census` is a file.

        Raises:
            TypeError: If `census` is neither not a string or a DataFrame.
            ValueError: If `census` is not a valid file.

        """
        if isinstance(census, pd.DataFrame):
            self._set_census_from_frame(
                census, total_pop=total_pop, to_cat=to_cat)
        elif isinstance(census, str):
            if not os.path.isfile(census):
                raise ValueError("Can't find file {} on disk".format(census))
            self._set_census_from_file(
                census, total_pop=total_pop, to_cat=to_cat, **kwargs)
        else:
            raise TypeError("census must be either a path, formated as str or\
                            a pandas DataFrame. Got: {}".format(type(census)))

    def _set_survey_from_file(
            self, file_survey,
            inverse=False, drop=False, to_cat=False,
            **kwargs):
        """define survey from file"""
        if inverse:
            self.inverse = inverse
        self.survey = pd.read_csv(file_survey, **kwargs)
        self._cut_survey(drop=drop)
        if to_cat:
            self._add_cat(to_cat)

    def _add_cat(self, to_cat, census=False):
        """Add category to survey from dict."""
        for key in to_cat:
            if isinstance(to_cat[key], list):
                self._survey_to_cat_single(
                    key, to_cat[key][0],
                    labels=to_cat[key][1],
                    census=census)
            else:
                self._survey_to_cat_single(
                    key, to_cat[key], census=census)

    def _set_survey_from_frame(
            self, frame_survey,
            inverse=False, drop=False, to_cat=False):
        """define survey from DataFrame."""
        if inverse:
            self.inverse = inverse
        self.survey = frame_survey
        if to_cat:
            self._add_cat(to_cat)
        self._cut_survey(drop=drop)

    def _set_tot_population(self, total_pop):
        """Add total population column to census"""
        if not total_pop and self.pop_col in self.census.columns:
            if self.verbose:
                print("Warning: using total population column on file --> ",
                      self.pop_col)
        if not total_pop and self.pop_col not in self.census.columns:
            raise ValueError('need total population')
        elif total_pop and self.pop_col in self.census.columns:
            print("Warning: will overwrite total population column on census")
            self.census.loc[:, self.pop_col] = total_pop
        elif total_pop and self.pop_col not in self.census.columns:
            self.census.loc[:, self.pop_col] = total_pop

    def _set_census_from_file(
            self, file_census,
            total_pop=False, to_cat=False, **kwargs):
        """define census from file"""
        self.census = pd.read_csv(file_census, **kwargs)
        self._set_tot_population(total_pop)
        if to_cat:
            self._add_cat(to_cat, census=True)

    def _set_census_from_frame(
            self, frame_census,
            total_pop=False, to_cat=False):
        """define census from DataFrame"""
        self.census = frame_census
        self._set_tot_population(total_pop)
        if to_cat:
            self._add_cat(to_cat, census=True)


def main():
    pass


if __name__ == "__main__":
    main()
    # import doctest
    # doctest.testmod()
