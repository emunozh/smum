#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Mon 19 Feb 2018 12:12:23 PM CET

"""

# system libraries
import os
import warnings
from pymc3 import Model
from pymc3 import Normal, HalfNormal, Bernoulli, Beta, Bound, Poisson, Gamma
from pymc3 import Deterministic, Categorical
from pymc3 import sample, trace_to_dataframe
import pymc3 as pm
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from smum.microsim.aggregates import Aggregates
sns.set_context('notebook')
warnings.filterwarnings('ignore')

# Global variables
PosNormal = Bound(Normal, lower=0, upper=np.inf)
CATEGORICAL_LIST = ['Categorical', 'Poisson', 'Bernoulli']


class PopModel(object):
    """Main population model class."""
    def __init__(self, name='noname', verbose=False, random_seed=12345):
        """Class initiator"""

        self.distributions = dict()
        self.distributions["Normal"] = "{0} = Normal('{0}', mu={1}, sd={2}); "
        self.distributions["PosNormal"] = "{0} =\
            PosNormal('{0}', mu={1}, sd={2}); "
        self.distributions["Gamma"] = "{0} = Gamma('{0}', mu={1}, sd={2});"
        self.distributions["Bertoulli"] = "{0} = Bernoulli('{0}', {1}); "
        self.distributions["Beta"] = "{0} = Beta('{0}', mu={1}, sd={2}); "
        self.distributions["Poisson"] = "{0} = Poisson('{0}', {1}); "
        self.distributions["Categorical"] = "{0} =\
            Categorical('{0}', p=np.array([{1}])); "
        self.distributions["dt"] = "{0} = Deterministic('{0}', {1}); "

        self.name = name
        self._table_model = dict()
        self._model_bounds = dict()
        self.basic_model = Model()
        self.command = ""
        self.pre_command = ""
        self.tracefile = os.path.join(
            os.getcwd(),
            "data/trace_{}.txt".format(self.name))
        self.mu = dict()
        self.regression_formulas = dict()
        self._models = 0
        self.verbose = verbose
        self.random_seed = random_seed
        self.aggregates = Aggregates(verbose=verbose)

    def _get_distribution(self, dis, var_name, p, simple=False):
        """Get distribution."""
        if ";" in dis:
            dis = dis.split(";")[-1]
        if self.verbose:
            print('computing for distribution: ', dis)
        if 'None' in dis:
            distribution_formula = ''
        elif dis in ['Normal', 'PosNormal', 'Gamma', 'Beta']:
            distribution_formula = self.distributions[dis].format(
                var_name, p['mu'], p['sd'])
        elif dis in ['Bernoulli', 'Categorical']:
            distribution_formula = self.distributions[dis].format(
                var_name, p['p'])
        elif dis == 'Poisson':
            distribution_formula = self.distributions[dis].format(
                var_name, p['mu'])
        elif dis == 'Deterministic':
            p_in = p['p']
            if simple:
                distribution_formula = self.distributions[dis].format(
                    var_name, p_in)
            else:
                distribution_formula = "{0} =\
                    _make_theano_var({1}, 'float64');".format(
                        var_name + 'theano', p_in)
                distribution_formula += self.distributions[dis].format(
                    var_name, var_name + "theano")
        else:
            raise ValueError('Unknown or unspecified distribution: {}'.format(
                dis))
        return distribution_formula

    def _make_regression_formula(
            self, yhat_name, table_model,
            formula=False,
            constant_name='Intercept'):
        """Construct regression formula for consumption model."""
        if not formula:
            exclusion = [yhat_name, constant_name]
            estimators = list()
            for i in table_model.index:
                if i not in exclusion:
                    if table_model.loc[i, 'dis'] in CATEGORICAL_LIST:
                        estimators.append("C({})".format(i))
                    else:
                        estimators.append(i)
        else:
            elements = formula.split(',')
            estimators = []
            for e in elements:
                for f in e.split('+'):
                    for g in f.split('*'):
                        if 'c_' not in g:
                            estimators.append(g.strip())
        estimators = ' + '.join(estimators)
        regression_formula = '{} ~ {}' .format(yhat_name, estimators)

        return(regression_formula)

    def _make_categories_formula(self, p, var_name, index_var_name):
        """Construct formula for categorical variables."""
        # TODO allow for normal distribution
        list_name = "c_{}_list".format(var_name)
        self.pre_command += "{} = [{}];".format(list_name, p['co_mu'])
        self.pre_command += "{0} = _make_theano({0});".format(list_name)
        c_command = ''
        if not index_var_name:
            index_var_name = var_name
        var_1 = "_index_model({}, {})".format(index_var_name, list_name)
        var_dic = {'p': var_1}
        c_command += self._get_distribution(
            'Deterministic',
            'c_' + var_name,
            var_dic,
            simple=True)

        return c_command

    def _make_linear_model(self, constant_name, yhat_name, formula, prefix):
        """Make linear model."""
        table_model = self._table_model[yhat_name]
        linear_model = "yhat_mu_{} = ".format(self._models)
        if not formula:
            formula = constant_name+"+"+"+".join(
                ["c_{0}*{0}".format(e) for e in table_model.index if
                    (e != constant_name) &
                    (table_model.loc[e, 'dis'] != "Categorical") &
                    (table_model.loc[e, 'dis'] != "Deterministic")
                 ])
            for deter in table_model.loc[
                    table_model.dis == 'Categorical'].index:
                formula += "+c_{}".format(deter)
        linear_model += formula

        for var_name in table_model.index:
            p = table_model.loc[var_name]
            dis = str(p['dis'])
            if var_name == constant_name:
                command_var = "intercept_{} =\
                    _make_theano_var({}, 'float64');".format(
                    self._models, p['p'])
                self.pre_command += command_var
                p_in = {'p': 'intercept_{}'.format(self._models)}
                l_distribution = self._get_distribution(
                    dis, var_name, p_in, simple=True)
            else:
                l_distribution = self._get_distribution(dis, var_name, p)
            self.command += l_distribution
            try:
                dis_split = dis.split(";")
                dis = dis_split[-1].strip()
                prefix_index = dis_split[-2]
                if len(prefix_index) == 1:
                    index_var_name = prefix_index + '_' + "_".join(
                        var_name.split('_')[1:])
                else:
                    index_var_name = "{}_{}".format(prefix, prefix_index)
                if self.verbose:
                    print("Index_var_name: ", index_var_name)
                    print("var_name: ", var_name)
            except IndexError:
                index_var_name = False
            if var_name != constant_name:
                if dis == 'Deterministic':
                    c_command = ''
                elif dis != 'Categorical':
                    this_mu = p['co_mu']
                    this_sd = p['co_sd']
                    c_command = self.distributions['Normal'].format(
                        'c_' + var_name, this_mu, this_sd)
                else:
                    c_command = self._make_categories_formula(
                        p, var_name, index_var_name)
            else:
                c_command = ''
            self.command += c_command

        return(linear_model)

    def print_command(self):
        """Print computed command."""
        print("The define model will be executed with the following commands:")
        print(self.pre_command.replace(';', '\n'))
        print(self.command.replace('; ', '\n'))

    def add_consumption_model(self, yhat_name, table_model,
                              k_factor=1,
                              sigma_mu=False,
                              sigma_sd=0.1,
                              prefix=False,
                              bounds=[-np.inf, np.inf],
                              constant_name='Intercept',
                              formula=False):
        """Define new base consumption model.
        """
        if prefix:
            constant_name = "{}_{}".format(prefix, constant_name)
        self.regression_formulas[yhat_name] = self._make_regression_formula(
            yhat_name, table_model,
            formula=formula, constant_name=constant_name)
        self._table_model[yhat_name] = table_model
        self._model_bounds[yhat_name] = bounds
        self._models += 1
        self.mu[yhat_name] = sigma_mu
        linear_model = self._make_linear_model(
            constant_name, yhat_name, formula, prefix)
        self.command += linear_model
        self.command += '; '
        self.command += "yhat_mu_{0} *= _make_theano({1}); ".format(
            self._models, k_factor)
        if sigma_mu:
            # Estimate var consumption with a normal distribution
            self.command += self.distributions['Normal'].format(
                'sigma_{}'.format(self._models), sigma_mu, sigma_sd)
            self.command += self.distributions['Normal'].format(
                yhat_name,
                'yhat_mu_{}'.format(self._models),
                'sigma_{}'.format(self._models))
        else:
            # Estimate deterministically
            self.command += self.distributions['Normal'].format(
                yhat_name,
                'yhat_mu_{}'.format(self._models)
            )

    def run_model(
            self, iterations=100000, population=False,
            burn=False, thin=2, njobs=2, **kwargs):
        """Run the model."""
        if not burn:
            burn = iterations * 0.01
        if not population:
            population = iterations
        iterations += burn
        iterations *= thin
        if self.verbose:
            print('will save the data to ', self.tracefile)

        if self.verbose:
            print(self.pre_command.replace(';', "\n"))
            print(self.command.replace('; ', "\n"))

        with self.basic_model:
            exec(self.pre_command)

        if self.verbose:
            print("Start model")

        with self.basic_model:
            exec(self.command)

        if self.verbose:
            print('Staring sampling')

        with self.basic_model:
            # use text as a backend
            backend = pm.backends.Text(self.tracefile)

            # Calculate the trace
            self.trace = sample(
                draws=int(np.ceil(iterations / njobs)),
                njobs=njobs,
                trace=backend,
                random_seed=self.random_seed,
                **kwargs
            )

        # Transform data to DataFrame
        self.trace = self.trace[int(burn)::int(thin)]
        self.df_trace = trace_to_dataframe(self.trace)
        new_col = [col.replace('__0', '') for col in self.df_trace.columns]
        self.df_trace.columns = new_col

        # Truncate values
        self._truncate()
        self.df_trace = self.df_trace.dropna()
        self.df_trace = self.df_trace.reset_index()

        # Add initial weight to trace
        weight_factor = population / self.df_trace.shape[0]
        if self.verbose:
            print(weight_factor)
        self.df_trace.loc[:, 'w'] = weight_factor

        # Save values
        self._save_trace()

    def _save_trace(self):
        """Save trace as csv file."""
        csvfile = self.tracefile.split('.')[0]
        csvfile += '.csv'
        self.df_trace.to_csv(csvfile)

    def _truncate(self):
        """Truncate distributions."""
        for yhat_name in self._table_model:
            table_model = self._table_model[yhat_name]
            bounds = self._model_bounds[yhat_name]
            if not np.isnan(bounds[0]) or not np.isnan(bounds[1]):
                self._truncate_single(table_model, yhat_name, bounds)
            for variable in table_model.index:
                var_bounds = table_model.loc[variable, ['lb', 'ub']]
                if not np.isnan(var_bounds['lb'])\
                        or not np.isnan(var_bounds['ub']):
                    self._truncate_single(
                        table_model, variable, var_bounds.tolist())

    def _truncate_single(self, table_model, variable, bounds):
        """Truncate trace values."""
        truncated_trace = self.df_trace
        if variable in truncated_trace.columns:
            if self.verbose:
                print("bounds: {} for var: {}".format(bounds, variable))
                print("start size:", self.df_trace.shape[0])
                print("mean: {:0.2f}".format(
                    self.df_trace.loc[:, variable].mean()))
            inx = (truncated_trace.loc[:, variable] >= bounds[0]) &\
                  (truncated_trace.loc[:, variable] <= bounds[1])
            truncated_trace = truncated_trace.loc[inx]
        else:
            if self.verbose:
                print("can't find variable: {} on trace".format(variable))
        self.df_trace = truncated_trace
        if variable in truncated_trace.columns:
            if self.verbose:
                print("final size: ", self.df_trace.shape[0])
                print("mean: {:0.2f}".format(
                    self.df_trace.loc[:, variable].mean()))

    def plot_model(self):
        """Model traceplot."""
        from pymc3 import traceplot
        traceplot(self.trace)
        plt.show()

    def plot_model_test(self, yhat_name, mu_var, sd_var):
        """Model test plot"""
        data = self.df_trace.loc[:, yhat_name]
        x_var = np.arange(data.min(), data.max())
        pdf_fitted = stats.norm.pdf(x_var, mu_var, sd_var)
        g = sns.distplot(data, label="Posterior (MCMC)")
        g.set_ylabel('households')
        g.set_xlabel(yhat_name)
        ax = g.twinx()
        ax.plot(
            pdf_fitted, color='r',
            label="Normal Distribution:\
                   \n$\mu={:0.0f}$; $\sigma={:0.0f}$".format(mu_var, sd_var))
        ax.set_ylabel("Density")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = g.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc=0)
        plt.show()


def main():
    pass


if __name__ == "__main__":
    main()
    # import doctest
    # doctest.testmod()
