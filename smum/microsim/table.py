#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Mon 19 Feb 2018 12:13:45 PM CET

"""

# system libraries
import os
import re
import math
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


class TableModel(object):
    """Static and dynamic table model."""
    def __init__(
            self,
            census_file=False,
            verbose=False,
            normalize=True):

        self.normalize = normalize

        if census_file:
            if not os.path.isfile(census_file):
                raise ValueError("{} not a file".format(census_file))
            self.census = pd.read_csv(census_file, index_col=0)
            self.dynamic = True
        else:
            self.dynamic = False

        self.models = dict()
        self.formulas = dict()
        self.skip = ['cat', 'Intercept']
        self.verbose = verbose
        if self.verbose:
            print('--> census cols: ', self.census.columns)
        self.ref_cat = "Undefined"

    def to_excel(self, sufix='', var=False, year=False, **kwargs):
        """Save table model as excel file."""
        from smum.microsim.util import _format_table_model, _to_excel
        if isinstance(var, str):
            var = [var]
        else:
            var = [i for i in self.models.keys()]
        if not isinstance(year, bool):
            year = [year]
        for v_var in var:
            # Create a Pandas Excel writer using XlsxWriter as the engine.
            file_name = "data/tableModel_{}{}.xlsx".format(v_var, sufix)
            writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
            print('creating', file_name)
            if isinstance(year, bool):
                year = self.models[v_var].axes[0].tolist()
            for y_year in year:
                df_table_model = _format_table_model(
                    self, y_year, v_var, verbose=self.verbose)
                _to_excel(df_table_model, y_year, v_var, writer, **kwargs)

            # Close the Pandas Excel writer and output the Excel file.
            writer.save()

    def make_model(self):
        """prepare model for simulation."""
        model_out = dict()
        for name in self.models:
            model_out[name] = {'table_model': self.models[name]}
            try:
                formula = self.formulas[name]
                model_out[name]['formula'] = formula
            except IndexError:
                pass
        return model_out

    def print_formula(self, name):
        """pretty print table_model formula."""
        print(name, "=")
        for formula in self.formulas[name].split('+'):
            print("\t", formula, "+")

    def add_formula(self, formula, name):
        """add formula to table model."""
        self.formulas[name] = formula

    def add_model(
            self, table, name,
            index_col=0, reference_cat=[],
            static=False,
            skip_cols=list(), **kwargs):
        """Add table model."""
        if self.verbose:
            print('adding {} model'.format(name), end=' ')
        table = pd.read_csv(table, index_col=index_col, **kwargs)

        self.skip.extend(skip_cols)

        # add prefix to table index based on table name
        new_index = list()
        for i in table.index:
            prefix = i.split('_')[0]
            if len(prefix) > 1:
                prefix = name[0].lower() + "_"
                j = prefix + i
                new_index.append(j)
            else:
                new_index.append(i)
        table.index = new_index

        self.ref_cat = reference_cat
        if self.verbose:
            if self.ref_cat:
                print('with reference categories: ', ','.join(self.ref_cat),
                      end='')
                print('. ', end='')
            else:
                print('', end='')

        self.models[name] = table
        if self.dynamic:
            if self.verbose:
                print("as dynamic model.")
            self.update_dynamic_model(name, static=static)
        else:
            if self.verbose:
                print("as static model.")

    def update_dynamic_model(
            self, name,
            val='p',
            static=False,
            sd_variation=0.01,
            select=False,
            specific_col_as=False,
            specific_col=False,
            compute_average=True):
        """Update dynamic model."""
        if val == 'mu' and compute_average:
            compute_average = 0
        table = self.models[name]
        if specific_col:
            if specific_col_as:
                temp_col = specific_col
                specific_col = specific_col_as
                specific_col_as = temp_col
            if self.verbose:
                print("\t| for specific column {}".format(specific_col))
            v_cols = [i for i in self.census.columns if
                      specific_col == i.split('_')[0] or
                      specific_col.lower() == i.split('_')[0]]
            if len(v_cols) == 0:
                v_cols = [i for i in self.census.columns if
                          specific_col == i.split('_')[-1] or
                          specific_col.lower() == i.split('_')[-1]]
            if self.verbose:
                print('\t| specific col: ', end='\t')
                print(v_cols)
        else:
            if self.verbose:
                print("\t| for all columns:")
            v_cols = self._get_cols(table, static=static)
            if self.verbose:
                for col in v_cols:
                    print("\t\t| {}".format(col))

        if len(v_cols) == 0:
            return False

        panel_dic = dict()
        for year in self.census.index:
            if self.verbose:
                print("\t|", year, end="\t")
            if isinstance(table, pd.DataFrame):
                if self.verbose:
                    print('| table is DataFrame')
                this_df = table.copy()
            elif isinstance(table, pd.Panel):
                if self.verbose:
                    print('| table is Panel')
                this_df = table[year].copy()
            else:
                print('unimplemented type: ', type(table))

            this_df = self._update_table(
                this_df, year,
                v_cols, val, name, specific_col,
                specific_col_as,
                compute_average,
                select=select,
                sd_variation=sd_variation,
                static=static)
            panel_dic[year] = this_df

        self.models[name] = pd.Panel(panel_dic)

    def _update_table(self, this_df, year, v_cols, val,
                      name, specific_col,
                      specific_col_as,
                      compute_average,
                      sd_variation=0.01,
                      select=False,
                      static=False):
        new_val = False
        prefix = name[0].lower()
        sd_default = 0
        if specific_col:
            e1, _, _ = self._get_positions(
                specific_col, specific_col_as)
            if not isinstance(compute_average, bool):
                new_val, sd_default = self._get_weighted_mean(v_cols, year)
                new_val += compute_average
                if self.verbose:
                    print('\t\t\t| computed average:')
                    print('\t\t\t| {}_{:<18} {:0.2f}'.format(
                        prefix, e1, new_val))
            else:
                if static:
                    new_val = self.census.loc[year, v_cols]
                else:
                    new_val = self.census.loc[year, v_cols].div(
                        self.census.loc[year, 'pop'])
                if val == 'mu':
                    new_val = new_val[0]
                    sd_default = new_val * sd_variation
                    if self.verbose:
                        print('\t\t\t| absolute values:')
                        print('\t\t\t| {}_{:<18} {}'.format(
                            prefix, e1, new_val))
                else:
                    new_val = ','.join([str(i) for i in new_val.values])
                    if not isinstance(select, bool):
                        new_val = new_val.split(',')
                        new_val = [float(i) for i in new_val]
                        new_val = [i / np.sum(new_val) for i in new_val]
                        new_val = new_val[select]
                    if self.verbose:
                        print('\t\t\t| categorical values:')
                        print('\t\t\t| {}_{:<18} {}'.format(
                            prefix, e1, new_val))
            try:
                this_df.loc['{}_{}'.format(prefix, e1), val] = new_val
                if val == 'mu':
                    this_df.loc['{}_{}'.format(prefix, e1), 'sd'] = sd_default
            except IndexError:
                print('Warning: could not assing new \
                       value to data set on {}_{}'.format(prefix, e1))
            return this_df
        else:
            for e in v_cols:
                e1, e2, sufix = self._get_positions(e, specific_col_as)
                if static:
                    if self.verbose:
                        print('\t\t\t| static')
                    return(this_df)
                else:
                    if self.normalize:
                        if self.verbose:
                            print('\t\t\t| normalize')
                        val_a = self._normalize(e, year)
                        val_a = val_a['{}_{}'.format(e2, sufix)]
                    else:
                        val_a = self.census.loc[year, '{}_{}'.format(
                            e2, sufix)]
                    val_b = self.census.loc[year, 'pop']
                    new_val = val_a / val_b
                    try:
                        this_df.loc['{}_{}'.format(prefix, e1), val] = new_val
                    except IndexError:
                        print('Warning: could not assing new \
                              value to data set on {}_{}'.format(prefix, e1))
                    if self.verbose:
                        print('\t\t\t|\
                              {}_{:<18} {:8.2f} / {:8.2f} = {:0.2f}'.format(
                                prefix, e1,
                                val_a, val_b,
                                new_val,
                                ), end='  ')
                        print('| {}_{}'.format(e2, sufix))
        return this_df

    def _normalize(self, e, year):
        census_cols = [c for c in self.census.columns if
                       e.lower() in [i.lower()
                                     for i in c.split("_")]]
        values = self.census.loc[year, census_cols]
        total_pop = self.census.loc[year, 'pop']
        val_a = values.div(values.sum()).mul(total_pop)
        return val_a

    def _find_values(self, inx):
        x_list = list()
        for i in inx:
            v_values = i.split('_')[-1]
            v_values = v_values.split('-')
            v_values = [re.sub('inf', '100', i) for i in v_values]
            v_values = [re.sub('[^0-9a-zA-Z]+', '', i) for i in v_values]
            v_values = [re.sub('[a-zA-Z]+', '_', i) for i in v_values]
            v_values = np.asarray([int(i) for i in v_values])
            x_list.append(np.mean(v_values))
        return(np.asarray(x_list))

    def _get_weighted_mean(self, inx, year):
        x_list = self._find_values(inx)
        weight = self.census.loc[year, inx].tolist()
        avr = np.average(x_list, weights=weight)
        variance = np.average((x_list - avr)**2, weights=weight)
        return(avr, math.sqrt(variance))

    def _get_positions(self, e, spa):
        sufix = ''
        e2 = ''
        census_cols = [c.split('_')[-1].lower()
                       for c in self.census.columns
                       if e.lower() in [i.lower()
                                        for i in c.split("_")]]
        for rf in self.ref_cat:
            if rf.lower() in census_cols:
                if spa:
                    e = spa
                return(e, e, rf)

        if e == 'Urban' or e == 'Urbanity':
            sufix = 'Urban'
            e2 = 'Urbanity'
        elif e == 'Sex':
            sufix = 'female'
            e2 = 'sex'
        elif e in [c.split('_')[0] for c in self.census.columns]:
            if spa:
                e = spa
            return(e, e, sufix)
        else:
            sufix = 'yes'
            e2 = e
        if spa:
            e = spa
        return(e, e2, sufix)

    def _get_cols(self, table, val='p', static=False):
        cols = list()
        for el in table.index:
            name = el.split('_')[-1]
            if self.verbose:
                print("\t\t| try: {}".format(name), end=' ')
            if name not in self.skip:
                if self.verbose:
                    print('not in skip', end='')
                if static:
                    if self.verbose:
                        print(', static', end='')
                    cols.append(name)
                else:
                    if self.verbose:
                        print(', not static', end='')
                    try:
                        value = float(table.loc[el, val])
                        if not np.isnan(value):
                            cols.append(name)
                        if self.verbose:
                            print(', OK!')
                    except TypeError:
                        if self.verbose:
                            print(', Fail!')
                        pass
            else:
                if self.verbose:
                    print('OK')
        return cols


def main():
    pass


if __name__ == "__main__":
    main()
    # import doctest
    # doctest.testmod()
