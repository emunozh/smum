#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Thu 13 Apr 2017 09:33:52 AM CEST

"""
from ipfn.ipfn import ipfn
import numpy as np
import pandas as pd
import os


class Table():
    """Simple input-output table"""

    def __init__(self, name, verbose=False):
        """class initiator"""
        self.name = name
        self.values = False
        self.verbose = verbose
        self.row_names = False
        self.col_names = False
        self.links = {'to': False, 'from': False, 'stock': False}
        self.flow = '?'
        self.conversion_factor = False
        self.raw = False

    def construct(self, a, b, relative = False, cf = np.asarray([])):
        """Construct a matrix based on known marginal sums"""
        if relative:
            a_i = a / a.sum()
            b_i = b / b.sum()
        else:
            a_i = a
            b_i = b

        if np.allclose(a.sum() , b.sum(), rtol=0.01):
            if self.verbose:
                print('Marginal sums are align!')
        else:
            print('Marginal sums are not align\nTable example:')
            print(
                u"""
          E.g.:
          Sector---------------+
          +---+---+---+        |
        R | 5 | 6 | 7 |        |
          +---+---+---+---+    V
          | . | . | . | 8 | Fuel Type
          | . | . | . | 6 |
          | . | . | . | 4 |
          +---+---+---+---+
                        TI

          ∑ R = ∑ TI
                """)
            print("a = ", a)
            print("b = ", b)

        m = np.ones((len(a_i), len(b_i)))
        IPF = ipfn(m, [b_i, a_i], [[1],[0]], verbose=(1 if self.verbose else 0))
        m = IPF.iteration()

        if isinstance(m, tuple):
            n = m[0]
        else:
            n = m

        if self.verbose:
            print(u"a = ∑ m_i; {}".format(np.allclose(n.sum(1), a_i)))
            print(u"b = ∑ m_j; {}".format(np.allclose(n.sum(0), b_i)))

        if len(cf) > 0:
            self.raw = n
            n = n * cf
            self.conversion_factor = cf

        self.values = n

        # update pandas
        self.to_pandas()

        return(n)

    def update(self, b, dimension=1):
        """Update inputs of matrix `m` given the outputs `b`"""
        if self.verbose:
            print("updating table")
        m = self.values

        # a = np.dot(m.T, m)
        # b = np.diag(np.ones(a.shape[0])) - a
        # q = np.linalg.solve(b, (b_o - m.sum(0)))

        IPF = ipfn(m.copy(), [b], [[dimension]], verbose=self.verbose)
        self.values = IPF.iteration()
        if isinstance(self.values, tuple):
            self.values = self.values[0]

        # update pandas
        self.to_pandas()


    def read_file(self, input, relative=False, **kargs):
        """read data from file"""
        ext = input.split(".")[-1]
        if ext == 'xlsx' or ext == 'xls':
            self.df = pd.read_excel(input, **kargs)
        else:
            self.df = pd.read_csv(input, **kargs)
        if self.verbose:
            print(self.values.df())
        self.values = np.asarray(self.df)

    def to_pandas(self):
        self.df = pd.DataFrame(self.values)
        if self.col_names:
            self.df.columns = self.col_names
        if self.row_names:
            self.df.index = self.row_names
        if self.verbose:
            print("converted to pandas")

    def set_names(self, names, direction):
        """set colum or row names."""
        if direction == 'cols':
            if len(names) == self.df.shape[1]:
                self.col_names = names
                if self.verbose: print("set {} as col_names".format(names))
            else:
                print("wrong number of names, got {} but expected {}".format(
                    len(names), self.df.shape[1]))
        elif direction == 'rows':
            if len(names) == self.df.shape[0]:
                self.row_names = names
                if self.verbose: print("set {} as row_names".format(names))
            else:
                print("wrong number of names, got {} but expected {}".format(
                    len(names), self.df.shape[0]))

        # update pandas
        self.to_pandas()


class IOTables():
    """Main I-O table class"""

    def __init__(self, relative=False, verbose=False):
        """Class initiator"""
        self.tables = dict()
        self.relative = relative
        self.verbose = verbose
        self.un_named_tables = 1

    def add(self, input, name = False, cols = False, rows = False,
            flow = '?', conversion_factor = np.asarray([]), **kargs):
        """Add new I-O table to tables list based on input type"""
        if not name:
            name = 'new_table_{}'.format(self.un_named_tables)
            un_named_tables += 1
        m = Table(name, verbose=self.verbose)
        m.flow = flow
        if isinstance(input, list) and len(input) == 2:
            if self.verbose:
                print("Constructing table based on marginal sums only")
            m.construct(input[0], input[1], relative=self.relative, cf = conversion_factor)
        elif isinstance(input, str) and os.path.isfile(input):
            extention = input.split('.')[-1]
            if extention == 'xlsx' or extention == 'xls':
                file_type = 'excel'
            else:
                file_type = 'plain text'
            if self.verbose:
                print("Reading data from file with extention: *.{}".format(
                    extention))
                print("Interpreted as an {} file".format(file_type))
                if kargs:
                    print("with following arguments:")
                    for k in kargs:
                        print("\t{} =\t{}".format(k, kargs[k]))
            m.read_file(input,
                        relative=self.relative,
                        **kargs)
        else:
            print("not implemented yet, or wrong data type")
            raise TypeError("can't understand input: ", input)

        # set col and row names
        if cols:
            m.set_names(cols, 'cols')
        if rows:
            m.set_names(rows, 'rows')

        # save as table list
        self.tables[name] = m

    def link(self, table_in, table_out):
        """Link input and Output tables."""
        if table_in not in self.tables.keys() or table_out not in self.tables.keys():
            print('tables not in memory')
            print('tables in memory:')
            for tab in self.tables.keys():
                print('\t' + tab)
        else:
            self.tables[table_in].links['to'] = table_out
            self.tables[table_out].links['from'] = table_in
            self._check_link(table_in, table_out)

    def _check_link(self, table_in, table_out):
        """Balance tables."""
        if self.verbose:
            print('Check link!')
        tab_i = self.tables[table_in].values
        tab_o = self.tables[table_out].values

        if np.allclose(np.sum(tab_i), np.sum(tab_o)):
            print('tables are balanced!')
        else:
            print('will allocate difference to stock table')

            flow_i = self._get_flow(table_in)
            flow_o = self._get_flow(table_out)

            a = self._make_margins(tab_i, tab_o, 1, flow_i, flow_o)
            b = self._make_margins(tab_i, tab_o, 0, flow_i, flow_o)

            tab_rows = [u"Δ" + i.split('_')[0] for i in self.tables[table_in].df.index]
            tab_cols = [u"Δ" + i.split('_')[0] for i in self.tables[table_in].df.columns]
            tab_name = "Stock {}-{}".format(table_in, table_out)
            self.add([a, b], name = tab_name, rows = tab_rows, cols = tab_cols, flow = u"Δ")

            self.tables[tab_name].links['from'] = table_in
            self.tables[tab_name].links['to'] = table_out
            self.tables[tab_name].links['stock'] = True
            self.tables[table_in].links['stock'] = tab_name
            self.tables[table_out].links['stock'] = tab_name

    def _get_flow(self, tab_n):
        flow_n = self.tables[tab_n].flow
        if flow_n == '+' or flow_n == '?':
            flow_n = 1
        elif flow_n == '-':
            flow_n = -1
        else:
            print('flow {} not understood, assuming input (+)'.format(flow_n))
            flow_n = 1
        return(flow_n)

    def _make_margins(self, tab_i, tab_o, pos, flow_i, flow_o, lim = 0.001):

        a = flow_i * tab_i.sum(axis = pos) + flow_o * tab_o.sum(axis = pos)
        _a = []
        for i in a:
            if i > lim:
                _a.append(i)
            else:
                _a.append(0)
        return(np.asarray(_a))

    def print_tables(self, print_only = list(), l = 7):
        """Print table names"""
        if len(print_only) == 0:
            print_only = self.tables.keys()
        elif isinstance(print_only, str):
            print_only = [print_only]

        for i, table in self.tables.items():
            if i in print_only:
                self._print_table(i, table, l)

    def _print_table(self, i, table, l):
        n = table.values.shape[1] + 1
        flow = self.tables[i].flow

        try:
            linked_to = self.tables[i].links['to']
        except:
            linked_to = False
        try:
            linked_from = self.tables[i].links['from']
        except:
            linked_from = False
        try:
            linked_stock = self.tables[i].links['stock']
        except:
            linked_stock = False

        df = table.df.copy()
        df.loc[u'∑'] = df.sum()
        df.loc[:, u'∑'] =  df.sum(axis=1)

        # print link from
        if linked_from and not (linked_stock == True):
            print('  +-- linked from: {}'.format(linked_from))
            print('  |')
            print('  V')
        elif linked_stock == True:
            print("  A | ")
            print("  | | linked to: {}".format(linked_to))
            print("  | | linked from: {}".format(linked_from))
            print("  | V ")
        else:
            print('\n'*1)

        # print table
        print('='*l*n)
        print('{:^{}}'.format("{} ({})".format(i, flow), l*n))
        print('-'*l*n)
        print(np.round(df, 2))
        print('='*l*n)

        if linked_to and not (linked_stock == True):
            print("  +-- linked to: {}".format(linked_to))
            print("  |")
            print("  V")
        # else:
            # print("\n"*1)


def main():
    io = IOTables(relative=True, verbose=True)
    a = np.array([5., 8., 7.])
    b = np.array([9., 6., 4., 1.])
    io.add([a, b], name='test')
    io.print_tables()

    b_o = b
    io.tables['test'].update(b_o)
    io.print_tables()

if __name__ == "__main__":
    main()
