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

    def construct(self, a, b, relative=False):
        """Construct a matrix based on known marginal sums"""
        if relative:
            a_i = a / a.sum()
            b_i = b / b.sum()
        else:
            a_i = a
            b_i = b

        if a.sum() == b.sum():
            if self.verbose:
                print('Marginal sums are align!')
        else:
            print('Marginal sums are not align\nTable example:')
            print(
                """
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

          \sum R = \sum TI
                """)

        m = np.ones((len(a_i), len(b_i)))
        IPF = ipfn(m, [b_i, a_i], [[1],[0]], verbose=(1 if self.verbose else 0))
        m = IPF.iteration()

        if isinstance(m, tuple):
            n = m[0]
        else:
            n = m

        if self.verbose:
            print(r"a = \sum_i m; {}".format(np.allclose(n.sum(1), a_i)))
            print(r"b = \sum_j m; {}".format(np.allclose(n.sum(0), b_i)))

        self.values = n

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

    def read_file(self, input, relative=False, **kargs):
        """read data from file"""
        ext = input.split(".")[-1]
        if ext == 'xlsx' or ext == 'xls':
            self.values = pd.read_excel(input, **kargs)
        else:
            self.values = pd.read_csv(input, **kargs)
        if self.verbose:
            print(self.values.head())


class IOTables():
    """Main I-O table class"""

    def __init__(self, relative=False, verbose=False):
        """Class initiator"""
        self.tables = dict()
        self.relative = relative
        self.verbose = verbose
        self.un_named_tables = 1

    def add(self, input, name=False, **kargs):
        """Add new I-O table to tables list based on input type"""
        if not name:
            name = 'new_table_{}'.format(self.un_named_tables)
            un_named_tables += 1
        m = Table(name, verbose=self.verbose)
        if isinstance(input, list) and len(input) == 2:
            if self.verbose:
                print("Constructing table based on marginal sums only")
            m.construct(input[0], input[1], relative=self.relative)
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
        self.tables[name] = m

    def print_tables(self):
        """Print table names"""
        l = 7
        for i, table in self.tables.items():
            n = table.values.shape[1]
            print('='*l*n)
            print('{:^{}}'.format(i, l*n))
            print('-'*l*n)
            # print(type(self.tables))
            print(np.round(table.values, 2))
            print('='*l*n)


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
