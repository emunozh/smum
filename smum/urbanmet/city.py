#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 25 Jan 2017 03:42:42 PM CET

"""
import pandas as pd


class Attribute(object):
    """
    Generic city attribute.

    Attributes:
        __dict__ (dict): Stream attributes dictionary.

    """
    def addattr(self, x_key, val):
        """Add data to dictionary.

        Args:
            x_key: dictionary key
            val: dictionary values

        """
        self.__dict__[x_key] = val


class City(object):
    """
    City class definition.

    Reads and process the input data required for a city level analysis.

    Attributes:
        name (str): city name.
        dat (City.Data): Data of city-object.
        g_info (City.Info): Info of city-object.

    """
    def __init__(self, name):
        self.get_data()
        self.name = name
        self.dat = False
        self.g_info = False

    def get_data(self, excel_file='./data/InputTables.xlsx'):
        """
        Get excel data.

        Args:
            excel_file (str): Excel file containing input-tables.

        """
        with pd.ExcelFile(excel_file) as xls:
            self.construct_frame(xls)

    def construct_frame(self, xls):
        """
        Construct pandas DataFrames from an excel file.

        Reads all sheets from the `xls` excel file. Sheet names `City` calls
        function `Info`, all the other sheet call function `Data.add_data`

        Args:
            xls (pandas.io.excel.ExcelFile): Excel file.

        """
        self.dat = self.Data()
        for sheet in xls.sheet_names:
            if sheet == 'City':
                info_frame = pd.read_excel(xls, 'City', index_col=0)
                self.g_info = self.Info(info_frame)
            else:
                sheet_data = pd.read_excel(xls, sheet, index_col=0)
                self.dat.add_data(sheet, sheet_data)

    class Data(Attribute):
        """
        Construct city data frames and methods of computation.

        """
        def add_data(self, sheet, sheet_data):
            """Add a data frame to `Data` class.

            Adds data:
                - Affluence
                - Technology

            Args:
                sheet_data (pandas.DataFrame): Excel file data as pandas
                    DataFrame.
                sheet (str): Excel file sheet name.

            """
            if 'Affluence' in sheet_data.index:
                sheet_methods = sheet_data.loc['Affluence']
                sheet_data = sheet_data.drop('Affluence')
                self.addattr(sheet+'Affluence', sheet_methods)
            if 'Technology' in sheet_data.index:
                sheet_indicator = sheet_data.loc['Technology']
                sheet_data = sheet_data.drop('Technology')
                self.addattr(sheet+'Technology', sheet_indicator)
            self.addattr(sheet, sheet_data)

    class Info(Attribute):
        """
        Construct city initial constants.

        Args:
            data (pandas.DataFrame): City constant info data.

        """
        def __init__(self, data):
            for f_data in data.index:
                val = data.ix[f_data, 0]
                f_data = f_data.replace(' ', '_')
                self.addattr(f_data, val)

        def print_info(self):
            pass
