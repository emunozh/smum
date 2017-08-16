#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 25 Jan 2017 03:42:42 PM CET

"""
import pandas as pd


class Attribute(object):
    """Generic city attribute.

    Attributes:
        __dict__ (dict): Stream attributes dictionary.

    """
    def addattr(self, x, val):
        """Add data to dictionary.

        Args:
            x: dictionary key
            val: dictionary values

        """
        self.__dict__[x] = val


class City(object):
    """City class definition.

    Reads and process the input data required for a city level analysis.

    Attributes:
        name (str): city name.
    """
    def __init__(self, name):
        self.getData()
        self.name = name

    def getData(self, excel_file='./data/InputTables.xlsx'):
        """Get excel data.

        Args:
            excel_file (str): Excel file containing input-tables.

        """
        with pd.ExcelFile(excel_file) as xls:
            self.constructFrame(xls)

    def constructFrame(self, xls):
        """Construct pandas DataFrames from an excel file.

        Reads all sheets from the `xls` excel file. Sheet names `City` calls
        function `Info`, all the other sheet call function `Data.addData`

        Args:
            xls (pandas.io.excel.ExcelFile): Excel file.

        """
        self.d = self.Data()
        for sheet in xls.sheet_names:
            if sheet == 'City':
                info_frame = pd.read_excel(xls, 'City', index_col=0)
                self.g = self.Info(info_frame)
            else:
                sheet_data = pd.read_excel(xls, sheet, index_col=0)
                self.d.addData(sheet, sheet_data)


    class Data(Attribute):
        """Construct city data frames and methods of computation.

        """
        def addData(self, sheet, sheet_data):
            """Add a data frame to `Data` class.

            Adds data:
                - Affluence
                - Technology

            Args:
                sheet_data (pandas.DataFrame): Excel file data as pandas DataFrame.
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
        """Construct city initial constants.

        Args:
            data (pandas.DataFrame): City constant info data.

        """
        def __init__(self, data):
            for f in data.index:
                val = data.ix[f, 0]
                f = f.replace(' ', '_')
                self.addattr(f, val)
