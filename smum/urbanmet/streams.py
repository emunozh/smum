#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Fri 27 Jan 2017 09:39:21 AM CET

"""
import pandas as pd
from smum.urbanmet.city import Attribute as Stream


#class Stream(object):
#    """Generic city attribute.
#
#    Attributes:
#        __dict__ (dict): Stream attributes dictionary.
#
#    """
#    def addattr(self, x, val):
#        """Add attribute to steam.
#
#        Args:
#            x ()
#
#        """
#        self.__dict__[x] = val

class Flow(Stream):
    """Defines all urban steam flows.

    """
    input = list()
    output = list()
    emissions = list()

    def __init__(self, flow_name, internal=False):
        """
        Class initiator for urban flows.

        """
        if internal:
            self.name = "Internal Urban Flow for <{}>".format(flow_name)
        else:
            self.name = "External Urban Flow for <{}>".format(flow_name)


class Stock(Stream):
    """Defines existing urban resources stocks.

    """
    name = "Urban Stocks"
    storage = pd.DataFrame()
    affluence = pd.DataFrame()
    storage = pd.DataFrame()
    affluence = pd.DataFrame()
    technology = pd.DataFrame()
    data = pd.DataFrame()

    def addFlow(self, flow_name, internal=False):
        """Add a flow to the stock.

        """
        self.addattr(flow_name, Flow(flow_name, internal=internal))

    def assignVariable(self, var):
        """Fetch a variable from the data.

        """
        if isinstance(var, str):
            return self.g.__getattribute__(var)
        else:
            return 1

    def computeVariable(self, var):
        """Computes a variable.

        Assign variable to object.

        """

        var_out = list()

        try:
            for m in var.split(','):
                var_out.append(self.assignVariable(m))
        except:
            var_out = self.assignVariable(var)

        return var_out

    def computeStock(self):
        """Compute the material stock depending on passed `affluence`.

        """

        output_data = pd.DataFrame()

        for i in range(self.affluence.shape[0]):
            intensities = self.data.ix[:, i]

            affl = self.computeVariable(self.affluence[i])
            tech = self.computeVariable(self.technology[i])
            print(tech)

            for a in affl:
                intensities = intensities * a
            for t in tech:
                intensities = intensities * t

            output_data = output_data.append(intensities)

        self.storage = self.storage.append(output_data)
