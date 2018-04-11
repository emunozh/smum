#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Fri 27 Jan 2017 03:04:11 PM CET

"""
from smum.urbanmet.streams import Flow, Stock


class WaterDemand():
    """
    **Water Demand Model**

    This class defines the household water demand model.

    """
    pass


class WaterFlow(Flow):
    """
    **Urban Water Flow**

    This class defines the `WaterFlow` of a city.

    This water flow is balanced as follows:

    """
    pass


class WaterStock(Stock):
    """
    Urban Water Stock

    This class defines the `WaterFlow` of a city.

    """

    def __init__(self, city):
        """
        Class initiator.

        """
        self.affluence = city.d.WaterAffluence
        self.technology = city.d.WaterTechnology
        self.data = city.d.Water
        self.g = city.g
        self.computeStock()
        self.storage = self.storage.T


def main():
    from smum.urbanmet.city import City
    my_city = City('test')
    # wstock = WaterStock(my_city)
    # print(wstock.storage)


if __name__ == "__main__":
    main()
