#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Tue 31 Jan 2017 04:59:53 PM CET

"""
from smum.urbanmet.streams import Flow, Stock


class EnergyFlow(Flow):
    r"""
    Energy Flow class.
    """

    def __init__(self, city):

        M = city.d.Energy
        print(M)

    def transport_I(self, mode, Pp, P, rho, h, epsilon):
        """

        """

        I = 0
        for m in mode:
            I += m * 1/Pp * P * rho * h * epsilon

        return I


class EnergyStock(Stock):
    r"""
    Energy Stock.
    """

    def __init__(self, city):
        """
        Energy Stock class initiator.

        - Require input: `city`
        """
        self.affluence = city.d.EnergyAffluence
        self.technology = city.d.EnergyTechnology
        self.data = city.d.Energy
        self.g = city.g
        self.computeStock()
        self.storage = self.storage.T


def main():
    from smum.urbanmet import City
    my_city = City()
    # mf = EnergyFlow(my_city)
    mstock = EnergyStock(my_city)
    print(mstock.storage)
    mstock.addFlow('new_Flow')
    print(mstock.new_Flow.name)


if __name__ == "__main__":
    main()
