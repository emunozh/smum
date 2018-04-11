#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 25 Jan 2017 03:42:42 PM CET

"""
from smum.urbanmet.streams import Flow, Stock


class MaterialsFlow(Flow):
    """Sample Flow class.
    """

    def __init__(self, city):
        """MaterialsFlow class initiator.

        Args:
            city (urbanmetabolism.city.City): city object.

        """
        M = city.d.Materials
        print(M)


class MaterialsStock(Stock):
    """Material Stock.

    This class defines the existing material stock by sector.

    Attributes:
        affluence (pandas.DataFrame): city affluence data
        technology (pandas.DataFrame): city technology data
        data (pandas.DataFrame): city materiasl data
        g (pandas.DataFrame): city constant values
        storage ()

    """

    def __init__(self, city):
        """MaterialsStock class initiator.

        Args:
            city (urbanmetabolism.city.City): city object.

        """
        self.affluence = city.d.MaterialsAffluence
        self.technology = city.d.MaterialsTechnology
        self.data = city.d.Materials
        self.g = city.g
        self.computeStock()
        self.storage = self.storage.T


def main():
    from urbanmetabolism.city import City
    my_city = City()
    # mf = MaterialsFlow(my_city)
    mstock = MaterialsStock(my_city)
    print(mstock.storage)
    mstock.addFlow('new_Flow')
    print(mstock.new_Flow.name)


if __name__ == "__main__":
    main()
