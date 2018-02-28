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

    All material streams are aggregated by sector.

    A data-set will the detail material stream is generated as a `csv` file and
    stored under the `/results` folder.

    The Material Stock is computed as follows:

    .. math::

        S_M = \sum_s \sum_m S^s_{M,m}

    The total materials stock of a city is expressed as the total sum of all
    type of materials :math:`m` of all urban structures :math:`s`.

    .. math::

        S^{rb}_{M,m} = P * f^{rb} * i^{rb}_{M,m}

    Where:

        - :math:`S^{rb}_{M,m}` Material stock of residential buildings.
        - :math:`P` Population of the urban agglomeration.
        - :math:`f^{rb}` Per-capita floor space for residential buildings.
        - :math:`i^{rb}_{M,m}` Material intensity per squared meter.

    .. math::

        S^{ti}_{M,m} = A * p^{ti} * i^{ti}_{M,m}

    Where:

        - :math:`S^{ti}_{M,m}` Material amount in linear transportation infrastructure.
        - :math:`A` Surface area of urban agglomeration.
        - :math:`p^{ti}` Density of urban infrastructure.
        - :math:`i^{ti}_{M,m}` Material intensity per kilometer of urban infrastructure.

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
