#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 01 Feb 2017 12:41:53 PM CET

"""
from urbanmetabolism.streams import Flow, Stock


class FoodFlow(Flow):
    r"""
    Food Flow class.

    .. math::

        I_F + P_F + I_{W,Kit} = O_{F,RetFW} + O_{F,ResFW} + O_{F,Met} + O_{F,S}

    Where:

        - :math:`I_F` mass of food and packaged drinks imported to the city.
        - :math:`P_F` mass of food and packaged drinks produced in the city, for internal consumption.
        - :math:`I_{W,Kit}` mass of kitchen water used during food preparation or drunk during meals.
        - :math:`O_{F,RetFW}` mass of retail food waste produced by grocery stores and restaurants.
        - :math:`O_{F,ResFW}` mass of residential food waste going to landfill, compost, or organic waste collection.
        - :math:`O_{F,Met}` mass of carbon and water lost via respiration and transpiration in residents metabolism.
        - :math:`O_{F,S}` mass of feces and urine exported to sewerage system.

    """

    def __init__(self, city):

        M = city.d.Food
        print(M)


class FoodStock(Stock):
    r"""
    Food Stock.

    """

    def __init__(self, city):
        """
        Food Stock class initiator.

        - Require input: `city`

        """
        pass


def main():
    print('hello!')


if __name__ == "__main__":
    main()
