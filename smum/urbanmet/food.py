#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 01 Feb 2017 12:41:53 PM CET

"""
from smum.urbanmet.streams import Flow, Stock


class FoodFlow(Flow):
    r"""
    Food Flow class.
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
