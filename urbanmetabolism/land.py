#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 01 Feb 2017 12:41:53 PM CET

"""
from urbanmetabolism.streams import Flow, Stock


class LandFlow(Flow):
    r"""
    land Flow class.

    """

    def __init__(self, city):

        M = city.d.land
        print(M)


class LandStock(Stock):
    r"""
    Land Stock.

    """

    def __init__(self, city):
        """
        Land Stock class initiator.

        - Require input: `city`
        """
        pass


def main():
    print('hello!')


if __name__ == "__main__":
    main()
