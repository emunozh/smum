#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Wed 01 Feb 2017 12:41:53 PM CET

"""
from urbanmetabolism.streams import Flow, Stock


class WasteFlow(Flow):
    r"""
    Waste Flow class.

    """

    def __init__(self, city):

        M = city.d.Waste
        print(M)


class WasteStock(Stock):
    r"""
    Waste Stock.

    """

    def __init__(self, city):
        """
        Waste Stock class initiator.

        - Require input: `city`

        """
        pass


def main():
    print('hello!')


if __name__ == "__main__":
    main()
