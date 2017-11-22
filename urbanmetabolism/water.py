#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Fri 27 Jan 2017 03:04:11 PM CET

"""
from urbanmetabolism.streams import Flow, Stock


class WaterDemand():
    """
    **Water Demand Model**

    This class defines the household water demand model.

    The household demand model is computed as function of:

        - Demographic characteristics of the household.
        - Disposable income of the household.
        - Average water price in the city.
        - Water saving penetration rate (SP) Yuan, X.-C. et al. (2014).
        - Water saving rate (SR) Yuan, X.-C. et al. (2014).

    .. math::

        Q^{hh}_{W,D} =
        \\beta_0 +
        \\beta_1 HH_{1} + \dots + \\beta_n HH_n +
        \\beta_y Y +
        \\beta_p P +
        \\epsilon

    Where:

        - :math:`Q^{hh}_{W,D}` Household water consumption.
        - :math:`HH` Household characteristic.
        - :math:`Y` Household income.
        - :math:`P` Water price.
        - :math:`\\epsilon` Random error term.

    Depending on the water tariff in place the variable :math:`P` can't be
    modeled as an dependent variable. If the water tariff is computes as
    function of consumed volume we cannot assume the error term.

    Household characteristics:

    Efficiency rate:

    The water saving penetration and water saving rate are computed at each
    simulation step. The water saving rate is an indicator for governmental
    actions to reduce water consumption. And the penetration rate is the
    likelihood of household to have adopted the water saving behaviour or
    technology.


    .. math::

        Q_{W,D}^{base}(SP_{W,D}, SR_{W,D}) =
        \\begin{cases}
          Q_{W,D}^{hh} \\times (1-SR_{W,D}) & \\quad \\text{if } rand < SP_{W,D}\\\\
          Q_{W,D}^{hh} & \\quad \\text{ else}\\\\
        \\end{cases}

    Where:

        - :math:`Q^{base}_{W,D}` Base water consumption.
        - :math:`SP_{W,D}` Water saving penetration rate.
        - :math:`SR_{W,D}` Water saving rate.
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
    **Urban Water Stock**

    This class defines the `WaterFlow` of a city.

    This water flow is balanced as follows:


    .. math::

        I_{W,percip} + I_{W,pipe} + I_{W,sw} + I_{W,gw} = O_{W,evap} + O_{W,out} + \Delta S_w

    Where:

        - :math:`I_{W,percip}` Is natural inflow from precipitation.
        - :math:`I_{W,pipe}` Is water piped into the city.
        - :math:`I_{W,sw}` Is the net surface water flow into the city.
        - :math:`I_{W,gw}` Is the net ground water flow into city aquifers.
        - :math:`O_{W,evap}` Evapotranspiration.
        - :math:`O_{W,out}` Water piped out of cities
        - :math:`\Delta S_w` Change in water storage of urban agglomeration.

    **Anthropogenic Water Use:**

    The anthropogenic water consumption is computed as follows:

    .. math::

        Q_W = Q_{W,D} + Q_{W,L}

    Where:

        - :math:`Q_{W,D}` Water demand.
        - :math:`Q_{W,L}` Water losses.

    .. math::

        Q_{W,D} = \\sum_{hh} Q^{base}_{W,D,hh} + CDD * i^{cooling}_W

    Where:

        - :math:`Q^{base}_{W,D}` Base water consumption.
        - :math:`CDD` Cooling Degree Days.
        - :math:`i^{cooling}_W` Intensity of water use for cooling.

    .. math::

        Q_{W,L} + A * p_{ti} * l

    Where:

        - :math:`Q_{W,L}` Water losses.
        - :math:`A` Surface area of urban agglomeration.
        - :math:`p^{ti}` Density of urban infrastructure.
        - :math:`l` Annual leakage rate per length of linear infrastructure.

    .. math::

        Q_{WWT} = Q_{WWE} + Q_{WWF} + Q_{INF}

    Where:

        - :math:`Q_{WWT}` Treated waste water.
        - :math:`Q_{WWE}` Generated waste water.
        - :math:`Q_{WWF}` Wet weather water flow.
        - :math:`Q_{INF}` Base infiltration.

    **Urban Aquifers:**

    .. math::

        \Delta S_{W,gw} = \Delta Q_{W,RO} + Q_{W,ar} + \Delta I_{W,gw} - \Delta Q_{W,DO} - Q_{W,gwpump}

    Where:

        - :math:`\Delta S_{W,gw}` Change in ground water storage of urban agglomeration.
        - :math:`\Delta Q_{W,RO}` Change in natural recharge from virgin conditions.
        - :math:`Q_{W,ar}` Net anthropogenic urban water recharge rate.
        - :math:`\Delta I_{W,gw}` Net change on ground-water inflow.
        - :math:`\Delta Q_{W,DO}` Change in natural discharge from virgin conditions.
        - :math:`Q_{W,gwpump}` Net pump rate of urban agglomeration.

    **Internal Renewable Water Resources (IRWR)**

    .. math::

        IRWR = S_{W,sw} + S_{W,gw} - S_{W,overlap}

    **External Renewable Water Resources (ERWR)**

    .. math::

        ERWR = I_{W,sw} - O_{W,sw} + I_{W,gw} - O_{W,gw}

    **Total Renewable Water Resources (TRWR)**

    .. math::

        TRWR = (S_{W,sw} + I_{W,sw} - O_{W,sw}) + (S_{W,gw} + I_{W,gw} - O_{W,gw}) - S_{W,overlap}

    Where:

        - :math:`S_{W,sw}` Surface water, produced internally.
        - :math:`S_{W,gw}` Groudwater, produced internally.
        - :math:`S_{W,overlap}` Overlap between surface water and groundwater.

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
    from urbanmetabolism.city import City
    my_city = City('test')
    mstock = WaterStock(my_city)
    print(mstock.storage)


if __name__ == "__main__":
    main()
