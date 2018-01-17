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

    Computed as the sum of residential and non-residential water demand.

    .. math::

        Q_W = Q^{hh}_{W,D} + Q^{nr}_{W,D}

    Where:
        - :math:`Q^{hh}_{W,D}` Household water consumption.
        - :math:`Q^{nr}_{W,D}` Non-Residential water consumption.

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

    Based on data availability and water consumption model definition.

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

    The non-residential water demand model is defined as the sum of (source: DGNB):

        - Water consumption by buildings occupants. :math:`Q^{nr}_{DU}`
        - Water consumption for cleaning. :math:`Q^{nr}_{DC}`
        - Water consumption by spa facilities. :math:`Q^{nr}_{DS}`
        - Water consumption by laundering facilities. :math:`Q^{nr}_{DL}` (not implemented)

    .. math::

        Q^{nr}_{W,D} =
        Q^{nr}_{W,DU} +
        Q^{nr}_{W,DC} +
        Q^{nr}_{W,DS} +
        Q^{nr}_{W,DL}

    Where:

    .. math::

        Q^{nr}_{W,DU} = \\sum_{i=1}^{n} wb_I

    .. math::

        wb_I = \\left(n_{NU} \\times f_{I} \\times as_{I} \\times d/a \\right) / 1000

    Where:

        - :math:`n_{NU}` Number of users/occupants/employees/visitors/customers
        - :math:`f_I` Installation factor of equipment (see :ref:`Tab. W1 <fi>`) :math:`[s/d]`
        - :math:`as_I` Equipment water demand factor (see :ref:`Tab. W2 <asi>`) :math:`[l/u]`
        - :math:`d` Occupancy rate in days

    .. _fi:

    .. table:: Tab. W1. Installed equipment factors :math:`f_I`

        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Equipment       | Office   | Hospital (number of beds                                   | Commerce            | Hotel                                          | Residential |
        |                 |          |                                                            |                     |                                                |             |
        |                 |          | (number of beds :math:`n_{e}`)                             |                     | (single :math:`n_{ez}`, double :math:`n_{dz}`) |             |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        |                 | Employee | Employee | Patient             | Visitor                   | Employee | Customer | Customer                                       | Occupant    |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | :math:`n_{NU}`  |          |          | :math:`0.8 * n_{e}` | :math:`0.5 * 0.8 * n_{e}` |          |          | :math:`(n_{ez} + (n_{DZ} * 1.2)) * 0.65`       |             |
        +=================+==========+==========+=====================+===========================+==========+==========+================================================+=============+
        | Toilet sink     | 75       | 45       | 135                 | 15                        | 45       | 15       | 75                                             | 195         |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | WC-Saving       | 4        | 1        | 2                   | 0.5                       | 1        | 0.3      | 1                                              | 4           |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | WC              | 1        | 1        | 1                   | 0.5                       | 1        | 0.5      | 1                                              | 1           |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Urinal          | 4        | 1        |                     | 0.5                       | 1        | 0.2      | 1                                              |             |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Shower          | 30       | 60       | 90                  |                           | 30       |          |                                                | 120         |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Kitchen sink    | 20       | 20       |                     |                           | 20       |          |                                                |             |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Sink-Spa        |          |          |                     |                           |          |          | 15                                             |             |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | WC-Saving-Spa   |          |          |                     |                           |          |          | 1                                              |             |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Shower-Spa      |          |          |                     |                           |          |          | 600                                            |             |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Dishwasher      |          |          |                     |                           |          |          |                                                | 0.5         |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+
        | Washing machine |          |          |                     |                           |          |          |                                                | 0.25        |
        +-----------------+----------+----------+---------------------+---------------------------+----------+----------+------------------------------------------------+-------------+

    .. _asi:

    .. table:: Tab. W2. Water demand factors

        +-----------------+----------+----------+----------+-------+-------------+
        | Equipment       | Office   | Hospital | Commerce | Hotel | Residential |
        +=================+==========+==========+==========+=======+=============+
        | Toilet sink     | 0.15     | 0.15     | 0.15     | 0.15  | 0.15        |
        | :math:`[l/s]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | WC-Saving       | 4.5      | 4.5      | 4.5      | 4.5   | 4.5         |
        | :math:`[l/u]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | WC              | 9        | 9        | 9        | 9     | 9           |
        | :math:`[l/u]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | Urinal          | 3        | 3        |          |       |             |
        | :math:`[l/u]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | Shower          | 0.25     | 0.25     | 0.25     | 0.25  | 0.25        |
        | :math:`[l/s]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | Bathtub         |          |          |          |       | Capacity    |
        | :math:`[l/u]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | Kitchen sink    |          | 0.25     | 0.25     |       |             |
        | :math:`[l/s]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | Dishwasher      |          |          |          |       | 20          |
        | :math:`[l/u]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+
        | Washing machine |          |          |          |       | 60          |
        | :math:`[l/u]`   |          |          |          |       |             |
        +-----------------+----------+----------+----------+-------+-------------+

    .. math::

        Q^{nr}_{W,DC} = \sum_{i = 1}^n \\left(A_{R,i} \\times wb_{R/A} \\right) / 1000

    .. math::

        Q^{nr}_{W,DS} = \sum_{i = 1}^n wb_I

    .. math::

        wb_I = \\left( n_{SPA} \\times f_I \\times as_I \\times 360 d/a \\right) / 1000

    .. math::

        n_{SPA} = n_{NU} \\times 0.25

    .. math::

        Q^{nr}_{W,DL} = \sum_{i = 1}^n wb_I

    Where:

        - :math:`A_R` Cleaning floor space :math:`[m^3/a]`
        - :math:`wb_R` Water demand per cleaning area (see :ref:`Tab. W3 <wbR>`) :math:`[l/(m^2 \\times a)]`
        - :math:`wb_I` Specific water demand of spa/laundry installations (see :ref:`Tab. W1 <fi>` and :ref:`Tab. W2 <asi>`) :math:`[m^3/a]`

    .. _wbR:

    .. table:: Tab. W3. Water demand per cleaning area. :math:`wb_R` in :math:`[l/m^2a]`

        +--------------+------------+--------+----------+----------+--------+-------------+
        | Type of area | Frequency  | Office | Hospital | Commerce | Hotel  | Residential |
        +==============+============+========+==========+==========+========+=============+
        | Floor        | 1 x Month  | 1.50   | 1.50     | 1.50     | 1.50   | 1.50        |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 1 x Week   | 6.25   | 6.25     | 6.25     | 6.25   | 6.25        |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 3 x Week   | 18.75  | 18.75    | 18.75    |        | 18.75       |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 4.5 x Week |        |          |          | 28.125 |             |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 5 x Week   |        | 31.25    |          |        |             |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 6 x Week   |        | 37.50    | 37.50    |        |             |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 7 x Week   |        | 43.75    |          | 43.75  |             |
        +--------------+------------+--------+----------+----------+--------+-------------+
        | Glass        | 2 x Year   | 0.60   |          |          |        | 0.60        |
        +              +------------+--------+----------+----------+--------+-------------+
        | surface      | 4 x Year   | 1.20   | 1.20     | 1.20     | 1.20   | 1,20        |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 6 x Year   | 1.80   |          |          |        | 1.80        |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 12 x Year  |        | 3.60     | 3.60     | 3.60   |             |
        +              +------------+--------+----------+----------+--------+-------------+
        |              | 24 x Year  |        |          | 7.20     | 7.20   |             |
        +--------------+------------+--------+----------+----------+--------+-------------+

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
    wstock = WaterStock(my_city)
    print(wstock.storage)


if __name__ == "__main__":
    main()
