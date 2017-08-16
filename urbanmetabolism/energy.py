#!/usr/bin/env python
# -*- coding:utf -*-
"""
#Created by Esteban.

Tue 31 Jan 2017 04:59:53 PM CET

"""
from urbanmetabolism.streams import Flow, Stock


class EnergyFlow(Flow):
    r"""
    Energy Flow class.

    .. math::

       I_{E} = I_{E,buildings} + I_{E,transport} + I_{E,industry} +
               I_{E,construction} + I_{E,water pumping} + I_{E,waste}

    The total energy demand of a city is expressed as the total sum of all
    energy sectors.

    The total energy consumption of the building sector is computed based on
    climatic conditions of the city (:math:`HDD` and :math:`CDD`) and energy
    consumption intensities based on building type.

    .. math::

        I_{E,buildings} = I_{E,heating} + I_{E,cooling} + I_{E,light-and-appl.} + I_{E,water-heating}

    .. math::

        I_{E,heating} = \sum_{building-type} HDD * i_{E,heating} * P * f

    .. math::

        I_{E,cooling} = \sum_{building-type} CDD * i_{E,cooling} * P * f * cp

    Where:

        - :math:`HDD` Heating degree days.
        - :math:`CDD` Cooling degree days.
        - :math:`i_{E,cooling}` heating intensity.
        - :math:`i_{E,heating}` cooling intensity.
        - :math:`P` Population of the urban agglomeration.
        - :math:`f` Floor space are per capita.
        - :math:`cp`

    The total energy demand for the transport sector is computed as follows:

    .. math::

        I_{E,transport} = I_{E,passenger} + I_{E,freight} + I_{E,aviation} + I_{E,marine}

    The computation of energy demand for the passenger transportation can be
    computed as follows:

    .. math::

        I_{E,passenger} = \sum_{mode} \frac{1}{P_p} * P * \rho_i * h * \varepsilon

    Where:

        - :math:`P_p` Average population density :math:`[km^{-2}]`.
        - :math:`\rho_i` Density of transportation infrastructure :math:`[km * km^{-2}]`.
        - :math:`h` utilization intensity of infrastructure :math:`[\text{veh-}km * km^{-2}]`.
        - :math:`\varepsilon` Fuel efficiency :math:`[J*\text{veh-}km^{-1}]`.

    The sum of the first four terms within the summation is equivalent to the vehicle-kilometers traveled (VKT) city indicator.

    Energy surface balance (not implemented):

    .. math::

        I_{E,S} + I_{E,F} + I_{E,I} = O_{E,L} + O_{E,G} + O_{E,E}

    Where:

        - :math:`I_{E,S}` Rate of arrival of radiant energy from the sun.
        - :math:`I_{E,F}` Rate of generation of heat due to combustion and dissipation in machinery.
        - :math:`I_{E,I}` Rate of heat arrival from the earth’s interior.
        - :math:`O_{E,L}` Rate of loss of heat by evapotranspiration.
        - :math:`O_{E,G}` Rate of loss of heat by conduction to soil, buildings, roads, etc.
        - :math:`O_{E,E}` Rate of loss of heat by radiation.

    """

    def __init__(self, city):

        M = city.d.Energy
        print(M)


class EnergyStock(Stock):
    r"""
    Energy Stock.

    This class defines the existing energy stock by sector.

    All energy streams are aggregated by sector.

    A data-set will the detail energy stream is generated as a `csv` file and
    stored under the `/results` folder.

    The Energy Stock is computed as follows:

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
    from urbanmetabolism.city import City
    my_city = City()
    # mf = EnergyFlow(my_city)
    mstock = EnergyStock(my_city)
    print(mstock.storage)
    mstock.addFlow('new_Flow')
    print(mstock.new_Flow.name)


if __name__ == "__main__":
    main()
