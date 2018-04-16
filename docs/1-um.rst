.. _um:

Top-Down: City-Systems (Urban Metabolism)
=========================================

The Urban Metabolism  (UM) module aims to describe the resource flows of
a city-system at an aggregated level with the use of input-output tables.

This module aims to provide:

  1. A framework for the description of resource flows;
  2. A description of macro-level drivers for the changes of these flows; and
  3. A description of linkages between different resource flows.

The advantgage of this library module--and of this python library-- is that each
module can be used independently. The UM module can be run independently from
the rest of the library.

The library is structured as two types of functions:

  1. A function dedicated to the description of a city and city-data, see :ref:`city`.
  2. Resource flow specific classes:

     - :ref:`materials` & api: :ref:`materials_api`
     - :ref:`water` & api: :ref:`water_api`
     - :ref:`energy` & api: :ref:`energy_api`
     - :ref:`food` & api: :ref:`food_api`
     - :ref:`waste` & api: :ref:`waste_api`

For each of these classes a Unified Modeling Language (UML_) class diagram has
been generated.

The description of the individual functions of this module can be found below
under: :ref:`apium`.

.. _UML: https://en.wikipedia.org/wiki/Unified_Modeling_Language

.. _energy:

Consumption model: Energy
-------

Energy Class: Energy Flow
~~~~~
"Energy flow is the ultimate of the urban metabolism. The magnitude of energy flows
from heating and cooling are typically related to climate, but other components
of urban energy use can be linked back to the shape and form of a city, reflected
by its infrastructure systems and hence material stocks. (Kennedy.2012)"


.. math::

   I_{E} = I_{E,buildings} + I_{E,transport} + I_{E,industry} +
           I_{E,construction} + I_{E,water pumping} + I_{E,waste}

The total energy demand of a city is expressed as the total sum of all
energy demands for the different energy sectors.
Each energy sector energy demand is calculated, based on different components
that are the cause of energy consumed.

**Buildings:**
For the building sector the total energy consumption is computed based on
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
    - :math:`f` Floor space area per capita.
    - :math:`cp`

**Transport:**
The total energy demand for the transport sector is computed based on the
different types of transportation observed within the analyzed system:

.. math::

    I_{E,transport} = I_{E,passenger} + I_{E,freight} + I_{E,aviation} + I_{E,marine}

When anaylzing a city the first xx (Summand) surface passenger transport will be
the most relevant. For this transportation category the energy demand is
calculated based on the different types of passenger transport found in the city:

.. math::

    I_{E,passenger} = \sum_{mode} \frac{1}{P_p} * P * \rho_i * h * \varepsilon

Where:

    - :math:`mode`
    - :math:`P_p` Average population density :math:`[km^{-2}]`.
    - :math:`\rho_i` Density of transportation infrastructure :math:`[km * km^{-2}]`.
    - :math:`h` utilization intensity of infrastructure :math:`[\text{veh-}km * km^{-2}]`.
    - :math:`\varepsilon` Fuel efficiency :math:`[J*\text{veh-}km^{-1}]`.

The product of the first four terms within the summation is equivalent to the vehicle-kilometers traveled (VKT):

.. math::
    VKT = \frac{1}{P_p} * P * \rho_i * h

A widely used city indicator when analyzing sustainability in urban environments.

Energy surface balance (NOT IMPLEMENTED):

.. math::

    I_{E,S} + I_{E,F} + I_{E,I} = O_{E,L} + O_{E,G} + O_{E,E}

Where:

    - :math:`I_{E,S}` Rate of arrival of radiant energy from the sun.
    - :math:`I_{E,F}` Rate of generation of heat due to combustion and dissipation in machinery.
    - :math:`I_{E,I}` Rate of heat arrival from the earth’s interior.
    - :math:`O_{E,L}` Rate of loss of heat by evapotranspiration.
    - :math:`O_{E,G}` Rate of loss of heat by conduction to soil, buildings, roads, etc.
    - :math:`O_{E,E}` Rate of loss of heat by radiation.

Energy class: Stock
~~~~~

This class defines the existing energy stock by sector.

All energy streams are aggregated by sector.

A data-set with the detail energy stream is generated as a `csv` file and
stored under the `/results` folder.

The Energy Stock is computed as follows:

.. _water:

Water
------

Water class: Water Demand
~~~~~~

Similar to Energy Flow, Water Demand is computed as the sum of different water
consumers. In a city most water is consumed at the building level. Therefore
total Water Demand (:math:`Q_W) is determined based on residential and
non-residential water demand.

.. math::

    Q_W = Q^{hh}_{W,D} + Q^{nr}_{W,D}

Where:
    - :math:`Q^{hh}_{W,D}` Household water consumption.
    - :math:`Q^{nr}_{W,D}` Non-Residential water consumption.

** Residential Water Demand:**
The residential water demand or household demand model is computed as function of the following indicators:

    - Demographic characteristics of the household.
    - Disposable income of the household.
    - Average water price in the city.
    - Water saving penetration rate (SP) Yuan, X.-C. et al. (2014).
    - Water saving rate (SR) Yuan, X.-C. et al. (2014).

.. math::

    Q^{hh}_{W,D} = \beta_0 + \sum^{n}_{i} \beta_i HH_{i} + \beta_y Y_{hh,$} + \beta_p P_{$} + \epsilon

Where:

    - :math:`Q^{hh}_{W,D}` Household water consumption.
    - :math:`HH` Household characteristic.
    - :math:`Y_{hh,$}` Household income.
    - :math:`P_{W,$}` Water price.
    - :math:`\beta_i`
    - :math:`\epsilon_{err}` Random error term.

Depending on the water tariff in place the variable :math:`P_{W,$}` cannot be
modeled as a dependent variable. If the water tariff is computed as a
function of consumed volume, the error term cannot be assumed.

The Household characteristics (:math:`HH) are
based on data availability and the definitions made within the water consumption.

Efficiency rate:

The water saving penetration (:math:`SP) and water saving rate (:math:`SR`) are computed at each
simulation step. The water saving rate is an indicator for governmental
actions to reduce water consumption. And the penetration rate is the
likelihood that a household has adopted the respective the water saving behaviour or
technology.

.. math::

    Q_{W,D}^{base}(SP_{W,D}, SR_{W,D}) =
    \begin{cases}
      Q_{W,D}^{hh} \times (1-SR_{W,D}) & \quad \text{if } rand < SP_{W,D}\\
      Q_{W,D}^{hh} & \quad \text{ else}\\
    \end{cases}

Where:

    - :math:`Q^{base}_{W,D}` Base water consumption.
    - :math:`SP_{W,D}` Water saving penetration rate.
    - :math:`SR_{W,D}` Water saving rate.

**Non-residential Water Demand:**
The non-residential water demand model is defined as the sum of (source: DGNB):

    - Water consumption by buildings occupants. :math:`Q^{nr}_{DU}`
    - Water consumption for cleaning. :math:`Q^{nr}_{DC}`
    - Water consumption by spa facilities. :math:`Q^{nr}_{DS}`
    - Water consumption by laundering facilities. :math:`Q^{nr}_{DL}` (not implemented)

.. math::

    Q^{nr}_{W,D} = Q^{nr}_{W,DU} + Q^{nr}_{W,DC} + Q^{nr}_{W,DS} + Q^{nr}_{W,DL}

Where:

.. math::

    Q^{nr}_{W,DU} = \sum_{i=1}^{n} wb_I

.. math::

    wb_I = \left(n_{NU} \times f_{I} \times as_{I} \times d/a \right) / 1000

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

    Q^{nr}_{W,DC} = \sum_{i = 1}^n \left(A_{R,i} \times wb_{R/A} \right) / 1000

.. math::

    Q^{nr}_{W,DS} = \sum_{i = 1}^n wb_I

.. math::

    wb_I = \left( n_{SPA} \times f_I \times as_I \times 360 d/a \right) / 1000

.. math::

    n_{SPA} = n_{NU} \times 0.25

.. math::

    Q^{nr}_{W,DL} = \sum_{i = 1}^n wb_I

Where:

    - :math:`A_R` Cleaning floor space :math:`[m^3/a]`
    - :math:`wb_R` Water demand per cleaning area (see :ref:`Tab. W3 <wbR>`) :math:`[l/(m^2 \times a)]`
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




Flow
~~~~~

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

    Q_{W,D} = \sum_{hh} Q^{base}_{W,D,hh} + CDD * i^{cooling}_W

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

Stock
~~~~~

.. _materials:

Materials
----------

Flow
~~~~~

Stock
~~~~~

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


.. _waste:

Waste
-----

Flow
~~~~~

Stock
~~~~~

.. _food:

Food
-----

Demand
~~~~~~

Computed as the sum of Fod demand.


Flow
~~~~~

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


Stock
~~~~~
