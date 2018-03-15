.. _intro:

Introduction
================

In order to understand the flow of resources occurring within a city-system we
represent all the inputs and outputs from these city-system. For the
computation of these inputs and outputs the library makes use of the urban
metabolism approach.

The quantification of resources flow at an aggregate level is not enough for
cities to take knowledge-based decisions on future infrastructure investment and policies
targeting a sustainable urban development. In order to understand the impact of
city level policies and investment strategies cities need to understand: 

a) the drivers of consumption and 
b) the plausible impact of these policies on their citizens. 

In order to get this level of understanding we propose the simulation
of consumption intensities at a micro-level. By describing the consumption
intensities at this level of detail (and implicitly their consumption drivers)
cities have a tool to assess the impact at this micro-level (e. g. at municipal or district level). The
microsimulation module of this library constructs a micro-level synthetic
sample with demographic-variables (drivers) and consumption values (benchmarked
to aggregated values from the Urban Metabolism (UM) model).

.. image:: ./_static/images/GI-REC_model.png
   :align: center
   :width: 10cm

This documentations aims to describe the main rationale behind the development
of this python library and to present an overview of the main modules and
functions implemented on the library.

The python library is composed of two main components:

  1. An Urban Metabolism section, see :ref:`um` that aims to balance all resource flows of
     city systems at an aggregated level (i.e. city-level) and;

  2. A Spatial Microsimulation section, see :ref:`sm`. which constructs a synthetic city and 
     allocates consumption values to micro-level agents

For a complete implementation example of the Spatial Microsimulation Urban Metabolism Model (SMUM), please
refer to the following link:
`ipython notebook <http://nbviewer.jupyter.org/github/emunozh/um/blob/master/docs/example_ph/Welcome.ipynb>`_

This set documentations provides the documentation of the individual library
modules and all the functions within these modules as well as some extended
examples on how to use the provided functions.

The main library function documentation can be found at:

  - :ref:`um-api` for the Urban Metabolism functions and at
    :ref:`um` for a general description of the UM module; and
  - :ref:`sm-api` for the Spatial Microsimulation functions and at
    :ref:`sm` for a general description of the SM module.

A complete list of examples can be found at:

  - :ref:`examples`

A complete list of the library authors and contributors can be found at:

  - :ref:`authors`; and
  - :ref:`contrib`
