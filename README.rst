evtpooling
==========

evtpooling contains the framework needed to improve tail risk forecasts through robust data cleaning, transformation, and loss return calculations.  
It provides flexible ETL utilities for handling time series stock data, validating completeness, transforming data, and calculating daily and weekly loss returns.

Features
--------

* Full ETL pipeline for financial time series data
* Data validation with dtype checking
* Missing data imputation by group means
* Categorical cleaning and fuzzy matching for string variables
* Daily percentage loss return calculations
* Weekly loss return calculations with anchor logic
* Flexible pivoting to generate wide-format datasets for downstream modeling
* Clean architecture with separate transform and test modules

Installation
------------

You can install the released version from PyPI using:

.. code-block:: bash

    pip install evtpooling

Or install directly from the source (development version):

.. code-block:: bash

    git clone https://github.com/JTKimQF/evtpooling.git
    cd evtpooling
    pip install -e .

Usage Example
-------------

Example ETL usage:

.. code-block:: python

    from evtpooling import (
        extract_file,
        transform_data,
        load_file,
        etl_pipeline
    )

    # filepath = 'path/to/your/data.csv'

    clean_df = etl_pipeline(filepath)

Documentation
-------------

Full documentation and function reference is available inside the code base (`src/evtpooling/etl/transform.py`).

License
-------

MIT License

Copyright (c) 2025 J.T. Kim

This package was created with `Cookiecutter`_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
