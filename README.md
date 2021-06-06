[![Build Status](https://travis-ci.org/CYINT/regressions.svg?branch=master)](https://travis-ci.org/CYINT/regressions) [![codecov](https://codecov.io/gh/CYINT/regressions/branch/master/graph/badge.svg)](https://codecov.io/gh/CYINT/regressions)

# Regressions
Helper module to automatically create the best regressions for your data

Library to wrap helper functions for pyodbc in Aptiviti code. Target OS: Windows 10

## Install as module

`python3 -m pip install cyint-regressions`

Then you can

`from cyint_regressions import ...`

Where ... is the function(s) you wish to import.

Examples:

`from cyint_regressions import normal_test, calculate_residuals, has_multicolinearity, errors_autocorrelate`
`from cyint_regressions import error_features_correlate, is_homoscedastic, boxcox_transform, join_dataset`

### Functions

## Dependencies

First run

`python3 -m pip install requirements.txt`

before developing locally

## Build

Delete the `dist` folder if it already exists.
Don't forget to increment the version number in `setup.py `prior to building.
`python3 .\setup.py bdist_wheel` to create the `dist` folder containing the package build.

## Contributing

Email [dfredriksen@cyint.technology](mailto:dfredriksen@cyint.technology) if you would like to contribute