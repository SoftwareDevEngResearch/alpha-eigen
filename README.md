# alpha-eigen
Using dynamic mode decomposition to calculate alpha-eigenvalues for time-dependent radiation transport problems.

## Installation
1. Clone this repo
2. Run `pip install alpha-eigen`, which will install the package and its dependencies
3. That's it! You're all set

## Running
You have a few options! 
### Tests
This package is compatible with PyTest, so if you'd like to run the integration test problem, just run `pytest` from the main directory. The configured integration test is the short version; to run the long version, from the main directory, run `mv alpha_eigen_modules/tests/long_kornreich_parsons.py alpha_eigen_modules/tests/test_long_kornreich_parsons.py`. Then, run `pytest`. Both short_ and long_ run the heterogeneous supercritical slab from the Kornreich-Parsons benchmark. 

### Command-Line
This package can be run from the command-line or using an IDE. From the command-line, running `python3 main.py --help` will return a list of available command-line arguments. As of 03/17/2022, those are:

--problem [Problem name]

--length [Slab length]

--cells_per_mfp [Cells per mean free path]

--num_angles [Number of SN angles]

--steps [Number of time-steps to use]

Only --problem is a required argument. If not specified, each input has a default steady-state configuration.


### IDE
From an IDE, you can navigate to the alpha_eigen_modules directory and run inputs.py with your chosen input function activated. 

## Available Examples
Various benchmark slabs from Kornreich, Parsons (2005) 10.1016/j.anucene.2005.02.004 are available for configuration in either steady-state or time-dependent mode, to either calculate the k- or alpha-eigenvalues. 

## Info
This package is intended to continue work from McClarren, R. G. (2019). Calculating Time Eigenvalues of the Neutron Transport Equation with Dynamic Mode Decomposition. Nuclear Science and Engineering, 193(8), 854–867. http://doi.org/10.1080/00295639.2018.1565014

## Acknowledgements
This work was supported by the Center for Exascale Monte-Carlo Neutron Transport (CEMeNT) a PSAAP-III project funded by the Department of Energy, grant number: DE-NA003967.
