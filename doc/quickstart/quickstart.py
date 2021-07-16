"""
==========
Quickstart
==========
test test test
If you have not installed *SMAC* yet take a look at the `installation instructions <installation.html>`_ and make sure that all the requirements are fulfilled.
Examples to illustrate the usage of *SMAC* - either by reading in a scenario file, or by directly using *SMAC* in Python - are provided in the examples-folder.

To get started, we will walk you through a few examples.

* First, we explain the basic usage of *SMAC* by optimizing the `Branin`__-function as a toy example.
* Second, we explain the usage of *SMAC* within Python by optimizing a `Support Vector Machine`__.
* Third, we show a real-world example, using an algorithm-wrapper to optimize the `SPEAR SAT-solver`__.

__ branin-example_
__ svm-example_
__ spear-example_

"""


###############################################################################
# .. _branin-example:
#
# Brainin
# =======
# First of, we'll demonstrate the usage of *SMAC* on the minimization of a standard 2-dimensional continuous test function (`Branin <https://www.sfu.ca/~ssurjano/branin.html>`_).
# This example aims to explain the basic usage of *SMAC*. There are different ways to use *SMAC*:
# 
# f_min-wrapper
# ~~~~~~~~~~~~~
# The easiest way to use *SMAC* is to use the `f_min SMAC wrapper
# <apidoc/smac.facade.func_facade.html#smac.facade.func_facade.fmin_smac>`_. It is
# implemented in `examples/branin/branin_fmin.py` and requires no extra files. We
# import the fmin-function and the Branin-function:

from branin import branin
from smac.facade.func_facade import fmin_smac

###############################################################################
# And run the f_min-function

x, cost, _ = fmin_smac(func=branin,  # function
                       x0=[0, 0],  # default configuration
                       bounds=[(-5, 10), (0, 15)],  # limits
                       maxfun=10,  # maximum number of evaluations
                       rng=3)  # random seed
print("Optimum at {} with cost of {}".format(x, cost))

