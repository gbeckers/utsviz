from unittest import TestLoader, TextTestRunner, TestSuite

from . import defs
from . import test_colormaps
from . import test_movies
from . import test_plots
from . import test_utils

modules = [test_colormaps, test_movies, test_plots, test_utils]


def test(verbosity=1):
    suite = TestSuite()
    for module in modules:
        suite.addTests(TestLoader().loadTestsFromModule(module))
    return TextTestRunner(verbosity=verbosity).run(suite)
