UTSVIZ
======

.. image:: https://github.com/gbeckers/utsviz/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/gbeckers/utsviz/actions/workflows/tests.yml
   :alt: test status

Visualization of uts signals: plots and movies of uniform time series,
multichannel uniform time series and their epoched and spectral counterparts.

Installation
------------

utsviz needs `uts <https://github.com/gbeckers/uts>`__, which is not on PyPI
and is therefore not installed automatically. Install it first, following its
own README, and then install utsviz from GitHub into the same environment: ::

    $ pip install git+https://github.com/gbeckers/utsviz@master

Everything else (numpy, scipy and matplotlib) is pulled in by pip.

Writing movies additionally needs ``ffmpeg`` on the path, and adding sound to a
movie needs ``sox`` as well. Neither is a python package; install them with
your package manager or with conda.

Requirements
------------

- python >= 3.10
- numpy >= 1.22 (both numpy 1 and numpy 2 work)
- scipy >= 1.9
- matplotlib >= 3.6

The test suite is run against python 3.11 with numpy 1.25, matplotlib 3.7 and
scipy 1.11, and against python 3.14 with numpy 2.5, matplotlib 3.11 and
scipy 1.18.

Running the tests
-----------------

::

    $ pytest --pyargs utsviz

or, from python: ::

    >>> import utsviz.tests
    >>> utsviz.tests.test()

Authors
-------

The `utsviz` library has been written by *Gabriel Beckers* up to version
0.2.4. That version has been modernized by *Claude Opus 5* under supervision
of Gabriel: the setup.py/versioneer build was replaced by pyproject.toml with
setuptools-scm, calls into removed numpy, scipy and matplotlib APIs were
repaired, a number of bugs were fixed and a test suite was added. Each change
was inspected by Gabriel.

AI Statement
------------
Since the modernization described above, Claude is used to audit the
human-designed and -written code for bugs and compatibility. Identified issues
are fixed, with tests, by Claude after approval and under supervision of the
human programmer who wrote the code.
