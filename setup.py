import sys
import versioneer
import setuptools

if sys.version_info < (3,6):
    print("Utsviz requires Python 3.6 or higher please upgrade")
    sys.exit(1)

DESCRIPTION = 'utsviz, a python package for uniform time series visualization'
LONG_DESCRIPTION = """Uniform Time Series Visualization

Uniform Time Series Visualization is a python package that enables uniform time
series object visualization."""

DISTNAME = 'utsviz'
LICENSE = 'BSD3'
AUTHOR = "Gabriel Beckers"
MAINTAINER = "Gabriel Beckers"
MAINTAINER_EMAIL = "gabriel@gbeckers.nl"
URL = "http://gbeckers.nl"
DOWNLOAD_URL = ''
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
]

setuptools.setup(
  name = DISTNAME,
  version=versioneer.get_version(),
  cmdclass=versioneer.get_cmdclass(),
  maintainer=MAINTAINER,
  packages=['utsviz', 'utsviz.movies', 'utsviz.plots'],
  maintainer_email=MAINTAINER_EMAIL,
  description=DESCRIPTION,
  license=LICENSE,
  url=URL,
  download_url=DOWNLOAD_URL,
  long_description=LONG_DESCRIPTION,
  classifiers=CLASSIFIERS,
  platforms='any', requires=['matplotlib', 'scipy', 'seaborn', 'numpy'],
  python_requires='>=3.6',
  install_requires=['numpy'],
  data_files = [("", ["LICENSE"])],
)