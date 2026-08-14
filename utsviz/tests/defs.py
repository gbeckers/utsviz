"""Shared setup and signal builders for the utsviz tests.

Importing this module selects matplotlib's Agg backend, so that the tests
draw into memory and need no display.

"""
import matplotlib

matplotlib.use('Agg')

import numpy as np
import uts
from uts.core.geometry import RectangularLattice
from uts.core.uniformfrequency import PowerSpectrum
from uts.core.uniformtime import (EpochedMultiChannelUniformTimeSeries,
                                  EpochedUniformTimeSeries,
                                  MultiChannelUniformTimeSeries)
from uts.signalfactory import testsignals

# The multichannel signals are built here rather than with uts's testmcuts and
# testemcuts, which could not produce one with a geometry until after uts
# 2.5.0. Every grid plot in utsviz needs a geometry, and building the signals
# here keeps these fixtures independent of which uts is installed.


def lattice(nx=2, ny=3, channelnames=None):
    """A rectangular geometry, filled row by row with `channelnames`."""
    if channelnames is None:
        channelnames = channelname_list(nx * ny)
    pointnames = [channelnames[row * nx:(row + 1) * nx] for row in range(ny)]
    return RectangularLattice(nx, ny, pointnames=pointnames)


def channelname_list(nchannels):
    return ['channel_%d' % i for i in range(nchannels)]


def _channelsamples(nchannels, nsamples, fs):
    samples = np.empty((nchannels, nsamples), dtype=np.float64)
    times = np.linspace(0., nsamples / float(fs), nsamples)
    for i in range(nchannels):
        samples[i, :] = np.cos(2 * np.pi * 10 * (i + 1) * times)
    return samples


def _epochs(nepochs, epochdur):
    starttimes = np.linspace(0., nepochs * epochdur * 1.1, nepochs)
    epochs = np.empty(starttimes.size, dtype=[('starttime', np.float64)])
    epochs['starttime'] = starttimes
    return epochs


def uts_signal(duration=0.1, fs=1000.):
    return testsignals.testuts(duration=duration, fs=fs, dtype=np.float64)


def mcuts_signal(duration=0.1, fs=1000., nx=2, ny=3):
    nchannels = nx * ny
    nsamples = int(fs * duration)
    channelnames = channelname_list(nchannels)
    return MultiChannelUniformTimeSeries(
        samples=_channelsamples(nchannels, nsamples, fs), fs=fs,
        channelaxis=0, channelnames=channelnames,
        geometry=lattice(nx, ny, channelnames), starttime=0.,
        dtype=np.float64)


def euts_signal(epochdur=0.1, fs=1000., nepochs=10):
    return testsignals.testeuts(epochdur=epochdur, fs=fs, nepochs=nepochs,
                                dtype=np.float64)


def int_euts_signal(epochdur=0.1, fs=1000., nepochs=10):
    nepochsamples = int(fs * epochdur)
    samples = np.arange(nepochs * nepochsamples, dtype=np.int32)
    samples = samples.reshape(nepochs, nepochsamples)
    return EpochedUniformTimeSeries(samples=samples, fs=fs,
                                    epochs=_epochs(nepochs, epochdur),
                                    epochaxis=0, dtype=np.int32)


def emcuts_signal(epochdur=0.1, fs=1000., nepochs=10, nx=2, ny=3):
    nchannels = nx * ny
    nepochsamples = int(fs * epochdur)
    channelnames = channelname_list(nchannels)
    samples = _channelsamples(nchannels, nepochsamples * nepochs, fs)
    samples = samples.reshape(nchannels, nepochs, nepochsamples)
    return EpochedMultiChannelUniformTimeSeries(
        samples=samples, fs=fs, epochs=_epochs(nepochs, epochdur),
        channelaxis=0, epochaxis=1, channelnames=channelnames,
        geometry=lattice(nx, ny, channelnames), dtype=np.float64)


def spectrum_signal():
    return uts.fft(uts_signal())


def powerspectrum_signal():
    spectrum = spectrum_signal()
    samples = np.abs(spectrum.samples.get(copy=False))
    return PowerSpectrum(samples=samples,
                         samplingfrequencies=spectrum.samplingfrequencies)
