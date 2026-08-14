import os
import tempfile
import unittest

import numpy as np
import matplotlib.pyplot as plt

from . import defs
from utsviz import plot, image, imshow, imsave, GridAxes
from utsviz.plots import plots


class PlotTestCase(unittest.TestCase):
    """Closes whatever figures a test left behind."""

    def tearDown(self):
        plt.close('all')


class TestPlotDispatch(PlotTestCase):

    def test_uts(self):
        ax = plot(defs.uts_signal())
        self.assertEqual(len(ax.lines), 1)

    def test_mcuts(self):
        signal = defs.mcuts_signal()
        ga = plot(signal)
        self.assertEqual(sorted(ga.axes), sorted(signal.channelnames))

    def test_euts(self):
        ax = plot(defs.euts_signal())
        self.assertEqual(len(ax.images), 1)

    def test_emcuts(self):
        signal = defs.emcuts_signal()
        ga = plot(signal)
        self.assertEqual(sorted(ga.axes), sorted(signal.channelnames))

    def test_spectrum(self):
        # dispatched to a function that did not exist under that name
        ga = plot(defs.spectrum_signal())
        self.assertEqual(sorted(ga.axes), ['magnitude', 'phase'])

    def test_powerspectrum(self):
        ax = plot(defs.powerspectrum_signal())
        self.assertEqual(len(ax.lines), 1)

    def test_unsupportedtyperaises(self):
        self.assertRaises(NotImplementedError, plot, np.arange(10))


class TestPlotUts(PlotTestCase):

    def test_ownaxes(self):
        fig, ax = plt.subplots()
        self.assertIs(plots.plot_uts(defs.uts_signal(), ax=ax), ax)

    def test_ylim(self):
        ax = plots.plot_uts(defs.uts_signal(), ylim=(-2., 3.))
        self.assertEqual(ax.get_ylim(), (-2., 3.))

    def test_lines(self):
        ax = plots.plot_uts(defs.uts_signal(), vlines=(0.01, 0.02),
                            hlines=(0.,))
        self.assertEqual(len(ax.lines), 4)  # the signal plus three lines

    def test_singlelinebackwardcompatibility(self):
        ax = plots.plot_uts(defs.uts_signal(), vline=0.01, hline=0.)
        self.assertEqual(len(ax.lines), 3)


class TestPlotMcuts(PlotTestCase):

    def test_ylim(self):
        ga = plots.plot_mcuts(defs.mcuts_signal(), ylim=(-2., 3.))
        for ax in ga.axes.values():
            self.assertEqual(ax.get_ylim(), (-2., 3.))

    def test_autoyticks(self):
        ga = plots.plot_mcuts(defs.mcuts_signal(), ylim=(-2., 3.),
                              autoyticks=True)
        self.assertEqual(list(ga.ylabelsubplot.get_yticks()), [-2., 3.])

    def test_reuseofgridaxes(self):
        signal = defs.mcuts_signal()
        ga = GridAxes(signal.geometry)
        self.assertIs(plots.plot_mcuts(signal, ga=ga), ga)


class TestGridAxes(PlotTestCase):

    def test_geometry(self):
        signal = defs.mcuts_signal(nx=2, ny=3)
        ga = GridAxes(signal.geometry)
        self.assertEqual(len(ga.axes), 6)

    def test_labels(self):
        ga = GridAxes(defs.mcuts_signal().geometry, xlabel='Time (s)',
                      ylabel='mV')
        self.assertEqual(ga.xlabelsubplot.get_xlabel(), 'Time (s)')
        self.assertEqual(ga.ylabelsubplot.get_ylabel(), 'mV')

    def test_frameoncolor(self):
        ga = GridAxes(defs.mcuts_signal().geometry, frameon=True,
                      framecolor=(1., 0., 0.))
        spine = list(ga.axes.values())[0].spines['left']
        self.assertEqual(tuple(spine.get_edgecolor())[:3], (1., 0., 0.))

    def test_lines(self):
        ga = GridAxes(defs.mcuts_signal().geometry, vlines=(0.01,),
                      hlines=(0., 1.))
        for ax in ga.axes.values():
            self.assertEqual(len(ax.lines), 3)


class TestEutsImage(PlotTestCase):

    def test_shape(self):
        signal = defs.euts_signal(nepochs=10)
        im = image(signal)
        self.assertEqual(im.shape[-1], 4)
        self.assertEqual(im.ndim, 3)

    def test_clipandrangesconflict(self):
        signal = defs.euts_signal()
        self.assertRaises(ValueError, image, signal, clip=(0., 1.),
                          ranges=(5., 95.))

    def test_ranges(self):
        im = image(defs.euts_signal(), ranges=(5., 95.))
        self.assertTrue(np.isfinite(im).all())

    def test_transparantcolor(self):
        # the color that is made transparant has to occur in the image for
        # this to test anything, so take it from the low end of the colormap,
        # which the lowest samples map onto
        signal = defs.euts_signal()
        colormap = defs.matplotlib.colormaps['hot']
        im = image(signal, colormap=colormap,
                   transparantcolor=colormap(0.)[:3])
        self.assertEqual(im[..., 3].min(), 0.)
        self.assertEqual(im[..., 3].max(), 1.)

    def test_integersignal(self):
        # euts_image normalizes in place, which cannot be done on integer
        # samples
        self.assertEqual(image(defs.int_euts_signal()).shape[-1], 4)

    def test_imshow(self):
        axesimage = imshow(defs.euts_signal())
        self.assertEqual(axesimage.get_array().shape[-1], 4)

    def test_imsave(self):
        with tempfile.TemporaryDirectory() as dirname:
            path = imsave(defs.euts_signal(), basefilename='im', path=dirname,
                          includebar=True)
            self.assertEqual(path, os.path.join(dirname, 'im.png'))
            # the image path is returned, not the color bar's
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.join(dirname,
                                                        'im_bar.png')))


class TestSaveMultipage(PlotTestCase):

    def test_pagesarewritten(self):
        signal = defs.mcuts_signal(duration=0.1)
        with tempfile.TemporaryDirectory() as dirname:
            dirpath = os.path.join(dirname, 'pages')
            plots.plot_mcuts_savemultipage([signal], starttime=0., endtime=0.1,
                                           pageduration=0.05, dirpath=dirpath,
                                           figsize=(4, 4))
            self.assertEqual(len(os.listdir(dirpath)), 2)

    def test_markepisodes(self):
        signal = defs.mcuts_signal(duration=0.1)
        with tempfile.TemporaryDirectory() as dirname:
            dirpath = os.path.join(dirname, 'pages')
            plots.plot_mcuts_savemultipage([signal], starttime=0., endtime=0.1,
                                           pageduration=0.05, dirpath=dirpath,
                                           ylims=[(-1., 1.)], figsize=(4, 4),
                                           mark_episodes={'red': [(0.01,
                                                                   0.02)]})
            self.assertEqual(len(os.listdir(dirpath)), 2)


if __name__ == '__main__':
    unittest.main()
