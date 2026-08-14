import unittest

import numpy as np
import matplotlib.pyplot as plt

from . import defs
from utsviz.movies import (FramesTimeAligned, HPanelImgPlusPlotMovie,
                           HPanelMovie, LetterAxes, SpecgramAxes,
                           TimeIndicatedLetterAxes,
                           TimeIndicatedSpecgramAxes, codecs)


def frames(duration=0.1, fs=1000., fps=25, slowdown=10.):
    signal = defs.mcuts_signal(duration=duration, fs=fs)
    return FramesTimeAligned([signal], fps=fps, slowdown=slowdown,
                             clips=[(-1., 1.)], units=['mV'])


class MovieTestCase(unittest.TestCase):

    def tearDown(self):
        plt.close('all')


class TestFramesTimeAligned(unittest.TestCase):

    def test_decimation(self):
        fta = frames(fs=1000., fps=25, slowdown=10.)
        self.assertEqual(fta.decimate, 4)
        self.assertEqual(fta.realslowdown, 10.)

    def test_frames(self):
        fta = frames()
        generated = list(fta.generate_frames())
        self.assertEqual(len(generated), fta.ntimesamples)
        arrays, frameno, time = generated[0]
        # one panel, laid out as (ny, nx, rgba)
        self.assertEqual(arrays[0].shape, (3, 2, 4))
        self.assertEqual(frameno, 0)
        self.assertTrue(np.isfinite(time))

    def test_defaultcolormap(self):
        # used to go through the removed matplotlib.cm.get_cmap
        fta = frames()
        self.assertTrue(callable(fta.colormaps[0]))

    def test_colormapbyname(self):
        signal = defs.mcuts_signal()
        fta = FramesTimeAligned([signal], colormaps=['divmap1'],
                                clips=[(-1., 1.)])
        from utsviz.colormaps import divmap1
        self.assertIs(fta.colormaps[0], divmap1)

    def test_notimealignedraises(self):
        one = defs.mcuts_signal(duration=0.1)
        other = defs.mcuts_signal(duration=0.2)
        self.assertRaises(ValueError, FramesTimeAligned, [one, other])


class TestHPanelMovie(MovieTestCase):

    def movie(self, **kwargs):
        parameters = dict(pannelwidth=1., pannelheight=1., dpi=50,
                          staticimlabels=['panel'], colorbarlabels=['mV'])
        parameters.update(kwargs)
        movie = HPanelMovie(frames(), **parameters)
        movie.framegenerator = movie.fta.generate_frames()
        return movie

    def test_figuresize(self):
        movie = self.movie()
        # both dimensions have to be an even number of pixels
        for size in movie.fig.get_size_inches():
            self.assertEqual((size * movie.dpi) % 2, 0)

    def test_updateimg(self):
        movie = self.movie()
        changed = movie.update_img(0)
        self.assertTrue(len(changed) > 0)
        self.assertEqual(movie.ims[0].get_array().shape, (3, 2, 4))

    def test_timebartext(self):
        movie = self.movie()
        movie.update_img(0)
        self.assertIn('Slowdown', movie.timebar.get_text())

    def test_frameno(self):
        movie = self.movie(include_frameno=True, include_dt=True)
        movie.update_img(0)
        self.assertIn('Frame: 0', movie.timebar.get_text())


class TestHPanelImgPlusPlotMovie(MovieTestCase):

    def movie(self, **kwargs):
        parameters = dict(imagepannelwidth=1., wavepannelwidth=2.,
                          pannelheight=1., dpi=50, colorbarlabel='mV')
        parameters.update(kwargs)
        movie = HPanelImgPlusPlotMovie(frames(), **parameters)
        movie.framegenerator = movie.fta.generate_frames()
        return movie

    def test_updateimg(self):
        movie = self.movie()
        self.assertTrue(len(movie.update_img(0)) > 0)

    def test_staticimlabel(self):
        # this used to raise NameError: name 'i' is not defined
        movie = self.movie(staticimlabels=['panel'])
        self.assertEqual(movie.staticbar[0].get_text(), 'panel')

    def test_timeindicatormoves(self):
        movie = self.movie()
        movie.update_img(0)
        first = movie.timeindicatorline.get_xdata()[0]
        movie.update_img(1)
        self.assertGreater(movie.timeindicatorline.get_xdata()[0], first)

    def test_multipanelraises(self):
        signal = defs.mcuts_signal()
        fta = FramesTimeAligned([signal, signal], clips=[(-1., 1.)] * 2)
        self.assertRaises(ValueError, HPanelImgPlusPlotMovie, fta)


class TestAxesHelpers(MovieTestCase):

    def test_letteraxes(self):
        fig, ax = plt.subplots()
        letters = LetterAxes(letters=('AB', 'CD'), axes=ax)
        self.assertEqual(len(letters.lettertexts), 2)
        self.assertEqual(len(letters.lettertexts[0]), 2)

    def test_timeindicatedletteraxes(self):
        fig, ax = plt.subplots()
        letters = TimeIndicatedLetterAxes(starttime=0., endtime=10.,
                                          timeindicators=(5.,), axes=ax)
        self.assertEqual(len(letters.timeindicatorlines), 1)
        letters.update(2.5, 0)
        self.assertAlmostEqual(letters.timeindicatorlines[0].get_xdata()[0],
                               0.25)
        letters.init()
        self.assertAlmostEqual(letters.timeindicatorlines[0].get_xdata()[0],
                               0.5)

    def test_specgramaxes(self):
        fig, ax = plt.subplots()
        sound = defs.uts_signal(duration=0.5, fs=2000.)
        specgram = SpecgramAxes(sound, starttime=0., endtime=0.5, axes=ax,
                                nperseg=64, maxfreq=1000.)
        self.assertEqual(specgram.specgram.ndim, 2)
        self.assertEqual(ax.get_ylim(), (0., 1000.))

    def test_timeindicatedspecgramaxes(self):
        fig, ax = plt.subplots()
        sound = defs.uts_signal(duration=0.5, fs=2000.)
        specgram = TimeIndicatedSpecgramAxes(sound, starttime=0., endtime=0.5,
                                             axes=ax, nperseg=64,
                                             maxfreq=1000.)
        specgram.update(0.25, 0)
        self.assertEqual(specgram.timeindicatorline.get_xdata()[0], 0.25)


class TestCodecs(unittest.TestCase):

    def test_definitions(self):
        for name, (arguments, extension) in codecs.items():
            self.assertIsInstance(arguments, tuple)
            self.assertTrue(extension.endswith(('.mp4', '.avi', '.png')))


if __name__ == '__main__':
    unittest.main()
