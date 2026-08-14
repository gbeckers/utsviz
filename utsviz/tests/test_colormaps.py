import os
import tempfile
import unittest

import numpy as np
import matplotlib

from . import defs  # noqa: F401  (selects the Agg backend)
from utsviz.colormaps import (check_colormap, divmap1, pngcolorbar, voltage,
                              voltage_r)


class TestCheckColormap(unittest.TestCase):

    def test_utsvizcolormapnames(self):
        self.assertIs(check_colormap('voltage'), voltage)
        self.assertIs(check_colormap('voltage_r'), voltage_r)
        self.assertIs(check_colormap('divmap1'), divmap1)

    def test_matplotlibcolormapname(self):
        # the registry hands out copies, so compare by name
        self.assertEqual(check_colormap('hot').name, 'hot')

    def test_colormapobjectispassedthrough(self):
        self.assertIs(check_colormap(divmap1), divmap1)
        cmap = matplotlib.colormaps['jet']
        self.assertIs(check_colormap(cmap), cmap)

    def test_unknownnameraises(self):
        # this used to return None, which then failed somewhere else
        self.assertRaises(ValueError, check_colormap, 'nosuchcolormap')


class TestColormaps(unittest.TestCase):

    def test_colormapsarecallable(self):
        for cmap in (voltage, voltage_r, divmap1):
            colors = cmap(np.linspace(0., 1., 16))
            self.assertEqual(colors.shape, (16, 4))

    def test_divmap1isdiverging(self):
        # blue-ish at the low end, red-ish at the high end, light in the
        # middle, which is what the seaborn call it was frozen from produced
        low, mid, high = divmap1(0.)[:3], divmap1(0.5)[:3], divmap1(1.)[:3]
        self.assertGreater(low[2], low[0])
        self.assertGreater(high[0], high[2])
        self.assertGreater(sum(mid), sum(low))
        self.assertGreater(sum(mid), sum(high))


class TestPngColorbar(unittest.TestCase):

    def test_writespng(self):
        with tempfile.TemporaryDirectory() as dirname:
            filename = os.path.join(dirname, 'bar.png')
            path = pngcolorbar(filename, 'divmap1', width=32, height=4)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(path, os.path.abspath(filename))

    def test_unknowncolormapraises(self):
        with tempfile.TemporaryDirectory() as dirname:
            filename = os.path.join(dirname, 'bar.png')
            self.assertRaises(ValueError, pngcolorbar, filename, 'nosuchmap')


if __name__ == '__main__':
    unittest.main()
