import os
import tempfile
import unittest

import numpy as np

from . import defs  # noqa: F401  (selects the Agg backend)
from utsviz.plots.utils import (iminterpolate, save_pngcolormap, tableau20,
                                to_image)


class TestToImage(unittest.TestCase):

    def setUp(self):
        self.samples = np.arange(12, dtype=np.float64).reshape(3, 4)

    def test_shape(self):
        self.assertEqual(to_image(self.samples).shape, (3, 4, 4))

    def test_clipandrangesconflict(self):
        self.assertRaises(ValueError, to_image, self.samples, clip=(0., 1.),
                          ranges=(5., 95.))

    def test_ranges(self):
        # reaches scipy.stats, which "import scipy" alone does not provide
        im = to_image(self.samples, ranges=(5., 95.))
        self.assertEqual(im.shape, (3, 4, 4))

    def test_clip(self):
        im = to_image(self.samples, clip=(2., 8.))
        self.assertEqual(im.shape, (3, 4, 4))

    def test_invertreversesthemap(self):
        im = to_image(self.samples)
        inverted = to_image(self.samples, invert=True)
        self.assertTrue(np.allclose(im[0, 0], inverted[-1, -1]))

    def test_nancolor(self):
        samples = self.samples.copy()
        samples[0, 0] = np.nan
        im = to_image(samples, nancolor=(1., 0., 0., 1.))
        self.assertEqual(tuple(im[0, 0]), (1., 0., 0., 1.))

    def test_colormapname(self):
        self.assertEqual(to_image(self.samples, colormap='divmap1').shape,
                         (3, 4, 4))


class TestIminterpolate(unittest.TestCase):

    def test_grayscale(self):
        im = np.arange(20, dtype=np.float64).reshape(4, 5)
        self.assertEqual(iminterpolate(im, 8, 10).shape, (8, 10))

    def test_rgba(self):
        # unpacking the full shape used to raise ValueError on these
        im = np.zeros((4, 5, 4), dtype=np.float64)
        self.assertEqual(iminterpolate(im, 8, 10).shape, (8, 10, 4))

    def test_samesizeisunchanged(self):
        im = np.arange(20, dtype=np.float64).reshape(4, 5)
        self.assertTrue(np.array_equal(iminterpolate(im, 4, 5), im))

    def test_values(self):
        im = np.zeros((4, 5), dtype=np.float64)
        im[:] = 3.
        self.assertTrue(np.allclose(iminterpolate(im, 8, 10), 3.))


class TestSavePngColormap(unittest.TestCase):

    def test_writespng(self):
        with tempfile.TemporaryDirectory() as dirname:
            filename = os.path.join(dirname, 'bar.png')
            path = save_pngcolormap(filename, width=32, height=4)
            self.assertTrue(os.path.exists(path))


class TestTableau20(unittest.TestCase):

    def test_colors(self):
        self.assertEqual(len(tableau20), 20)
        for color in tableau20:
            self.assertEqual(len(color), 3)


if __name__ == '__main__':
    unittest.main()
