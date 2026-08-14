"""Colormaps used for uts signal visualization."""

import os.path

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imsave

__all__ = ['check_colormap', 'divmap1', 'pngcolorbar', 'voltage', 'voltage_r']


class VoltageMap(LinearSegmentedColormap):

    __version__ = '20150719'

    cdict = {'red':   ((0.0, 0.5, 0.5),
                       (0.25, 0.3, 0.3),
                       (0.5, 0.0, 0.0),
                       (0.8, 0.88 ,0.88),
                       (1.0, 1.0, 1.0)),
             'green': ((0.0, 1.0, 1.0),
                       (0.25, 0.3, 0.3),
                       (0.5, 0.0, 0.0),
                       (0.76, 0.2,0.2),
                       (1.0, 0.95, 0.95)),
             'blue': ((0.0, 1.0, 1.0),
                     (0.25, 1.0, 1.0),
                     (0.5, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}

    def __init__(self):
        LinearSegmentedColormap.__init__(self, 'voltage', self.cdict, 256)


class VoltageMapR(LinearSegmentedColormap):

    __version__ = '20150719'

    cdict = {'red':   ((0.0, 1.0, 1.0),
                       (0.2, 0.88, 0.88),
                       (0.5, 0.0, 0.0),
                       (0.75, 0.3 ,0.3),
                       (1.0, 0.5, 0.5)),
             'green': ((0.0, 0.95, 0.95),
                       (0.24, 0.2, 0.2),
                       (0.5, 0.0, 0.0),
                       (0.75, 0.3,0.3),
                       (1.0, 1.0, 1.0)),
             'blue':  ((0.0, 0.0, 0.0),
                      (0.5, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.0, 1.0, 1.0))}

    def __init__(self):
        LinearSegmentedColormap.__init__(self, 'voltage_r', self.cdict, 256)


voltage = VoltageMap()
voltage_r = VoltageMapR()

# A nice diverging map, used for heatmaps. Originally produced at import time
# by seaborn:
#
#     divmap1 = seaborn.diverging_palette(h_neg=210, h_pos=350, s=90, l=30,
#                                         as_cmap=True)
#
# The 256 colors that call returns are frozen below, so that the only thing
# utsviz needed seaborn (and, through it, pandas) for is gone. seaborn 0.12.2
# and 0.13.2 both produce exactly these colors; they are stored 8-bit
# quantized, which is the precision a colormap is rendered at anyway.
_divmap1_hex = (
    '134e57154f5817505918525a1a535c1c545d1e555e1f575f215860235962'
    '245b63265c64285d652a5e672b60682d61692f626a30646c32656d34666e'
    '36686f376970396a723b6b733c6d743e6e75406f7742717843727945737a'
    '47747b48767d4a777e4c787f4e7a804f7b82517c83537d84547f85568087'
    '5881885a83895b848a5d858b5f868d60888e62898f648a90668c92678d93'
    '698e946b8f956c91966e929870939972959a73969b75979d77989e789a9f'
    '7a9ba07c9ca27e9ea3809fa482a1a684a2a785a3a887a5aa89a6ab8aa7ac'
    '8ca9ad8eaaae90abb091acb193aeb295afb396b0b598b2b69ab3b79cb4b8'
    '9db5b99fb7bba1b8bca2b9bda4bbbea6bcc0a8bdc1a9bfc2abc0c3adc1c5'
    'aec2c6b0c4c7b2c5c8b4c6c9b5c8cbb7c9ccb9cacdbbcbcebccdd0beced1'
    'c0cfd2c1d1d3c3d2d4c5d3d6c7d4d7c8d6d8cad7d9ccd8dbcddadccfdbdd'
    'd1dcded3dde0d4dfe1d6e0e2d8e1e3d9e3e4dbe4e6dde5e7dfe6e8e0e8e9'
    'e2e9ebe4eaece5ecede7edeee9eeefebeff1ecf1f2f0f2f2f2f1f1f2eff0'
    'f1edeff0ecedf0eaecefe8ebeee6eaede5e8ece3e7ebe1e6ebdfe4eadee3'
    'e9dce2e8dae1e7d9dfe6d7dee6d5dde5d3dbe4d2dae3d0d9e2ced8e1ccd6'
    'e1cbd5e0c9d4dfc7d2dec5d1ddc4d0dcc2cedbc0cddbbeccdabdcbd9bbc9'
    'd8b9c8d7b8c7d6b6c5d6b4c4d5b2c3d4b1c2d3afc0d2adbfd1abbed1aabc'
    'd0a8bbcfa6bacea4b9cda3b7cca1b6cc9fb5cb9db3ca9cb2c99ab1c898af'
    'c796aec695adc693acc591aac490a9c38ea8c28ca6c18aa5c189a4c087a3'
    'bf85a1be83a0bd819ebc7f9dbb7d9bba7c9ab97a99b97898b87696b77595'
    'b67394b57192b46f91b46e90b36c8fb26a8db1688cb0678baf6589af6388'
    'ae6187ad6086ac5e84ab5c83aa5b82aa5980a9577fa8557ea7547da6527b'
    'a5507aa44e79a44d77a34b76a24975a14773a046729f44719f42709e406e'
    '9d3f6d9c3d6c9b3b6a9a3a699a3868993667983465973364963163952f61'
    '952d60942c5f932a5e92285c91265b90255a8f23588f21578e1f568d1e55'
    '8c1c538b1a528a19518a174f89154e88134d')

divmap1 = LinearSegmentedColormap.from_list(
    'divmap1',
    ['#' + _divmap1_hex[i:i + 6] for i in range(0, len(_divmap1_hex), 6)],
    N=256)

# Colormaps that utsviz defines itself, and that are therefore not in
# matplotlib's registry.
_localcolormaps = {'voltage': voltage,
                   'voltage_r': voltage_r,
                   'divmap1': divmap1}


def check_colormap(colormap):
    """Return a colormap object for `colormap`.

    Accepts a colormap object, which is returned unchanged, or the name of
    one of the utsviz colormaps ('voltage', 'voltage_r', 'divmap1') or of a
    colormap registered with matplotlib.

    """
    if not isinstance(colormap, str):
        return colormap
    if colormap in _localcolormaps:
        return _localcolormaps[colormap]
    if colormap in matplotlib.colormaps:
        # matplotlib.cm.get_cmap, which this used to call, was removed in
        # matplotlib 3.9.
        return matplotlib.colormaps[colormap]
    raise ValueError("'{}' is not a known colormap".format(colormap))


def pngcolorbar(filename, colormap, width=300, height=20):
    """Save a horizontal color bar for `colormap` as a png file."""
    colormap = check_colormap(colormap)
    bar = np.empty((height, width, 4), dtype=np.float64)
    bar[:, ..., :] = colormap(np.linspace(0., 1., width))
    imsave(filename, bar)
    return os.path.abspath(filename)
