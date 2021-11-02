import os.path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.image import imsave
from matplotlib.cm import get_cmap
from matplotlib.pyplot import colormaps
from seaborn import diverging_palette

__all__ = ['check_colormap','divmap1', 'pngcolorbar','voltage', 'voltage_r']


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

# nice diverging map use for heatmaps
divmap1 = diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)

# fixme include seaborn maps?
def check_colormap(colormap):
    if isinstance(colormap, 'str'):
        if colormap in colormaps():
            colormap = get_cmap(colormap)
        elif colormap == 'voltage':
            return voltage
        elif colormap == 'voltage_r':
            return voltage_r
        elif colormap == 'divmap1':
            return divmap1
    else:
        return colormap


def pngcolorbar(filename, colormap, width=300, height=20):
    colormap = check_colormap(colormap)
    bar = np.empty((height, width, 4), dtype=np.float64)
    bar[:, ..., :] = colormap(np.linspace(0., 1., width))
    imsave(filename, bar)
    return os.path.abspath(filename)
