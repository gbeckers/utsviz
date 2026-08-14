import numpy as np
# Submodules of scipy are not imported by "import scipy"; the functions used
# below were reached that way and so raised AttributeError.
import scipy.ndimage
import scipy.stats

from utsviz.colormaps import check_colormap, pngcolorbar

__all__ = ['save_pngcolormap', 'tableau20', 'to_image', 'iminterpolate']


def save_pngcolormap(filename, colormap='jet', width=300, height=20):
    return pngcolorbar(filename, colormap=colormap, width=width,
                       height=height)


# FIXME possibility of setting nan to number
def to_image(samples, ranges=None, clip=None, zeroincenter=False,
             invert=False, colormap='jet', includebar=False, nancolor=None):
    if (not (clip is None)) and (not (ranges is None)):
        raise ValueError("You cannot set ranges and clip parameters at the same time")
    colormap = check_colormap(colormap)
    samples = np.array(samples, np.float64)
    low = np.nanmin(samples)
    high = np.nanmax(samples)
    if not clip is None:
        low, high = clip
    if not ranges is None:
        low = scipy.stats.scoreatpercentile(samples.flatten(), ranges[0])
        high = scipy.stats.scoreatpercentile(samples.flatten(), ranges[1])
    np.clip(samples, low, high, samples)
    if zeroincenter:
        high = max(abs(low), abs(high))
        low = -high
    samples -= low
    high -= low
    samples /= high
    if invert:
        samples -= 1.0
        samples[:] = np.absolute(samples)
    if not nancolor is None:
        nans = np.isnan(samples)
    else:
        nans = np.zeros(samples.shape, '?')
    samples[nans] = 0.0
    im = colormap(samples)
    im[nans] = nancolor
    return im


def iminterpolate(im, newxsize, newysize, xyvalranges=(0, 1, 0, 1),
                  newxyvalranges=(0, 1, 0, 1)):
    '''
    Interpolates image (rows,cols) or (rows,cols,channels)
    :param im:
    :param newxsize:
    :param newysize:
    :param xyvalranges:
    :param newxyvalranges:
    '''
    im = np.asarray(im)
    needxinterp = (newxsize != im.shape[0])
    needyinterp = (newysize != im.shape[1])

    if needxinterp or needyinterp:  # we need to interpolate image
        xfrom, xto, yfrom, yto = xyvalranges
        newxfrom, newxto, newyfrom, newyto = newxyvalranges
        # only the first two axes are spatial; unpacking the whole shape
        # failed on the (rows, cols, channels) images this documents
        xsize, ysize = im.shape[:2]

        # scipy.ogrid and scipy.mgrid were aliases of the numpy objects and
        # are gone from the scipy namespace.
        x, y = np.ogrid[xfrom:xto:xsize * 1j, yfrom:yto:ysize * 1j]
        newx, newy = np.mgrid[newxfrom:newxto:newxsize * 1j, newyfrom:newyto:newysize * 1j]
        x0 = x[0, 0]
        y0 = y[0, 0]
        dx = x[1, 0] - x0
        dy = y[0, 1] - y0
        ivals = (newx - x0) / dx
        jvals = (newy - y0) / dy
        coords = np.array([ivals, jvals])
        if im.ndim == 2:
            im = scipy.ndimage.map_coordinates(im, coords)
        else:  # map_coordinates takes one coordinate per axis, so per channel
            im = np.stack([scipy.ndimage.map_coordinates(im[..., i], coords)
                           for i in range(im.shape[2])], axis=-1)
    return im

# def tableau20():
#     colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

#     # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts. 
#     return [(r / 255., g / 255., b / 255.) for r, g, b in colors]
tableau20 = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
             (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
             (1.0, 0.4980392156862745, 0.054901960784313725),
             (1.0, 0.7333333333333333, 0.47058823529411764),
             (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
             (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
             (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
             (1.0, 0.596078431372549, 0.5882352941176471),
             (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
             (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
             (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
             (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
             (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
             (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
             (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
             (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
             (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
             (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
             (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
             (0.6196078431372549, 0.8549019607843137, 0.8980392156862745)]