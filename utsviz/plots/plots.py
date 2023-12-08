from __future__ import print_function

__author__ = 'gabriel'

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec
import matplotlib.cm
import uts
from uts.core.geometry import RectangularLattice
from .core import GridAxes


def plot_uts(s, ax=None, ylim=None, vlines=None, hlines=None, hline=None, vline=None, vlinecolor='k',
             hlinecolor='k', frameon=False, **kwargs):
    # vline hline parameter for backward compatibility
    defaults = {'color': (0.3,0.3,0.3)}
    for key, item in defaults.items():
        if not key in kwargs:
            kwargs[key] = item
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(frameon=frameon)
    pl = ax.plot(s.samplingtimes(),
        s.samples.get(False), **kwargs)
    ax.tick_params(direction='out', top=False, right=False)
    if ylim is None:
        low = np.min(uts.min(s))
        high = np.max(uts.max(s))
        margin = (high - low) * 0.05
        low -= margin
        high += margin
    else:
        low, high = ylim
    ax.set_ylim([low, high])
    if vline is not None:
        vlines = [vline]
    if not vlines is None:
        for vline in vlines:
            ax.axvline(vline, color=vlinecolor)
    if hline is not None:
        hlines = [hline]
    if not hlines is None:
        for hline in hlines:
            ax.axhline(hline, color=hlinecolor)
    return ax


def plot_mcuts(mcuts, ylim=None, xlim=None, wspace=0.06,
               hspace=0.06, left=None, right=None, bottom=None, top=None,
               ylabel=None, xlabel='Time (s)', yticks=(), xticks=(),
               autoyticks=False, ga=None, vlines=None, hlines=None,
               vlinecolor=(0.5, 0.5, 0.5), hlinecolor=(0.5,0.5,0.5),
               vlinewidth=1., hlinewidth=1., frameon=False, **kwargs):
    if ga is None:
        ga = GridAxes(mcuts.geometry, wspace=wspace, hspace=hspace,
                      left=left, right=right,
                      bottom=bottom, top=top,
                      ylabel=ylabel, xlabel=xlabel,
                      yticks=yticks, xticks=xticks,
                      vlines=vlines, vlinecolor=vlinecolor,
                      hlines=hlines, hlinecolor=hlinecolor,
                      vlinewidth=vlinewidth, hlinewidth=hlinewidth,
                      frameon=frameon, fig=None)

    defaults = {'color': (0.3,0.3,0.3)}
    if 'linewidth' in kwargs:
        linewidth = kwargs['linewidth']
    else:
        linewidth = 1.0
    # if plot_params is None:
    #     plot_params = {}
    for key, item in defaults.items():
        if not key in kwargs:
            kwargs[key] = item
    if ylim is None:
        ylow = np.min(uts.min(mcuts, 'time'))
        yhigh = np.max(uts.max(mcuts, 'time'))
        margin = (yhigh - ylow) * 0.05
        ylow -= margin
        yhigh += margin
    else:
        ylow, yhigh = ylim
    samplingtimes = mcuts.samplingtimes()
    if xlim is None:
        xlow = samplingtimes[0]
        xhigh = samplingtimes[-1]
    else:
        xlow, xhigh = xlim
    for i, channelname in enumerate(mcuts.channelnames):
        ax = ga.axes[channelname]
        pl = ax.plot(samplingtimes,
            mcuts[:, channelname].samples.get(False), **kwargs)
        if not np.isnan(np.array([ylow, yhigh])).any():
            ax.set_ylim([ylow, yhigh])
        if not np.isnan(np.array([xlow, xhigh])).any():
            ax.set_xlim([xlow, xhigh])
    if autoyticks and not np.isnan(np.array([ylow, yhigh])).any():
        ga.set_yticks((ylow,yhigh))
    return ga


def plot_emcuts(emcuts, clim=None, wspace=0.06, cmap='hot',
                hspace=0.06, left=None, right=None, bottom=None, top=None,
                ylabel=None, xlabel='Time (s)', yticks=(), xticks=(),
                ga=None, vlines=None, hlines=None, vlinecolor=(0.5, 0.5, 0.5),
                hlinecolor=(0.5,0.5,0.5), vlinewidth=1., hlinewidth=1., frameon=False,
                **kwargs):
    if ga is None:
        ga = GridAxes(emcuts.geometry, wspace=wspace, hspace=hspace,
                      left=left, right=right,
                      bottom=bottom, top=top,
                      ylabel=ylabel, xlabel=xlabel,
                      yticks=yticks, xticks=xticks,
                      vlines=vlines, vlinecolor=vlinecolor,
                      hlines=hlines, hlinecolor=hlinecolor,
                      vlinewidth=vlinewidth, hlinewidth=hlinewidth,
                      frameon=frameon, fig=None)
    defaults = {'aspect': 'auto', 'interpolation': 'nearest'}
    if 'linewidth' in kwargs:
        linewidth = kwargs['linewidth']
    else:
        linewidth = 1.0
    for key, item in defaults.items():
        if not key in kwargs:
            kwargs[key] = item
    if clim is None:
        low = np.min(uts.min(emcuts))
        high = np.max(uts.max(emcuts))
        clim = (low,high)
    extent = list(emcuts.timerange)
    extent.extend([emcuts.nepochs, 0])
    for i, channelname in enumerate(emcuts.channelnames):
        ax = ga.axes[channelname]
        pl = ax.imshow(emcuts[:, channelname].samples.get(False),
                       cmap=cmap, clim=clim, extent=extent, **kwargs)
        # if not vline is None:
        #     ax.axvline(vline, color=vlinecolor, linewidth=linewidth)
        # if not hline is None:
        #     ax.axhline(hline, color=vlinecolor, linewidth=linewidth)
    return ga

# FIXME create a figkwargs and axkwargs
def plot_euts(euts, ax=None, clim=None, cmap='hot',
                ylabel=None, xlabel='Time (s)', yticks=None, xticks=None,
                vlines=None, hlines=None, vlinecolor=(0.5, 0.5, 0.5),
                hlinecolor=(0.5,0.5,0.5), vlinewidth=1., hlinewidth=1.,
                frameon=False, **kwargs):
    defaults = {'aspect': 'auto', 'interpolation': 'nearest'}

    for key, item in defaults.items():
        if not key in kwargs:
            kwargs[key] = item
    if clim is None:
        low = np.min(uts.min(euts))
        high = np.max(uts.max(euts))
        clim = (low,high)
    extent = list(euts.timerange)
    extent.extend([euts.nepochs, 0])
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(frameon=frameon)
    pl = ax.imshow(euts.samples.get(False),
                   cmap=cmap, clim=clim, extent=extent, **kwargs)
    if not vlines is None:
        for vline in vlines:
            ax.axvline(vline, color=vlinecolor, linewidth=vlinewidth)
    if not hlines is None:
        for hline in hlines:
            ax.axhline(hline, color=hlinecolor, linewidth=hlinewidth)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    return ax

def euts_image(euts, ranges=None, clip=None, zeroincenter=False,
               invert=False, colormap=None, includebar=False,
               transparantcolor=None):
    from scipy.stats import scoreatpercentile

    if (not (clip is None)) and (not (ranges is None)):
        raise ValueError("You cannot set ranges and clip parameters at the"
                         " same time")
    if colormap is None:
        colormap = matplotlib.cm.hot
    samples = euts.samples.get(copy=True)
    if euts.epochaxis == 1:
        samples = np.rot90(samples)
    low = samples.min()
    high = samples.max()
    if not clip is None:
        low, high = clip
    if not ranges is None:
        low = scoreatpercentile(samples.flatten(), ranges[0])
        high = scoreatpercentile(samples.flatten(), ranges[1])
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
    colors = colormap(samples)
    if not transparantcolor is None:
        i = (colors[:, :, :3] == transparantcolor)
        j = (i.sum(-1) == 3)
        colors[j, 3] = 0.0
    return colors


# FIXME use plt.imsave
def save_euts_image(euts, basefilename, path='.', ranges=None, clip=None,
                    zeroincenter=False, invert=False, colormap=None,
                    includebar=False, transparantcolor=None):
    """
    ranges: 2-item sequence or None
        The range in *percentiles*, (pmin, pmax).
    clip: 2-item sequence or None
        The values at which clipping should take place (cmin, cmax).

    """
    from scipy.misc import toimage
    from os.path import abspath, join
    from pylab import cm

    if colormap is None:
        colormap = cm.jet
    if (not (clip is None)) and (not (ranges is None)):
        raise ValueError("You cannot set ranges and clip parameters at the"
                         " same time")
    path = abspath(path)
    im = euts_image(euts=euts, ranges=ranges, clip=clip,
                    zeroincenter=zeroincenter, invert=invert,
                    colormap=colormap, includebar=includebar,
                    transparantcolor=transparantcolor)
    im = toimage(im)
    absfilename = join(path, '%s.png' % basefilename)
    im.save(absfilename)
    if includebar:
        if invert:
            bar = np.linspace(1.0, 0.0, 512).reshape(1, -1)
        else:
            bar = np.linspace(0.0, 1.0, 512).reshape(1, -1)
        barim = toimage(colormap(bar))
        absfilename = join(path, '%s_bar.png' % basefilename)
        barim.save(absfilename)
    return absfilename

    ## uts
    # def get_plot(self, labels={}, indexrange=None):
    # from uts.display.singlechannel import waveplot
    #
    # samples = self.samples.get(copy=False)
    #     return waveplot(x=self.samplingtimes(), y=samples,
    #                     xlabel="Time (s)", ylabel="Amplitude",
    #                     title="Wave Plot", labels=labels, indexrange=indexrange)

    ## euts
    # def get_plot(self):
    #        from uts.display.singlechannel import ImagePlot
    #
    #        samples = self.samples.get(copy=False)
    #        if self.epochaxis == 1:
    #            samples = samples.T
    #        x = self.samplingboundarytimes()
    #        y = np.arange(self.nepochs + 1)
    #        return ImagePlot(img=samples, x=x, y=y, xlabel="time (s)",
    #                         ylabel="Epoch", title="")


def plot_spectrum(s, xlim=None, wspace=0.06,
                   hspace=0.1, left=None, right=None, bottom=None, top=None,
                   yticks=None, xticks=None, ga=None, frameon=False, **kwargs):
    defaults = {'color': 'black'}
    for key, item in defaults.items():
        if not key in kwargs:
            kwargs[key] = item
    if ga is None:
        g = RectangularLattice(1, 2, [['magnitude'],['phase']])
        ga = GridAxes(g, wspace=wspace, hspace=hspace,
                      left=left, right=right,
                      bottom=bottom, top=top,
                      ylabel='', xlabel='',
                      yticks=yticks, xticks=xticks, fig=None,
                      frameon=frameon)
    if xlim is None:
        xlim = s.frequencyrange
    ax1 = ga.axes['magnitude']
    samples = np.abs(s.samples.get(False))
    ylim =  (samples.min(), samples.max())
    pl1 = ax1.plot(s.samplingfrequencies, samples, **kwargs)
    ax1.set_ylabel('Magnitude')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_yticks(_ticks(ylim))
    ax1.tick_params(direction='out', top=False, right=False)

    ax2 = ga.axes['phase']
    pl2 = ax2.plot(s.samplingfrequencies,
                   np.angle(s.samples.get(False)), **kwargs)
    ax2.set_ylabel('Phase')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.tick_params(direction='out', top=False, right=False)
    ax2.set_xlim(xlim)
    ax2.set_ylim((-np.pi, np.pi))
    ax2.set_yticks((-np.pi, 0., np.pi))
    ax2.set_xticks(_ticks(xlim))

    return ga


def plot_powerspectrum(s, ax=None, db=True, **kwargs):
    defaults = {'color': (0.3,0.3,0.3)}
    for key, item in defaults.items():
        if not key in kwargs:
            kwargs[key] = item
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    samples = s.samples.get(False) ** 2.
    if db:
        samples = 10 * np.log10(samples)
    pl = ax.plot(s.samplingfrequencies,
                 samples, **kwargs)
    ax.tick_params(direction='out', top=False, right=False)
    return ax

def _ticks(extremes):
    if extremes[0] < 0. and extremes[1] > 0.:
        return (extremes[0], 0., extremes[1])
    else:
        return extremes



def plot_mcuts_savemultipage(signals, starttime, endtime, pageduration,
                               dirpath, ylims=None, colors=None,
                               figsize=(16, 16), wspace=0.1, hspace=0.1,
                               left=0.01, right=0.99, bottom=0.05, top=0.99,
                               figextension='png', transparent=False,
                               mark_episodes=None):
    nframes = int(math.ceil((endtime - starttime) / float(pageduration)))
    nsignals = len(signals)
    if ylims is None:
        ylims = [None] * nsignals
    if colors is None:
        colors = [None] * nsignals
    fstring = '{{:0{}d}}.{}'.format(int(math.ceil(math.log10(nframes))),
                                    figextension)
    if mark_episodes is None:
        mark_episodes = {}
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    for framen in range(nframes):
        fig = plt.figure(figsize=figsize)
        framestarttime = float(framen * pageduration) + starttime
        frameendtime = framestarttime + pageduration
        for (s, ylim, color) in zip(signals, ylims, colors):
            ga = GridAxes(s.geometry, wspace=wspace, hspace=hspace, left=left,
                          right=right, bottom=bottom, top=top,
                          ylabel='', xlabel='Time (s)', yticks=(),
                          xticks=(framestarttime, frameendtime), )
            hasstarttime, hasendtime =  s.hastime([framestarttime, frameendtime])
            if (hasstarttime or hasendtime):
                st = framestarttime if hasstarttime else s.timerange[0]
                et = frameendtime if hasendtime else s.timerange[1]
                plot_mcuts(s[st:et], ylim=ylim, color=color, ga=ga)
        for facecolor, times in mark_episodes.items():
            episodes = []
            for (markstart, markend) in times:
                if (markstart >= framestarttime) and (
                    markend <= frameendtime):  # episode within frame
                    episodes.append((markstart, markend))
                elif (markstart <= framestarttime) and (
                    markend >= frameendtime):  # frame within episode
                    episodes.append((framestarttime, frameendtime))
                elif (
                    markend <= frameendtime):  # episode started before and ends within frame
                    episodes.append((framestarttime, markend))
                elif (
                    markstart <= frameendtime):  # episode started within and ends after frame
                    episodes.append((markstart, frameendtime))

            for (markstart, markend) in episodes:
                dur = markend - markstart
                height = ylim[1] - ylim[0]
                for axname, ax in ga.axes.items():
                    ax.add_patch(
                        patches.Rectangle((markstart, ylim[0]), dur, height,
                                          facecolor=facecolor,
                                          linewidth=0,
                                          alpha=0.15))
        filepath = os.path.join(dirpath, fstring.format(framen))
        fig.savefig(filepath, dpi=300, transparent=transparent)
        plt.close(fig)