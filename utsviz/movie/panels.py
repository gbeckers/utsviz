"""Panels: the contents of the axes of a Movie.

A panel is set up once (`setup`), drawing everything that does not change, and
is then asked for each frame to show a signal time (`update`). `update`
returns the artists it changed; `animated_artists` lists all artists that can
change, so that the movie can draw everything else once, as a background.

To write a new kind of panel, subclass `Panel` and implement `draw` and
`update`.

Note: when a movie is written, everything that is not returned by `draw` as changing is drawn
once, as a background, and the changing artists are drawn on top of it in every frame. A static
artist that should be visible on top of a changing one in the same axes (e.g. a line on an
`ElectrodeImage`) must therefore be part of a panel and returned as changing, too.
"""

import matplotlib
import matplotlib.collections
import matplotlib.colors
import matplotlib.patches
import numpy as np

from ..colormaps import check_colormap

__all__ = ['Panel', 'ElectrodeImage', 'Oscillogram', 'StateBar', 'TimeText', 'SiteMarkers',
           'Trajectory']


class Panel:
    "Base class of movie panels."

    timeaxis = False  # True if the x axis of the panel's axes is (display) time, as in Oscillogram

    def setup(self, ax, movie):
        "Called by `Movie.add`: draws the panel in `ax`."
        self.ax = ax
        self.movie = movie
        self.clock = movie.clock
        self._animated = list(self.draw(ax) or ())

    def draw(self, ax):
        "Draws the static content and creates the artists that change; returns the latter."
        return ()

    def update(self, time):
        "Shows signal time `time`; returns the artists that changed."
        return ()

    def animated_artists(self):
        return self._animated

    def displaytime(self, time):
        return self.clock.displaytime(time)


def _sampleindex(signal, time):
    "Index of the sample of a uts time series whose period contains `time`, clipped to range."
    i = int(np.floor((time - signal.starttime) * signal.fs))
    return min(max(i, 0), signal.ntimesamples - 1)


def _channelsamples(signal, channel):
    "All samples of one channel as a 1-D float array."
    return np.asarray(signal[:, channel].samples[:], dtype='float64').ravel()


class ElectrodeImage(Panel):
    """The values of a multichannel signal at the current time, as an image in which every
    pixel is an electrode, laid out as in the geometry of the signal.

    The image has data coordinates in which the centre of the pixel of the
    electrode at grid row `r` and column `c` is at (x, y) = (c, r), row 0 at the
    top. `channelxy` gives these coordinates for channel names, for drawing
    annotations on the image.

    Parameters
    ----------
    signal: uts MultiChannelUniformTimeSeries with a RectangularLattice geometry
    vmin, vmax: float
        values at the ends of the colour map; values outside are shown in the end colours
    cmap: str or Colormap
        default utsviz's `voltage_r`
    nancolor: color or None
        colour of electrodes without data (NaN), and of grid points without
        electrode; None shows the axes background
    interpolation: str
        matplotlib imshow interpolation; 'nearest' (default) shows the electrodes
        as squares
    aspect: 'equal', 'auto' or float
        see matplotlib imshow
    colorbar: mosaic key, Axes or None
        the axes in which to draw a horizontal colour bar
    colorbarlabel: str
    colorbarsize: (float, float)
        width and height of the colour bar as a fraction of its axes; it is
        centred in its axes
    colorbarticks: sequence of floats or None
        default (vmin, vmax)

    """

    def __init__(self, signal, vmin, vmax, cmap='voltage_r', nancolor=None,
                 interpolation='nearest', aspect='equal', colorbar=None, colorbarlabel='',
                 colorbarsize=(0.8, 0.3), colorbarticks=None):
        self.signal = signal
        self.norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = check_colormap(cmap)
        if nancolor is not None:
            cmap = cmap.with_extremes(bad=nancolor)
        self.cmap = cmap
        self.interpolation = interpolation
        self.aspect = aspect
        self.colorbar = colorbar
        self.colorbarlabel = colorbarlabel
        self.colorbarsize = colorbarsize
        self.colorbarticks = (vmin, vmax) if colorbarticks is None else colorbarticks
        g = signal.geometry
        positions = np.array([g[ch] for ch in signal.channelnames])  # (row, col)
        self._rows, self._cols = positions.T
        self.gridshape = (self._rows.max() + 1, self._cols.max() + 1)
        self._channelpos = dict(zip(signal.channelnames, zip(self._cols, self._rows)))
        self._timeaxis = signal.timeaxis
        self._channelaxis = signal.channelaxis

    def channelxy(self, channels):
        "Image (x, y) coordinates of the electrode of a channel name, or of a list of them."
        if isinstance(channels, str):
            return self._channelpos[channels]
        return np.array([self._channelpos[ch] for ch in channels]).reshape(-1, 2)

    def values(self, time):
        "Values of all channels at signal time `time`, in the order of `signal.channelnames`."
        index = [slice(None)] * 2
        index[self._timeaxis] = _sampleindex(self.signal, time)
        return np.asarray(self.signal.samples[tuple(index)], dtype='float64')

    def grid(self, time):
        "Values at `time` on the electrode grid, NaN where there is no electrode."
        grid = np.full(self.gridshape, np.nan)
        grid[self._rows, self._cols] = self.values(time)
        return grid

    def draw(self, ax):
        ax.set_axis_off()
        self.image = ax.imshow(np.full(self.gridshape, np.nan), cmap=self.cmap, norm=self.norm,
                               interpolation=self.interpolation, aspect=self.aspect,
                               origin='upper')
        if self.colorbar is not None:
            cbax = self.colorbar
            if not isinstance(cbax, matplotlib.axes.Axes):
                cbax = self.movie[cbax]
            cbax.set_axis_off()
            w, h = self.colorbarsize
            cax = cbax.inset_axes([(1 - w) / 2, (1 - h) / 2, w, h])
            self.cbar = ax.figure.colorbar(self.image, cax=cax, orientation='horizontal',
                                           ticks=self.colorbarticks)
            self.cbar.set_label(self.colorbarlabel, labelpad=-10)
        return (self.image,)

    def update(self, time):
        self.image.set_data(self.grid(time))
        return (self.image,)


class Oscillogram(Panel):
    """Traces of the channels of a signal stacked vertically, with a cursor at the current time.

    Parameters
    ----------
    signal: uts UniformTimeSeries or MultiChannelUniformTimeSeries
    channels: sequence of channel names or None
        channels to plot, top to bottom; default all, in the order of the signal
    spacing: float or None
        vertical distance between the zero lines of neighbouring traces, in
        signal units after `scale`; default 6 times the median standard
        deviation of the channels
    scale: float
        the traces are multiplied by this
    margin: float or None
        space above the first and below the last trace's zero line, in signal
        units after `scale`; default `spacing`
    ylabels: None, True, sequence or dict
        labels at the zero lines: True for the channel names, a sequence of
        labels in the order of `channels` ('' for none), or a dict channel -> label
    color, linewidth: trace colour and width
    cursor: bool
        draw a vertical line at the current time
    cursorcolor, cursorwidth: of the cursor
    window: float or None
        None: show the whole time range of the clock, the cursor moves. A float:
        show a window of this many seconds around the current time, the traces move
    xlabel: str
    xticks: sequence of floats or None
        ticks in display time; None for matplotlib's default
    label: str or None
        a label at the left, centred vertically (e.g. 'Acc')

    """

    timeaxis = True

    def __init__(self, signal, channels=None, spacing=None, scale=1., margin=None, ylabels=None,
                 color=(0.2, 0.2, 0.2), linewidth=0.75, cursor=True, cursorcolor='#F97306',
                 cursorwidth=1., window=None, xlabel='Time (s)', xticks=None, label=None):
        self.signal = signal
        if channels is None:
            channels = list(signal.channelnames) if hasattr(signal, 'channelnames') else [None]
        self.channels = list(channels)
        self.scale = scale
        self.spacing = spacing
        self.margin = margin
        self.ylabels = ylabels
        self.color = color
        self.linewidth = linewidth
        self.cursor = cursor
        self.cursorcolor = cursorcolor
        self.cursorwidth = cursorwidth
        self.window = window
        self.xlabel = xlabel
        self.xticks = xticks
        self.label = label

    def _traces(self):
        if self.channels == [None]:
            return [np.asarray(self.signal.samples[:], dtype='float64').ravel() * self.scale]
        return [_channelsamples(self.signal, ch) * self.scale for ch in self.channels]

    def _ylabels(self):
        if self.ylabels is None:
            return None
        if self.ylabels is True:
            return [str(ch) for ch in self.channels]
        if isinstance(self.ylabels, dict):
            return [self.ylabels.get(ch, '') for ch in self.channels]
        if len(self.ylabels) != len(self.channels):
            raise ValueError('ylabels should have one label per channel')
        return list(self.ylabels)

    def draw(self, ax):
        traces = self._traces()
        spacing = self.spacing
        if spacing is None:
            sds = [np.nanstd(tr) for tr in traces if np.isfinite(tr).any()]
            spacing = 6 * np.median(sds) if sds and np.median(sds) > 0 else 1.
        margin = spacing if self.margin is None else self.margin
        times = self.displaytime(self.signal.samplingtimes())
        self.offsets = -spacing * np.arange(len(traces))  # top trace at 0
        for trace, offset in zip(traces, self.offsets):
            ax.plot(times, trace + offset, color=self.color, linewidth=self.linewidth)
        ax.set_ylim(self.offsets[-1] - margin, margin)
        ax.set_xlim(*self.clock.displayrange)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        labels = self._ylabels()
        if labels is None:
            ax.set_yticks([])
        else:
            keep = [i for i, lb in enumerate(labels) if lb != '']
            ax.set_yticks(self.offsets[keep], [labels[i] for i in keep])
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.xticks is not None:
            ax.set_xticks(self.xticks)
        if self.label is not None:
            ax.set_ylabel(self.label, rotation=0, ha='right', va='center')
        animated = []
        if self.cursor:
            self.cursorline = ax.axvline(self.clock.displayrange[0], color=self.cursorcolor,
                                         linewidth=self.cursorwidth)
            animated.append(self.cursorline)
        return animated

    def update(self, time):
        t = self.displaytime(time)
        if self.cursor:
            self.cursorline.set_xdata([t, t])
        if self.window is not None:
            self.ax.set_xlim(t - self.window / 2, t + self.window / 2)
            return (self.ax,)
        return (self.cursorline,) if self.cursor else ()

    def animated_artists(self):
        if self.window is not None:  # everything in the axes moves
            return [self.ax]
        return self._animated


class StateBar(Panel):
    """A horizontal bar showing which state (e.g. sleep stage) the signal is in over time.

    Parameters
    ----------
    intervals: dict or sequence
        either a dict state -> list of (starttime, endtime), or a sequence of
        (starttime, endtime, state); times in display time (i.e. relative to the
        clock's origin) if `displaytimes` is True, else in signal time
    colors: dict
        state -> colour; a pair of colours draws the bottom half in the first
        and the top half in the second colour (e.g. for a transitional state)
    displaytimes: bool
    cursor: bool
        draw a vertical line at the current time
    cursorcolor, cursorwidth: of the cursor
    label: str or None
        a label at the left

    """

    timeaxis = True

    def __init__(self, intervals, colors, displaytimes=True, cursor=False, cursorcolor='#F97306',
                 cursorwidth=1., label=None):
        if isinstance(intervals, dict):
            intervals = [(start, end, state) for state, ses in intervals.items()
                         for start, end in ses]
        self.intervals = list(intervals)
        self.colors = colors
        self.displaytimes = displaytimes
        self.cursor = cursor
        self.cursorcolor = cursorcolor
        self.cursorwidth = cursorwidth
        self.label = label

    def draw(self, ax):
        ax.set_axis_on()
        for start, end, state in self.intervals:
            if not self.displaytimes:
                start, end = self.displaytime(start), self.displaytime(end)
            color = self.colors[state]
            if matplotlib.colors.is_color_like(color):
                ax.axvspan(start, end, 0, 1, facecolor=color, edgecolor='none')
            else:
                bottom, top = color
                ax.axvspan(start, end, 0, 0.5, facecolor=bottom, edgecolor='none')
                ax.axvspan(start, end, 0.5, 1, facecolor=top, edgecolor='none')
        ax.set_xlim(*self.clock.displayrange)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if self.label is not None:
            ax.set_ylabel(self.label, rotation=0, ha='right', va='center')
        if self.cursor:
            self.cursorline = ax.axvline(self.clock.displayrange[0], color=self.cursorcolor,
                                         linewidth=self.cursorwidth)
            return (self.cursorline,)
        return ()

    def update(self, time):
        if not self.cursor:
            return ()
        t = self.displaytime(time)
        self.cursorline.set_xdata([t, t])
        return (self.cursorline,)


class TimeText(Panel):
    """Text showing the current time, e.g. '8.002 s    Slowdown: 5.00x'.

    Parameters
    ----------
    template: str
        format string; available fields: `t` (display time, s), `time` (signal
        time, s), `frame` (frame number), `slowdown`, `fps`, and `datetime` if
        `signal` is given
    xy: (float, float)
        position in axes coordinates
    signal: uts time series or None
        to convert times to date-times
    kwargs: passed on to matplotlib `text`, e.g. fontsize, ha, va

    """

    def __init__(self, template='{t:.3f} s    Slowdown: {slowdown:.2f}x', xy=(0., 0.5),
                 signal=None, **kwargs):
        self.template = template
        self.xy = xy
        self.signal = signal
        kwargs.setdefault('ha', 'left')
        kwargs.setdefault('va', 'center')
        self.kwargs = kwargs

    def draw(self, ax):
        ax.set_axis_off()
        self.text = ax.text(*self.xy, '', transform=ax.transAxes, **self.kwargs)
        self.text.set_in_layout(False)  # a long text should not change the layout of the axes
        return (self.text,)

    def update(self, time):
        fields = dict(t=self.displaytime(time), time=time, frame=self.movie.frameno,
                      slowdown=self.clock.slowdown, fps=self.clock.fps)
        if self.signal is not None:
            fields['datetime'] = self.signal.time_to_datetime(time)
        self.text.set_text(self.template.format(**fields))
        return (self.text,)


def _timeindex(time, fs, starttime, n):
    "Index of the sample whose period contains `time`, or None outside the samples."
    i = int(np.floor((time - starttime) * fs))
    return i if 0 <= i < n else None


class SiteMarkers(Panel):
    """Outlines the sites (e.g. electrodes) that are active at the current time, on an image in which
    the site at grid row r and column c is centred at (x, y) = (c, r), as in `ElectrodeImage`.

    Add it to the same axes as the image. The sites need not be electrodes: positions can be in
    between grid points (e.g. bipolar derivations of neighbouring electrodes).

    Parameters
    ----------
    active: bool array (time, site)
        which sites are active at each sample
    positions: array (site, 2)
        (x, y) = (column, row) of each site; see `ElectrodeImage.channelxy` for electrodes
    fs: float
        sampling rate of `active`
    starttime: float
        signal time of the first sample of `active`
    shape: 'square' or 'circle'
    size: float
        width of the outline in grid units (1 = one electrode)
    color, linewidth: of the outline

    """

    def __init__(self, active, positions, fs, starttime=0., shape='square', size=0.9,
                 color='white', linewidth=2.):
        self.active = np.asarray(active, dtype=bool)
        self.positions = np.asarray(positions, dtype='float64').reshape(-1, 2)
        if self.active.ndim != 2 or self.active.shape[1] != len(self.positions):
            raise ValueError(f'active should be (time, site) with {len(self.positions)} sites, '
                             f'not {self.active.shape}')
        self.fs = float(fs)
        self.starttime = float(starttime)
        if shape not in ('square', 'circle'):
            raise ValueError(f"shape should be 'square' or 'circle', not {shape!r}")
        self.shape = shape
        self.size = size
        self.color = matplotlib.colors.to_rgba(color)
        self.linewidth = linewidth

    def draw(self, ax):
        h = self.size / 2
        if self.shape == 'square':
            patches = [matplotlib.patches.Rectangle((x - h, y - h), self.size, self.size)
                       for x, y in self.positions]
        else:
            patches = [matplotlib.patches.Circle((x, y), h) for x, y in self.positions]
        self.collection = matplotlib.collections.PatchCollection(
            patches, facecolor='none', edgecolor=(0, 0, 0, 0), linewidth=self.linewidth, zorder=3)
        ax.add_collection(self.collection)
        return (self.collection,)

    def activesites(self, time):
        "Boolean array of the sites active at signal time `time` (none outside the samples)."
        i = _timeindex(time, self.fs, self.starttime, len(self.active))
        return self.active[i] if i is not None else np.zeros(len(self.positions), dtype=bool)

    def update(self, time):
        colors = np.zeros((len(self.positions), 4))
        colors[self.activesites(time)] = self.color
        self.collection.set_edgecolor(colors)
        return (self.collection,)


class Trajectory(Panel):
    """The recent path of a moving point, e.g. the centroid of a wave on the electrode grid: a line
    through its positions in the last `tail` seconds that fades towards the past, and a marker at
    the current position.

    Parameters
    ----------
    xy: array (time, 2)
        (x, y) per sample, in the data coordinates of the axes (for an `ElectrodeImage`: (column,
        row)); NaN where there is no point (the path is interrupted there)
    fs: float
        sampling rate of `xy`
    starttime: float
        signal time of the first sample of `xy`
    tail: float
        duration of the path that is shown (s)
    color, linewidth: of the path
    marker, markersize: of the current position; marker None for none

    """

    def __init__(self, xy, fs, starttime=0., tail=0.1, color='white', linewidth=2., marker='o',
                 markersize=8.):
        self.xy = np.asarray(xy, dtype='float64').reshape(-1, 2)
        self.fs = float(fs)
        self.starttime = float(starttime)
        self.ntail = max(1, int(round(tail * self.fs)))
        self.color = matplotlib.colors.to_rgba(color)
        self.linewidth = linewidth
        self.marker = marker
        self.markersize = markersize

    def draw(self, ax):
        self.path = matplotlib.collections.LineCollection([], linewidths=self.linewidth,
                                                          capstyle='round', zorder=4)
        ax.add_collection(self.path)
        artists = [self.path]
        if self.marker is not None:
            self.head, = ax.plot([], [], linestyle='none', marker=self.marker,
                                 markersize=self.markersize, color=self.color, zorder=5)
            artists.append(self.head)
        return artists

    def update(self, time):
        i = _timeindex(time, self.fs, self.starttime, len(self.xy))
        if i is None:
            segments, colors, head = [], [], None
        else:
            pts = self.xy[max(0, i - self.ntail):i + 1]
            segments = [pts[k:k + 2] for k in range(len(pts) - 1)
                        if np.isfinite(pts[k:k + 2]).all()]
            ages = [len(pts) - 2 - k for k in range(len(pts) - 1)
                    if np.isfinite(pts[k:k + 2]).all()]  # 0 for the newest segment
            colors = [(*self.color[:3], self.color[3] * (1 - a / self.ntail)) for a in ages]
            head = self.xy[i] if np.isfinite(self.xy[i]).all() else None
        self.path.set_segments(segments)
        self.path.set_color(colors if colors else [(0, 0, 0, 0)])
        artists = [self.path]
        if self.marker is not None:
            self.head.set_data(*(([head[0]], [head[1]]) if head is not None else ([], [])))
            artists.append(self.head)
        return artists
