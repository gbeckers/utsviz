"""A movie figure: a matplotlib figure laid out as the video frame, with panels
that update to a given time."""

import shutil
import subprocess
import sys
import time as _time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .clock import Clock

__all__ = ['Movie', 'codecs']

# ffmpeg output arguments. All use yuv420p, which is what players and browsers
# support; the even frame size it needs is ensured by Movie.
codecs = {
    'h264': ['-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-tune', 'animation'],
    'h264_small': ['-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', '-tune', 'animation'],
    'h264_hq': ['-c:v', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', '-tune', 'animation'],
    'h265': ['-c:v', 'libx265', '-crf', '23', '-pix_fmt', 'yuv420p', '-tag:v', 'hvc1'],
}


def _even(n):
    n = int(round(n))
    return n + (n % 2)


def _inches(pixels, dpi):
    """Figure size in inches that gives exactly `pixels` pixels at `dpi`. pixels / dpi can come
    out a floating-point tick too small (402 / 50 * 50 = 401.99999999999994), and matplotlib
    < 3.8 truncates that to 401 pixels; so take the smallest value that is not too small."""
    inches = pixels / dpi
    while inches * dpi < pixels:
        inches = np.nextafter(inches, np.inf)
    return float(inches)


def _fitsize(rgba, height, width):
    """Crops or pads (with the last row/column) a frame to exactly height x width pixels;
    the canvas can be a pixel off the requested size through rounding."""
    if rgba.shape[:2] == (height, width):
        return rgba
    rgba = rgba[:height, :width]
    return np.pad(rgba, ((0, height - rgba.shape[0]), (0, width - rgba.shape[1]), (0, 0)),
                  mode='edge')


class Movie:
    """A figure laid out as a video frame, with panels that show a moment in time.

    The layout is a matplotlib mosaic (see `matplotlib.figure.Figure.subplot_mosaic`):
    every letter or name in it is an axes, and panels are added to these axes
    with `add`. Several panels can share one axes, e.g. an image and the
    annotations drawn on it.

    Parameters
    ----------
    mosaic: str or nested list
        the layout, e.g. ``'IW\\nCW'``, see matplotlib's `subplot_mosaic`. Use '.' for empty cells.
    clock: Clock
        which signal time each frame shows
    size: (int, int)
        width and height of the video in pixels; rounded up to even numbers,
        which most video codecs need
    dpi: float
        dots per inch; together with `size` this determines the size of text and lines
    width_ratios, height_ratios: sequence of floats or None
        relative widths of the mosaic columns and heights of its rows
    layout: 'constrained' or None
        'constrained' (default): matplotlib positions the axes. The layout is
        fixed before the first frame is written, so axes do not move during the
        movie. None: position the axes yourself, using `gridspec_kw`.
    gridspec_kw: dict or None
        passed on to `subplot_mosaic`, e.g. ``dict(wspace=0.1)``
    facecolor: color
        background of the frame

    Examples
    --------
    >>> clock = Clock.from_signal(eeg, fps=50, slowdown=5)
    >>> mv = Movie('IW\\nCW', clock, size=(1600, 800), dpi=150, width_ratios=(1, 1.3))
    >>> mv.add('I', ElectrodeImage(eeg, vmin=-200, vmax=200, colorbar='C'))
    >>> mv.add('W', Oscillogram(eeg))
    >>> mv.render(5.0)            # a still, for checking
    >>> mv.save('episode.mp4')

    """

    def __init__(self, mosaic, clock, size=(1280, 720), dpi=100, width_ratios=None,
                 height_ratios=None, layout='constrained', gridspec_kw=None,
                 facecolor='white'):
        if not isinstance(clock, Clock):
            raise TypeError(f'clock should be a Clock, not {type(clock)}')
        self.clock = clock
        self.dpi = float(dpi)
        self.size = (_even(size[0]), _even(size[1]))
        self.facecolor = facecolor
        figsize = (_inches(self.size[0], self.dpi), _inches(self.size[1], self.dpi))
        self.fig = plt.figure(figsize=figsize, dpi=self.dpi, facecolor=facecolor, layout=layout)
        self.axes = self.fig.subplot_mosaic(mosaic, width_ratios=width_ratios,
                                            height_ratios=height_ratios,
                                            gridspec_kw=gridspec_kw)
        for ax in self.axes.values():  # cells without a panel stay empty
            ax.set_axis_off()
        self.panels = []
        self.time = None  # signal time currently shown
        self.frameno = None  # frame number currently shown, if the time is that of a frame

    def __getitem__(self, key):
        return self.axes[key]

    def __repr__(self):
        return f'<Movie {self.size[0]}x{self.size[1]} px, {len(self.panels)} panels, {self.clock!r}>'

    def add(self, key, panel):
        """Adds a panel to the axes named `key` in the mosaic and returns the panel.

        `key` can also be a matplotlib Axes of the figure.
        """
        ax = self.axes[key] if not isinstance(key, matplotlib.axes.Axes) else key
        if not any(p.ax is ax for p in self.panels):  # the first panel decides, e.g. images hide the axis
            ax.set_axis_on()
        panel.setup(ax, self)
        self.panels.append(panel)
        return panel

    def sharex(self, *keys):
        "Makes the axes named by `keys` share their x axis (time) with the first one."
        first = self.axes[keys[0]]
        for key in keys[1:]:
            self.axes[key].sharex(first)

    def update(self, time):
        """Sets all panels to signal time `time`. Returns the artists that changed."""
        self.time = time
        k = self.clock.frame(time)
        self.frameno = k if np.isclose(self.clock.time(k), time) else None
        artists = []
        for panel in self.panels:
            artists.extend(panel.update(time))
        return artists

    def render(self, time=None, frame=None):
        """Shows the movie at signal `time` (or at frame number `frame`) in the figure, for
        checking a frame before writing the whole movie. Returns the figure."""
        if frame is not None:
            time = self.clock.time(frame)
        elif time is None:
            time = self.clock.starttime
        self.update(time)
        self.fig.canvas.draw_idle()
        return self.fig

    def view(self, time=None, **kwargs):
        """Shows the movie interactively in a window, with a slider; see `Viewer` for the
        controls and `kwargs`. Returns the Viewer; call its `show` to open the window."""
        from .viewer import Viewer
        return Viewer(self, time=time, **kwargs)

    def savefig(self, path, time=None, frame=None, **kwargs):
        "Saves a single frame as an image, at the exact pixel size of the video."
        self.render(time=time, frame=frame)
        kwargs.setdefault('facecolor', self.facecolor)
        self.fig.savefig(path, dpi=self.dpi, **kwargs)

    def _animated_artists(self):
        artists = []
        for panel in self.panels:
            artists.extend(panel.animated_artists())
        return artists

    def iter_frames(self, frames=None):
        """Yields (frame number, RGBA array) for each frame, rendered without a display.

        Only the artists that change are drawn per frame, on top of a background
        that is drawn once; this is much faster than redrawing the whole figure.

        Parameters
        ----------
        frames: iterable of int or None
            frame numbers; default all
        """
        if frames is None:
            frames = range(len(self.clock))
        fig = self.fig
        origcanvas = fig.canvas
        canvas = FigureCanvasAgg(fig)  # independent of the GUI backend and screen dpi
        animated = self._animated_artists()
        try:
            # draw once with the layout engine, to fix the axes positions
            self.update(self.clock.starttime)
            canvas.draw()
            engine = fig.get_layout_engine()
            fig.set_layout_engine('none')
            for a in animated:
                a.set_animated(True)
            canvas.draw()
            background = canvas.copy_from_bbox(fig.bbox)
            for k in frames:
                canvas.restore_region(background)
                for a in self.update(self.clock.time(k)):
                    fig.draw_artist(a)
                yield k, np.asarray(canvas.buffer_rgba())
        finally:
            for a in animated:
                a.set_animated(False)
            fig.set_layout_engine(engine)
            fig.set_canvas(origcanvas)

    def save(self, path, codec='h264', frames=None, overwrite=False, progress=True,
             ffmpeg=None, extra_args=None):
        """Writes the movie to a video file with ffmpeg.

        Parameters
        ----------
        path: str or Path
            output file, e.g. 'episode.mp4'
        codec: str
            one of `codecs`: 'h264' (default), 'h264_small', 'h264_hq', 'h265'
        frames: iterable of int or None
            frame numbers to write; default all
        overwrite: bool
            overwrite an existing file
        progress: bool
            print progress
        ffmpeg: str or None
            ffmpeg executable; default matplotlib's ``rcParams['animation.ffmpeg_path']``
        extra_args: list of str or None
            ffmpeg output arguments in place of those of `codec`
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"'{path}' exists; use overwrite=True to overwrite it")
        ffmpeg = ffmpeg or matplotlib.rcParams['animation.ffmpeg_path']
        if shutil.which(ffmpeg) is None:
            raise FileNotFoundError(f"ffmpeg executable '{ffmpeg}' not found; install ffmpeg or set "
                                    "matplotlib.rcParams['animation.ffmpeg_path']")
        outargs = list(codecs[codec]) if extra_args is None else list(extra_args)
        width, height = self.size
        cmd = [ffmpeg, '-y', '-loglevel', 'error',
               '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{width}x{height}',
               '-r', f'{self.clock.fps}', '-i', '-', *outargs, str(path)]
        frames = range(len(self.clock)) if frames is None else list(frames)
        nframes = len(frames)
        t0 = _time.time()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i, (k, rgba) in enumerate(self.iter_frames(frames)):
                proc.stdin.write(_fitsize(rgba, height, width).tobytes())
                if progress and (i % 100 == 0 or i == nframes - 1):
                    print(f'\rframe {i + 1}/{nframes}  ({_time.time() - t0:.0f} s)', end='',
                          file=sys.stderr, flush=True)
        except BrokenPipeError:  # ffmpeg quit; its error message is raised below
            pass
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            errors = proc.stderr.read().decode(errors='replace')
            proc.stderr.close()
            returncode = proc.wait()
        if progress:
            print(file=sys.stderr)
        if returncode != 0:
            raise RuntimeError(f'ffmpeg failed (exit code {returncode}):\n{errors}')
        return path

    def close(self):
        plt.close(self.fig)
