"""The time base of a movie: which signal time each video frame shows."""

import numpy as np

__all__ = ['Clock']


class Clock:
    """Maps video frame numbers to signal times.

    Frame `k` shows signal time ``starttime + k / (fps * slowdown)``. Panels only
    ever receive signal times, never frame numbers, so that a single frame can
    be rendered for any time, and all panels of a movie stay in sync.

    No resampling is involved: a panel that shows a sampled signal shows the
    sample at the frame time. With `slowdown` chosen such that ``fs / (fps *
    slowdown)`` is 1, every sample gets exactly one frame. If it is larger,
    samples in between frames are skipped (not averaged); low-pass filter
    the signal first if that matters.

    Parameters
    ----------
    starttime: float
        signal time of the first frame (s)
    endtime: float
        the last frame is the last one with a time smaller than this (s)
    fps: float
        frames per second of the video
    slowdown: float
        factor by which the video is slower than real time
    origin: float
        signal time that is shown as 0 in the movie (time axes, time text). Use
        e.g. the start of an episode, or the time of an event of interest, so
        that the movie shows times relative to it.
    timerange: (float, float) or None
        the stretch of signal time the movie covers, used as the default x
        limits of time axes. Default: (starttime, endtime).

    """

    def __init__(self, starttime, endtime, fps=25., slowdown=1., origin=0., timerange=None):
        if endtime <= starttime:
            raise ValueError(f'endtime ({endtime}) should be larger than starttime ({starttime})')
        if fps <= 0 or slowdown <= 0:
            raise ValueError('fps and slowdown should be positive')
        self.starttime = float(starttime)
        self.endtime = float(endtime)
        self.fps = float(fps)
        self.slowdown = float(slowdown)
        self.origin = float(origin)
        if timerange is None:
            timerange = (starttime, endtime)
        self.timerange = (float(timerange[0]), float(timerange[1]))
        self.dt = 1. / (self.fps * self.slowdown)  # signal seconds per frame
        # the small tolerance keeps e.g. 30 s / 0.004 s from becoming 7499 frames
        self.nframes = int(np.ceil((self.endtime - self.starttime) / self.dt - 1e-9))

    @classmethod
    def from_signal(cls, signal, fps=25., slowdown=1., origin='start'):
        """A clock that covers a uts time series, the first frame at the centre of its
        first sample.

        Parameters
        ----------
        signal: uts time series
        fps, slowdown: see `Clock`
        origin: 'start' or float
            'start' (default): time is shown relative to the start of `signal`;
            a float: see `Clock`
        """
        if isinstance(origin, str):
            if origin != 'start':
                raise ValueError(f"origin should be 'start' or a float, not {origin!r}")
            origin = signal.starttime
        return cls(starttime=signal.starttime + 0.5 / signal.fs, endtime=signal.endtime,
                   fps=fps, slowdown=slowdown, origin=origin,
                   timerange=(signal.starttime, signal.endtime))

    def __repr__(self):
        return (f'Clock(starttime={self.starttime!r}, endtime={self.endtime!r}, fps={self.fps!r}, '
                f'slowdown={self.slowdown!r}, origin={self.origin!r}, timerange={self.timerange!r})')

    def __len__(self):
        return self.nframes

    def __iter__(self):
        for k in range(self.nframes):
            yield self.time(k)

    @property
    def duration(self):
        "duration of the video (s)"
        return self.nframes / self.fps

    def time(self, frame):
        "signal time of a frame number"
        if not 0 <= frame < self.nframes:
            raise IndexError(f'frame {frame} out of range (0-{self.nframes - 1})')
        return self.starttime + frame * self.dt

    def frame(self, time):
        "number of the frame nearest to a signal time"
        k = int(np.round((time - self.starttime) / self.dt))
        return min(max(k, 0), self.nframes - 1)

    def displaytime(self, time):
        "signal time as shown in the movie, i.e. relative to `origin`"
        return time - self.origin

    @property
    def displayrange(self):
        "`timerange` in display time"
        return self.displaytime(self.timerange[0]), self.displaytime(self.timerange[1])
