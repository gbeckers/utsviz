"""Movies of uts signals.

A `Movie` is a matplotlib figure laid out as a video frame. Its axes hold
panels (`ElectrodeImage`, `Oscillogram`, `StateBar`, `TimeText`, overlays on an image
(`SiteMarkers`, `Trajectory`), or your own
subclass of `Panel`) that all show the same moment in time, which a `Clock`
determines for each frame. A `Viewer` shows a movie interactively, with a slider.

This replaces the older `utsviz.movies` module.
"""

from .clock import Clock
from .movie import Movie, codecs
from .panels import (ElectrodeImage, Oscillogram, Panel, SiteMarkers, StateBar, TimeText,
                     Trajectory)
from .viewer import Viewer

__all__ = ['Clock', 'Movie', 'codecs', 'Panel', 'ElectrodeImage', 'Oscillogram', 'StateBar',
           'TimeText', 'SiteMarkers', 'Trajectory', 'Viewer']
