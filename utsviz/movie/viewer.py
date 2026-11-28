"""Interactive viewer of a Movie: the same figure and panels, in a window, with a slider."""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import key_press_handler
from matplotlib.widgets import Slider

__all__ = ['Viewer']


class Viewer:
    """Shows a `Movie` in a window, at any of its frames, chosen with a slider, the keyboard or the
    mouse. The panels are those of the movie, so what is shown is exactly the video frame (with a
    strip for the slider added below it).

    Controls
    --------
    - slider: drag to any frame
    - click or drag in a time axes (`Oscillogram`, `StateBar`): go to that time
    - right / left: next / previous frame; up / down: 10 frames; page up / page down: 100 frames
    - home / end: first / last frame
    - space: play / pause, at the frame rate of the movie (slower if drawing a frame takes longer)
    - other keys: matplotlib's defaults (e.g. 'q' closes the window), or your own `actions`

    The figure needs an interactive matplotlib backend (e.g. TkAgg or QtAgg). The movie's layout
    is fixed when the viewer is made; do not use the movie for writing a video while it is shown.

    Parameters
    ----------
    movie: Movie
    time: float or None
        signal time to show first; default the first frame
    slider: bool
        add a slider below the frame
    sliderheight: float
        height of the strip for the slider (inches)
    actions: dict or None
        key -> function(viewer), for keys of your own (e.g. to go to the next event); they take
        precedence over the built-in keys
    loop: bool
        when playing, continue at the first frame after the last one

    Examples
    --------
    >>> mv = Movie('IW\\nCW', clock, size=(1600, 800), dpi=150)
    >>> ...  # add panels, as for writing a video
    >>> Viewer(mv).show()

    """

    def __init__(self, movie, time=None, slider=True, sliderheight=0.35, actions=None, loop=False):
        self.movie = movie
        self.clock = movie.clock
        self.fig = fig = movie.fig
        self.actions = dict(actions or {})
        self.loop = loop
        self.frame = self.clock.frame(self.clock.starttime if time is None else time)
        self.background = None
        self.playing = False
        self._dragging = False
        movie.update(self.clock.time(self.frame))
        fig.draw_without_rendering()  # fix the layout, so that the axes do not move between frames
        fig.set_layout_engine('none')
        self.slider = self._addslider(sliderheight) if slider else None
        self.animated = movie._animated_artists()
        if self.slider is not None:
            self.animated.append(self.slider.ax)
        for a in self.animated:
            a.set_animated(True)
        self.timeaxes = {p.ax for p in movie.panels if getattr(p, 'timeaxis', False)}
        canvas = fig.canvas
        manager = canvas.manager
        if manager is not None and getattr(manager, 'key_press_handler_id', None) is not None:
            canvas.mpl_disconnect(manager.key_press_handler_id)  # our keys first, then the defaults
            manager.key_press_handler_id = None
        self._cids = [canvas.mpl_connect(name, func) for name, func in
                      [('draw_event', self._ondraw), ('key_press_event', self._onkey),
                       ('button_press_event', self._onpress),
                       ('motion_notify_event', self._onmotion),
                       ('button_release_event', self._onrelease)]]
        self.timer = canvas.new_timer(interval=1000. / self.clock.fps)
        self.timer.add_callback(self._tick)
        self._settext()

    def _addslider(self, height):
        "Adds a strip of `height` inches at the bottom of the figure, and a slider in it."
        fig = self.fig
        w, h = fig.get_size_inches()
        total = h + height
        for ax in fig.axes:
            if ax.get_axes_locator() is None:  # inset axes (colour bars) follow their parent
                x0, y0, aw, ah = ax.get_position().bounds
                ax.set_position([x0, (y0 * h + height) / total, aw, ah * h / total])
        for text in fig.texts:
            if text.get_transform() == fig.transFigure:
                x, y = text.get_position()
                text.set_position((x, (y * h + height) / total))
        fig.set_size_inches(w, total)
        ax = fig.add_axes([0.1, 0.3 * height / total, 0.65, 0.4 * height / total])
        slider = Slider(ax, 'frame', 0, len(self.clock) - 1, valinit=self.frame, valstep=1,
                        valfmt='%d', initcolor='none')
        slider.drawon = False  # the viewer draws it, with the panels
        slider.on_changed(lambda val: self.setframe(int(round(val))))
        return slider

    def _settext(self):
        if self.slider is not None:
            t = self.clock.displaytime(self.clock.time(self.frame))
            self.slider.valtext.set_text(f'{self.frame}/{len(self.clock) - 1}   {t:.3f} s')

    # drawing

    def _ondraw(self, event):
        "After a full draw (first show, resize, zoom): store the background, draw the panels on it."
        canvas = self.fig.canvas
        self.background = canvas.copy_from_bbox(self.fig.bbox) if canvas.supports_blit else None
        self._drawanimated()

    def _drawanimated(self):
        for a in self.animated:
            self.fig.draw_artist(a)

    def refresh(self):
        "Draws the panels at the current frame on screen."
        canvas = self.fig.canvas
        if self.background is None:  # not drawn yet, or no blitting
            canvas.draw_idle()
            return
        canvas.restore_region(self.background)
        self._drawanimated()
        canvas.blit(self.fig.bbox)
        canvas.flush_events()

    # navigation

    def setframe(self, frame):
        "Shows frame number `frame` (clipped to the frames of the movie)."
        frame = min(max(int(frame), 0), len(self.clock) - 1)
        if frame == self.frame and self.movie.time == self.clock.time(frame):
            return
        self.frame = frame
        self.movie.update(self.clock.time(frame))
        if self.slider is not None and self.slider.val != frame:
            self.slider.set_val(frame)  # calls setframe again, which returns at once
        self._settext()
        self.refresh()

    def settime(self, time):
        "Shows the frame nearest to signal time `time`."
        self.setframe(self.clock.frame(time))

    def step(self, n):
        self.setframe(self.frame + n)

    def play(self):
        if self.frame == len(self.clock) - 1:
            self.setframe(0)
        self.playing = True
        self.timer.start()

    def pause(self):
        self.playing = False
        self.timer.stop()

    def toggle(self):
        self.pause() if self.playing else self.play()

    def _tick(self):
        if self.frame < len(self.clock) - 1:
            self.step(1)
        elif self.loop:
            self.setframe(0)
        else:
            self.pause()

    # events

    _keysteps = {'right': 1, 'left': -1, 'up': 10, 'down': -10, 'pageup': 100, 'pagedown': -100}

    def _onkey(self, event):
        key = event.key
        if key in self.actions:
            self.actions[key](self)
        elif key in self._keysteps:
            self.step(self._keysteps[key])
        elif key == 'home':
            self.setframe(0)
        elif key == 'end':
            self.setframe(len(self.clock) - 1)
        elif key == ' ':
            self.toggle()
        else:
            key_press_handler(event, self.fig.canvas, getattr(self.fig.canvas, 'toolbar', None))

    def _navigating(self):
        "True when the toolbar's pan or zoom is on, so that clicks are for that."
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return toolbar is not None and bool(getattr(toolbar, 'mode', ''))

    def _timeat(self, event):
        if event.inaxes in self.timeaxes and event.xdata is not None:
            return event.xdata + self.clock.origin  # time axes show display time
        return None

    def _onpress(self, event):
        if event.button != 1 or self._navigating():
            return
        time = self._timeat(event)
        if time is not None:
            self._dragging = True
            self.settime(time)

    def _onmotion(self, event):
        if self._dragging:
            time = self._timeat(event)
            if time is not None:
                self.settime(time)

    def _onrelease(self, event):
        self._dragging = False

    def show(self, block=True):
        "Opens the window; with `block`, returns when it is closed."
        self.fig.canvas.draw_idle()
        plt.show(block=block)
        return self

    def close(self):
        self.pause()
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
        plt.close(self.fig)
