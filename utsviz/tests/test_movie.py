import shutil
import tempfile
import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg

from . import defs
from utsviz.movie import (Clock, ElectrodeImage, Movie, Oscillogram, Panel, SiteMarkers,
                          StateBar, TimeText, Trajectory, Viewer)


def signal(duration=0.2, fs=1000.):
    return defs.mcuts_signal(duration=duration, fs=fs, nx=2, ny=3)


def movie(s=None, fps=25., slowdown=40., **kwargs):
    s = signal() if s is None else s
    clock = Clock.from_signal(s, fps=fps, slowdown=slowdown)
    mv = Movie('IW\nCW\nTW', clock, size=(401, 301), dpi=50, **kwargs)
    mv.add('I', ElectrodeImage(s, vmin=-1, vmax=1, colorbar='C'))
    mv.add('W', Oscillogram(s, ylabels=True))
    mv.add('T', TimeText())
    return mv


class MovieTestCase(unittest.TestCase):

    def tearDown(self):
        plt.close('all')


class TestClock(unittest.TestCase):

    def test_framecount(self):
        c = Clock(0., 30., fps=50, slowdown=5)
        self.assertEqual(len(c), 7500)
        self.assertAlmostEqual(c.dt, 0.004)
        self.assertAlmostEqual(c.duration, 150.)

    def test_timeframe_roundtrip(self):
        c = Clock(1., 2., fps=25, slowdown=2)
        for k in (0, 7, len(c) - 1):
            self.assertEqual(c.frame(c.time(k)), k)

    def test_outofrange(self):
        c = Clock(0., 1., fps=10)
        self.assertRaises(IndexError, c.time, 10)
        self.assertEqual(c.frame(100.), 9)

    def test_fromsignal_onesampleperframe(self):
        s = signal(fs=1000.)
        c = Clock.from_signal(s, fps=25, slowdown=40)
        self.assertEqual(len(c), s.ntimesamples)
        np.testing.assert_allclose(list(c), s.samplingtimes())
        self.assertEqual(c.displayrange, (0., s.duration))

    def test_origin(self):
        c = Clock(10., 20., fps=10, origin=15.)
        self.assertEqual(c.displaytime(10.), -5.)
        self.assertEqual(c.displayrange, (-5., 5.))

    def test_badrange(self):
        self.assertRaises(ValueError, Clock, 2., 1.)


class TestMovie(MovieTestCase):

    def test_evensize(self):
        mv = movie()
        self.assertEqual(mv.size, (402, 302))
        canvas = FigureCanvasAgg(mv.fig)
        self.assertEqual(canvas.get_width_height(), (402, 302))

    def test_exactpixels(self):
        # independent of how the matplotlib version rounds: the figure is never a
        # fraction of a pixel smaller than requested (matplotlib < 3.8 truncates)
        from utsviz.movie.movie import _inches
        for dpi in (50, 72, 96, 100, 150, 300):
            for pixels in range(100, 2001, 2):
                self.assertGreaterEqual(_inches(pixels, dpi) * dpi, pixels)
                self.assertLess(_inches(pixels, dpi) * dpi, pixels + 1e-6)

    def test_needsclock(self):
        self.assertRaises(TypeError, Movie, 'A', None)

    def test_fastpath_equals_fulldraw(self):
        # the frames that are written (background + changed artists) should look
        # exactly like a complete redraw at the same time
        mv = movie()
        frames = {k: rgba.copy() for k, rgba in mv.iter_frames([0, 50, 199])}
        canvas = FigureCanvasAgg(mv.fig)
        mv.fig.set_layout_engine('none')  # positions as fixed by iter_frames
        for k, rgba in frames.items():
            mv.update(mv.clock.time(k))
            canvas.draw()
            np.testing.assert_array_equal(rgba, np.asarray(canvas.buffer_rgba()))

    def test_framesdiffer(self):
        mv = movie()
        frames = [rgba.copy() for _, rgba in mv.iter_frames([0, 100])]
        self.assertFalse(np.array_equal(*frames))

    def test_layoutfixed(self):
        mv = movie()
        positions = []
        for _ in mv.iter_frames([0, 199]):
            positions.append([ax.get_position().bounds for ax in mv.fig.axes])
        self.assertEqual(positions[0], positions[1])

    def test_canvasrestored(self):
        mv = movie()
        canvas = mv.fig.canvas
        list(mv.iter_frames([0]))
        self.assertIs(mv.fig.canvas, canvas)
        self.assertFalse(any(a.get_animated() for a in mv._animated_artists()))

    def test_frameno(self):
        mv = movie()
        mv.render(frame=3)
        self.assertEqual(mv.frameno, 3)
        mv.render(time=mv.clock.time(3) + 0.3 * mv.clock.dt)
        self.assertIsNone(mv.frameno)

    @unittest.skipIf(shutil.which(matplotlib.rcParams['animation.ffmpeg_path']) is None,
                     'ffmpeg not available')
    def test_save(self):
        mv = movie()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'test.mp4'
            mv.save(path, frames=range(10), progress=False)
            self.assertGreater(path.stat().st_size, 0)
            self.assertRaises(FileExistsError, mv.save, path, progress=False)


class TestElectrodeImage(MovieTestCase):

    def test_grid(self):
        s = signal()
        mv = movie(s)
        im = mv.panels[0]
        t = mv.clock.time(17)
        grid = im.grid(t)
        self.assertEqual(grid.shape, (3, 2))
        values = s.samples[:, 17]  # channelaxis is 0 in this signal
        for ch, value in zip(s.channelnames, values):
            row, col = s.geometry[ch]
            self.assertAlmostEqual(grid[row, col], value)
        np.testing.assert_array_equal(im.image.get_array(), np.ma.masked_invalid(grid))

    def test_channelxy(self):
        s = signal()
        im = movie(s).panels[0]
        ch = s.channelnames[3]  # row 1, col 1
        self.assertEqual(tuple(im.channelxy(ch)), (1, 1))
        self.assertEqual(im.channelxy(list(s.channelnames[:2])).shape, (2, 2))

    def test_colorbarlimits(self):
        im = movie().panels[0]
        np.testing.assert_allclose(im.cbar.get_ticks(), [-1, 1])


class TestOscillogram(MovieTestCase):

    def test_cursormoves(self):
        mv = movie()
        osc = mv.panels[1]
        mv.update(mv.clock.time(100))
        self.assertAlmostEqual(osc.cursorline.get_xdata()[0], mv.clock.displaytime(mv.clock.time(100)))

    def test_stacking(self):
        s = signal()
        clock = Clock.from_signal(s, fps=25, slowdown=40)
        mv = Movie('W', clock, size=(200, 200))
        osc = mv.add('W', Oscillogram(s, channels=s.channelnames[::-1], spacing=3.))
        np.testing.assert_allclose(osc.offsets, -3. * np.arange(s.nchannels))
        self.assertEqual(len(mv['W'].lines), s.nchannels + 1)  # traces + cursor

    def test_window(self):
        s = signal()
        clock = Clock.from_signal(s, fps=25, slowdown=40)
        mv = Movie('W', clock, size=(200, 200))
        mv.add('W', Oscillogram(s, window=0.05))
        t = clock.time(100)
        mv.update(t)
        np.testing.assert_allclose(mv['W'].get_xlim(), (t - 0.025, t + 0.025))

    def test_ylabels(self):
        s = signal()
        clock = Clock.from_signal(s, fps=25, slowdown=40)
        mv = Movie('W', clock, size=(200, 200))
        labels = [''] * s.nchannels
        labels[0] = 'top'
        mv.add('W', Oscillogram(s, ylabels=labels))
        self.assertEqual([t.get_text() for t in mv['W'].get_yticklabels()], ['top'])


class TestStateBar(MovieTestCase):

    def test_intervals(self):
        clock = Clock(0., 10., fps=10)
        mv = Movie('S', clock, size=(200, 100))
        sb = mv.add('S', StateBar({'a': [(0, 4)], 'b': [(4, 10)]},
                                  colors={'a': 'red', 'b': ('blue', 'green')}))
        self.assertEqual(len(mv['S'].patches), 3)  # b is two-tone
        self.assertEqual(sb.update(1.), ())

    def test_sequenceintervals(self):
        clock = Clock(0., 10., fps=10)
        mv = Movie('S', clock, size=(200, 100))
        mv.add('S', StateBar([(0, 4, 'a'), (4, 10, 'a')], colors={'a': 'red'}, cursor=True))
        self.assertEqual(len(mv['S'].patches), 2)


class TestTimeText(MovieTestCase):

    def test_text(self):
        clock = Clock(0., 10., fps=10, slowdown=2., origin=5.)
        mv = Movie('T', clock, size=(200, 100))
        tt = mv.add('T', TimeText('{t:.1f} s {slowdown:.0f}x frame {frame}'))
        mv.render(frame=20)
        self.assertEqual(tt.text.get_text(), '-4.0 s 2x frame 20')


class TestOverlays(MovieTestCase):

    def overlaymovie(self):
        s = signal()
        mv = movie(s)
        im = mv.panels[0]
        n = s.ntimesamples
        active = np.zeros((n, s.nchannels), dtype=bool)
        active[50:100, :2] = True  # first two channels active in samples 50-99
        xy = np.full((n, 2), np.nan)
        xy[50:100] = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 2, 50)])
        markers = mv.add('I', SiteMarkers(active, im.channelxy(list(s.channelnames)), fs=s.fs,
                                          starttime=s.starttime))
        traj = mv.add('I', Trajectory(xy, fs=s.fs, starttime=s.starttime, tail=0.01))
        return mv, markers, traj

    def test_markers(self):
        mv, markers, _ = self.overlaymovie()
        mv.update(mv.clock.time(60))
        alpha = markers.collection.get_edgecolor()[:, 3]
        np.testing.assert_array_equal(alpha > 0, [True, True, False, False, False, False])
        mv.update(mv.clock.time(10))
        self.assertFalse((markers.collection.get_edgecolor()[:, 3] > 0).any())

    def test_trajectory(self):
        mv, _, traj = self.overlaymovie()
        mv.update(mv.clock.time(70))  # tail 0.01 s = 10 samples
        self.assertEqual(len(traj.path.get_segments()), 10)
        alphas = traj.path.get_colors()[:, 3]
        self.assertTrue((np.diff(alphas) > 0).all())  # fades towards the past
        np.testing.assert_allclose(traj.head.get_xdata(), [traj.xy[70, 0]])
        mv.update(mv.clock.time(120))  # outside the path
        self.assertEqual(len(traj.path.get_segments()), 0)
        self.assertEqual(len(traj.head.get_xdata()), 0)

    def test_fastpath_with_overlays(self):
        mv, _, _ = self.overlaymovie()
        frames = {k: rgba.copy() for k, rgba in mv.iter_frames([10, 60, 99])}
        canvas = FigureCanvasAgg(mv.fig)
        mv.fig.set_layout_engine('none')
        for k, rgba in frames.items():
            mv.update(mv.clock.time(k))
            canvas.draw()
            np.testing.assert_array_equal(rgba, np.asarray(canvas.buffer_rgba()))

    def test_badshape(self):
        self.assertRaises(ValueError, SiteMarkers, np.zeros((10, 3), bool), np.zeros((2, 2)), fs=1.)


class TestCustomPanel(MovieTestCase):

    def test_subclass(self):
        class Dot(Panel):
            def draw(self, ax):
                self.dot, = ax.plot([0], [0], 'o')
                return (self.dot,)

            def update(self, time):
                self.dot.set_data([time], [time])
                return (self.dot,)

        clock = Clock(0., 1., fps=10)
        mv = Movie('A', clock, size=(100, 100))
        dot = mv.add('A', Dot())
        mv.render(frame=5)
        self.assertAlmostEqual(dot.dot.get_xdata()[0], 0.5)
        self.assertEqual(len(list(mv.iter_frames())), 10)


class TestViewer(MovieTestCase):

    def viewer(self, **kwargs):
        mv = movie()
        mv.add('C', StateBar([(0., 0.1, 'a')], colors={'a': 'red'}, cursor=True))
        return mv, Viewer(mv, **kwargs)

    def key(self, vw, key):
        KeyEvent('key_press_event', vw.fig.canvas, key)._process()

    def test_frameaxes_unchanged(self):
        mv = movie()
        mv.fig.draw_without_rendering()
        before = {k: ax.get_window_extent().bounds for k, ax in mv.axes.items()}
        vw = Viewer(mv, sliderheight=0.5)
        self.assertEqual(vw.fig.get_size_inches()[1] * mv.dpi, mv.size[1] + 0.5 * mv.dpi)
        for k, ax in mv.axes.items():
            x, y, w, h = ax.get_window_extent().bounds
            np.testing.assert_allclose((x, y - 0.5 * mv.dpi, w, h), before[k], atol=1e-6)

    def test_setframe(self):
        mv, vw = self.viewer()
        vw.setframe(30)
        self.assertEqual(mv.frameno, 30)
        self.assertEqual(vw.slider.val, 30)
        osc = mv.panels[1]
        self.assertAlmostEqual(osc.cursorline.get_xdata()[0], mv.clock.displaytime(mv.clock.time(30)))
        vw.setframe(10 ** 6)  # clipped
        self.assertEqual(vw.frame, len(mv.clock) - 1)
        vw.slider.set_val(5)
        self.assertEqual(mv.frameno, 5)

    def test_blit_equals_fulldraw(self):
        mv, vw = self.viewer()
        canvas = vw.fig.canvas
        canvas.draw()  # stores the background
        vw.setframe(40)
        blitted = np.asarray(canvas.buffer_rgba()).copy()
        canvas.draw()
        np.testing.assert_array_equal(blitted, np.asarray(canvas.buffer_rgba()))

    def test_keys(self):
        mv, vw = self.viewer()
        self.key(vw, 'right')
        self.assertEqual(vw.frame, 1)
        self.key(vw, 'up')
        self.assertEqual(vw.frame, 11)
        self.key(vw, 'end')
        self.assertEqual(vw.frame, len(mv.clock) - 1)
        self.key(vw, 'home')
        self.assertEqual(vw.frame, 0)
        calls = []
        vw.actions['n'] = calls.append
        self.key(vw, 'n')
        self.assertEqual(calls, [vw])

    def test_click_timeaxes(self):
        mv, vw = self.viewer()
        vw.fig.canvas.draw()
        for key in ('W', 'C'):  # oscillogram and state bar
            ax = mv[key]
            t = mv.clock.time(25)
            x, y = ax.transData.transform((mv.clock.displaytime(t), np.mean(ax.get_ylim())))
            MouseEvent('button_press_event', vw.fig.canvas, x, y, button=1)._process()
            MouseEvent('button_release_event', vw.fig.canvas, x, y, button=1)._process()
            self.assertEqual(vw.frame, 25)
            vw.setframe(0)
        x, y = mv['I'].transAxes.transform((0.5, 0.5))  # not a time axes
        MouseEvent('button_press_event', vw.fig.canvas, x, y, button=1)._process()
        self.assertEqual(vw.frame, 0)

    def test_play(self):
        mv, vw = self.viewer()
        vw.setframe(len(mv.clock) - 2)
        vw.play()
        vw._tick()
        self.assertEqual(vw.frame, len(mv.clock) - 1)
        vw._tick()  # at the end: stops
        self.assertFalse(vw.playing)
        vw.loop = True
        vw.play()  # restarts at the first frame
        self.assertEqual(vw.frame, 0)
        vw.pause()

    def test_noslider(self):
        mv, vw = self.viewer(slider=False)
        np.testing.assert_allclose(vw.fig.get_size_inches() * mv.dpi, mv.size)
        vw.setframe(3)
        self.assertEqual(mv.frameno, 3)


if __name__ == '__main__':
    unittest.main()
