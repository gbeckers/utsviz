
import matplotlib
#matplotlib.rcParams['animation.frame_format'] = 'jpeg'

import numpy as np
import uts
from sys import stdout
from matplotlib import cm
import os
from pathlib import Path
import subprocess
import matplotlib.animation as animation
from matplotlib.pylab import plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.lines as lines
from scipy.signal import spectrogram



__all__ = ['codecs', 'FramesTimeAligned', 'interpolations', 'HPanelMovie',
           'TimeIndicatedLetterAxes', 'LetterAxes', 'SpecgramAxes',
           'TimeIndicatedSpecgramAxes']

interpolations = ("nearest", "bilinear", "bicubic", "spline16",
                  "spline36", "hanning", "hamming", "hermite",
                  "kaiser", "quadric", "catrom", "gaussian",
                  "bessel", "mitchell", "sinc", "lanczos")

codecs = {'png': (('-an', '-vcodec', 'png'), '%06d.png'),
          'h264_yuv444': (('-vcodec', 'libx264', '-crf', '5'), '.mp4'),
          'h264_yuv422': (
          ('-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv422p' ), '.mp4'),
          'h264_yuv420': (
          ('-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p'), '.mp4'),
          'h265_yuv444': (
          ('-vcodec', 'libx265', '-crf', '5', '-pix_fmt', 'yuv444p'), '.mp4'),
          'h265_yuv422': (
          ('-vcodec', 'libx265', '-crf', '5', '-pix_fmt', 'yuv422p'), '.mp4'),
          'h265_yuv420': (
          ('-vcodec', 'libx265', '-crf', '5', '-pix_fmt', 'yuv420p'), '.mp4'),
          'avi_effectivelylossless1': (
          ('-vcodec', 'huffyuv', '-pix_fmt', 'rgb24'), '.avi'),
          'avi_effectivelylossless2': (
          ('-vcodec', 'huffyuv', '-pix_fmt', 'yuv422p'), '.avi'),
              'avi_perceptuallylossless': (
          ('-vcodec', 'mjpeg', '-qscale:v', '1'), '.avi')}



# FIXME rename clips to scaling points oid
# FIXME clips should be calculated if None
class FramesTimeAligned(uts.UniformTime):
    def __init__(self, mcutss, fps=25, slowdown=10., chunksize=None,
                 show_progress=None, hflips=None,
                 colormaps=None, clips=None, units=None, scales=None):

        """
        Generates

        :param mcutss:
        :param fps: number of frames per second that are to be produced
        :param slowdown:
        :param chunksize:
        :param show_progress:
        :param hflips:
        :param colormaps:
        :param clips:
        :param units:
        :param scales:
        """
        self.mcutss = self.check_mcutss(mcutss)
        self.npannels = len(self.mcutss)
        self.fps = float(fps)
        self.requestedslowdown = slowdown
        self.decimate = int(np.round(mcutss[0].fs / (slowdown * fps)))
        ntimesamples = mcutss[0].ntimesamples // self.decimate
        self.realslowdown = mcutss[0].fs / float(self.decimate * fps)
        starttime = mcutss[0].timerange[0]
        origintime = mcutss[0].origintime
        startdatetime = mcutss[0].startdatetime

        uts.UniformTime.__init__(self, ntimesamples=ntimesamples,
                                 fs=float(fps*self.realslowdown),
                                 starttime=starttime, origintime=origintime,
                                 startdatetime=startdatetime)

        self.show_progress = show_progress
        if chunksize is None:
            chunksize = int(np.round(mcutss[0].fs * 1.0)) * self.decimate
        self.chunksize = chunksize
        self.chunkindices = [ind for ind in
                             self.mcutss[0].movingtimeframeindices(
                                 framesize=self.chunksize,
                                 include_remainder=True)]
        self.hflips = self.check_hflips(hflips, self.mcutss)
        self.colormaps = self.check_colormaps(colormaps, self.mcutss)
        self.clips = self.check_clips(clips, self.mcutss)
        if units is None:
            self.units = ['' for i in range(self.npannels)]
        else:
            self.units = units
        self.scales = scales


    def check_mcutss(self, mcutss):
        for i in range(1, len(mcutss)):
            if not uts.istimealigned(mcutss[0], mcutss[i]):
                raise ValueError('not all signals are time-aligned')
        return mcutss

    def check_hflips(self, hflips, mcutss):
        if hflips is None:
            hflips = [False] * len(mcutss)
        else:
            for hflip, mcuts in zip(hflips, mcutss):
                if not isinstance(hflip, bool):
                    raise TypeError(
                        'hflip parameter is not a bool but %s' % type(hflip))
        return hflips

    def check_colormaps(self, colormaps, mcutss):
        if colormaps is None:
            return [cm.jet for mcuts in mcutss]
        else:
            return [cmap if hasattr(cmap, '__call__') else cm.get_cmap(cmap)
                    for cmap in colormaps]
    # TODO choose something reasonable if None
    def check_clips(self, clips, mcutss):
        if clips is None:
            return [None for mcuts in mcutss]
        else:
            return clips

    def generate_frames(self):
        ars = []
        gmapindices = []
        if self.show_progress:
            print('Generating {} frames: '.format(self.ntimesamples), end='')
        for i, mcuts in enumerate(self.mcutss):
            gmapindices.append(
                mcuts.geometry.indexfrom1Dto2D(mcuts.channelnames,
                                               hflip=self.hflips[i]))
            x, y = mcuts.geometry.shape
            ars.append(np.empty((y, x, 4), dtype=np.float64))
        k = -1
        for i, (chunkstart, chunkend) in enumerate(self.chunkindices):
            chunks = []
            for mcuts in self.mcutss:
                chunks.append(
                    uts.decimate(mcuts[chunkstart:chunkend], self.decimate))
            nchunkframes = chunks[0].ntimesamples
            for j in range(nchunkframes):
                time = (chunks[0][j].timerange[0] + chunks[0][j].timerange[
                    1]) / 2.0
                for chunkn, chunk in enumerate(chunks):

                    samples = np.squeeze(chunk[j].samples.get(copy=False))
                    if not (self.scales is None):
                        samples = samples * self.scales[chunkn]
                    if not self.clips[chunkn] is None:
                        samples -= self.clips[chunkn][0]
                        samples /= (
                            self.clips[chunkn][1] - self.clips[chunkn][0])
                    ars[chunkn][gmapindices[chunkn]] = self.colormaps[chunkn](
                        samples)
                k += 1
                if self.show_progress and (((k % self.show_progress) == 0) or (
                            k == (self.ntimesamples - 1))):
                    print('<%d>' % k, end='')
                    stdout.flush()
                # this yield the frame array, the total number of frames,
                # the frame number, and the mean time of the frame
                yield ars, k, time


class HPanelMovie(object):
    def __init__(self, fta, staticimlabels=None, dynamicimlabels=None,
                 pannelwidth=2.5, pannelheight=2.5, interpolation='nearest',
                 hmargin=0.15, vmargin=0.0, hspace=0.15,
                 timebarheight=0.4, legendbarheight=0.8,
                 staticimlabelbarheight=0.4, dynamicimlabelbarheight=0.4,
                 labelbarfontsize=12, timebarfontsize=10, dpi=100,
                 facecolor='white', colorbarshrink=0.8, colorbarlabels=None,
                 include_frameno=False, include_dt=False, sound=None,
                 topaxesheight=0.0, multitopaxes=False,
                 staticimlabelscolor='black', timecolor='black',
                 showdatetime=False, savefig_kwargs=None):

        self.fta = fta
        self.npannels = fta.npannels
        self.framegenerator = None
        self.dt = 1000 * fta.decimate / fta.fs
        self.staticimlabels = staticimlabels
        self.dynamicimlabels = dynamicimlabels
        if staticimlabels is None:
            staticimlabelbarheight = 0.0
        if dynamicimlabels is None:
            dynamicimlabelbarheight = 0.0
        self.staticimlabelbarheight = staticimlabelbarheight
        self.dynamicimlabelbarheight = dynamicimlabelbarheight
        self.colorbarlabels = colorbarlabels
        if colorbarlabels is None:
            self.colorbarlabels = ['' for i in range(self.npannels)]
        self.pannelwidth = pannelwidth
        self.pannelheight = pannelheight
        self.timebarheight = timebarheight
        self.legendbarheight = legendbarheight
        self.hmargin = hmargin
        self.vmargin = vmargin
        self.hspace = hspace
        self.dpi = dpi
        self.labelbarfontsize = labelbarfontsize
        self.timebarfontsize = timebarfontsize
        self.colorbarshrink = colorbarshrink
        self.include_frameno = include_frameno
        self.include_dt = include_dt
        self.showdatetime = showdatetime
        self.sound = sound

        self.topaxess = []
        self.topaxesheight = topaxesheight
        self.multitopaxes = multitopaxes
        self.topaxesupdatefuncs = []
        self.topaxesinitfuncs = []

        self.dynamicimlabeltext = []
        self.staticimlabelscolor = staticimlabelscolor
        self.facecolor = facecolor
        self.timecolor = timecolor
        self.savepng = None
        self.fig = None
        self.axes = None
        self.ims = None
        self.pngfullpath = None
        self.savefig_kwargs = {'facecolor': self.facecolor,
                               'edgecolor': 'none'}
        if savefig_kwargs is not None:
            self.savefig_kwargs.update(savefig_kwargs)
        self.npannels = fta.npannels
        self.interpolation = interpolation
        self.setup_fig()

    def calc_sizes(self):

        figheight = self.vmargin * 2 \
                    + self.topaxesheight \
                    + self.staticimlabelbarheight \
                    + self.dynamicimlabelbarheight \
                    + self.pannelheight \
                    + self.legendbarheight \
                    + self.timebarheight
        figwidth = self.hmargin * 2 \
                   + self.pannelwidth * self.npannels \
                   + self.hspace * (self.npannels - 1 )
        # the next is necessary because we want even number of pixels in
        # both width and height; some compress ion codecs won't work otherwise
        width_even_pixels = uts.round_nearesteven(figwidth * self.dpi)
        height_even_pixels = uts.round_nearesteven(figheight * self.dpi)
        figwidth = width_even_pixels / float(self.dpi)
        figheight = height_even_pixels / float(self.dpi)

        # calc coordinates
        self.pannelheight_c = self.pannelheight / figheight
        self.vmargin_c = self.vmargin / figheight
        self.topaxesheight_c = self.topaxesheight / figheight
        self.staticimlabelbarheight_c = self.staticimlabelbarheight / figheight
        self.dynamicimlabelbarheight_c = self.dynamicimlabelbarheight / figheight
        self.legendbarheight_c = self.legendbarheight / figheight
        self.timebarheight_c = self.timebarheight / figheight

        self.pannelwidth_c = self.pannelwidth / figwidth
        self.hmargin_c = self.hmargin / figwidth
        self.hspace_c = self.hspace / figwidth

        self.figwidth = figwidth
        self.figheight = figheight

        #print(self.pannelheight_c, self.vmargin_c, self.topaxesheight_c,
        #      self.staticimlabelbarheight_c, self.dynamicimlabelbarheight_c,
        #      self.legendbarheight_c, self.timebarheight_c,
        #      self.pannelwidth_c, self.hmargin_c, self.hspace_c)

    def setup_fig(self):

        def clear_axes(ax):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_frame_on(False)

        self.calc_sizes()
        self.fig = fig = plt.figure(figsize=(self.figwidth, self.figheight),
                                    facecolor='white', edgecolor=None,
                                    frameon=True)
        self.gs = gs = gridspec.GridSpec(4, self.npannels, wspace=0.0,
                                         hspace=0.00)

        axes = []
        ims = []
        cbs = []
        hpos = self.hmargin_c

        self.staticbar = []
        for i in range(self.npannels):
            # panel labels
            x = hpos + self.pannelwidth_c / 2.0
            y = 1 - self.vmargin_c - self.topaxesheight_c - self.staticimlabelbarheight_c / 2.0
            if not self.staticimlabels is None:
                t = self.fig.text(x, y, self.staticimlabels[i],
                              horizontalalignment='center',
                              verticalalignment='center', color=self.staticimlabelscolor,
                              fontsize=self.labelbarfontsize)
                self.staticbar.append(t)
            y = 1 - self.vmargin_c - self.topaxesheight_c - self.staticimlabelbarheight_c \
                - self.dynamicimlabelbarheight_c / 2.0
            if not self.dynamicimlabels is None:
                txt = fig.text(x, y, 'dynamic label',
                               horizontalalignment='center',
                               verticalalignment='center', color='black',
                               fontsize=self.labelbarfontsize)
                # txt.set_animated(True)
                self.dynamicimlabeltext.append(txt)

            y = 1 - self.vmargin_c - self.topaxesheight_c - self.staticimlabelbarheight_c \
                - self.dynamicimlabelbarheight_c - self.pannelheight_c
            # image plots, own axes
            ax_im = fig.add_axes([hpos, y, self.pannelwidth_c,
                                  self.pannelheight_c])  # [left, bottom, width, height] values in 0-1 relative figure coordinates
            # print hpos, y, pannelwidth_c, pannelheight_c
            clear_axes(ax_im)
            #print 'panels', ax_im.get_position()
            ims.append(ax_im.imshow([[0, 1], [1, 0]],
                                    interpolation=self.interpolation,
                                    cmap=self.fta.colormaps[i], aspect='auto'))

            # color bars, own axes
            y = self.vmargin_c + self.timebarheight_c
            ax_cb = fig.add_axes(
                [hpos, y, self.pannelwidth_c, self.legendbarheight_c])
            clear_axes(ax_cb)
            #print 'cb', ax_cb.get_position()
            cb = fig.colorbar(ims[-1], ax=ax_cb, orientation='horizontal',
                              fraction=1, ticks=(0., 1), aspect=30,
                              shrink=self.colorbarshrink, anchor=(0.5, 0.5))
            cb.set_label(self.colorbarlabels[i], labelpad=-10,
                         color=self.staticimlabelscolor)

            cb.ax.set_xticklabels(
                ['{} {}'.format(v, self.fta.units[i]) for v in
                 self.fta.clips[i]])
            cbs.append(cb)
            # topaxis
            if self.topaxesheight != 0.0:
                y = 1 - self.vmargin_c - self.topaxesheight_c
                if self.multitopaxes:
                    ax = self.fig.add_axes(
                        [hpos, y, self.pannelwidth_c, self.topaxesheight_c])
                    ax.set_axis_off()
                    self.topaxess.append(ax)
                    self.topaxesupdatefuncs.append(None)
                    self.topaxesinitfuncs.append(None)
                elif i == 0:
                    width = 1 - self.hmargin_c
                    ax = self.fig.add_axes(
                        [self.hmargin_c, y, width, self.topaxesheight_c])
                    ax.set_axis_off()
                    self.topaxess.append(ax)
                    self.topaxesupdatefuncs.append(None)
                    self.topaxesinitfuncs.append(None)
            hpos += self.pannelwidth_c + self.hspace_c


        y = self.vmargin_c + self.timebarheight_c / 2.0
        #print(y)
        self.timebar = self.fig.text(self.hmargin_c, y, "Time bar",
                                     horizontalalignment='left',
                                     verticalalignment='center', color=self.timecolor,
                                     fontsize=self.timebarfontsize)
        #self.timebar.set_animated(True)

        self.ims = ims
        self.cbs = cbs

    def update_img(self, n):

        ars, frameno, time = next(self.framegenerator)
        changed = []
        if not np.isnat(self.fta.startdatetime) and self.showdatetime==True:
            textline = f'{self.fta.time_to_datetime(time)}'[:-6]
        else:
            textline = f'{time:.3f} s.'
        if self.include_frameno:
            textline = '{}    Frame: {:d}'.format(textline, frameno)
        if self.include_dt:
            textline = '{}    dt: {:.3f} ms'.format(textline,
                                                    1000 * self.fta.decimate / float(
                                                        self.fta.mcutss[0].fs))
        if self.fta.realslowdown != 1.0:
            textline = '{}    Slowdown: {:.2f}x'.format(textline,
                                                        self.fta.realslowdown)
        self.timebar.set_text(textline)
        changed.append(self.timebar)
        if not self.dynamicimlabels is None:
            for i, dl in enumerate(self.dynamicimlabels):
                labelrow = dl[
                    ((dl['starttime'] <= time) & (dl['endtime'] > time))]
                if len(labelrow) == 0:
                    self.dynamicimlabeltext[i].set_text('')
                    self.dynamicimlabeltext[i].set_color('#ffffff')
                elif len(labelrow) == 1:
                    fields = set(labelrow.dtype.fields) - set(['starttime', 'endtime'])
                    for field in fields:
                        self.dynamicimlabeltext[i].set(
                            **{field: labelrow[0][field]})
                elif len(labelrow) > 1:
                    raise ValueError(
                        'More than one label at time {} s.'.format(time))
                changed.append(self.dynamicimlabeltext[i])
                # print self.dynamicimlabeltext[i].get_animated(), self.dynamicimlabeltext[i].get_text(),\
            #                       self.dynamicimlabeltext[i].get_position(), self.dynamicimlabeltext[i].get_color()
        for i, (im, ar) in enumerate(zip(self.ims, ars)):
            im.set_data(ar)
            changed.append(im)
        for topaxesupdatefunc in self.topaxesupdatefuncs:
            if not topaxesupdatefunc is None:
                #topaxesupdatefunc(time, frameno)
                for change in topaxesupdatefunc(time, frameno):
                    changed.append(change)
        self.fig.canvas.blit(self.fig.bbox)
        if self.savepng:
            self.fig.savefig(
                os.path.join(self.pngfullpath, 'frame%.3d.png' % (n)),
                dpi=self.dpi, **self.savefig_kwargs)

        return tuple(changed)

    # FIXME update path to python3
    def add_sound(self, filename, path='.', keepwav=False,
                  keeporiginal=False, ffmpegbin='ffmpeg',
                  vextraargs=('-vcodec', 'libx265', '-crf', '5'),
                  soxbin='sox'):

        slowdown = self.fta.realslowdown
        with uts.io.utils.tempdir(path, keep=keepwav) as tempwavdir:
            wavpath = os.path.join(tempwavdir, 'even.wav')
            uts.io.audio.save_wav(self.sound, wavpath)
            if slowdown != 1.0:
                args = [soxbin, wavpath,
                        os.path.join(tempwavdir, 'even_slow.wav'),
                        "speed", "%f" % (1 / slowdown), "dither", "-s"]
                subprocess.call(args)
                wavpath = os.path.abspath(
                    os.path.join(tempwavdir, 'even_slow.wav'))
            moviewithsoundname = os.path.splitext(filename)[0] + '_snd.mp4'
            args = [ffmpegbin, '-y', '-i', filename, '-i', wavpath]
            args.extend(vextraargs)
            args.extend(['-strict',  '-2'])
            args.append(moviewithsoundname)
            subprocess.check_call(args)
        if not keeporiginal:
            Path(filename).unlink()


    def create(self, basefilename, path='.',
               codec=None,
               vextraargs=None,
               blit=False,
               overwrite=False, savepng=False):
        """default codec: h264_yuv420"""

        def initanim():
            changed = []
            for im in self.ims:
                im.set_data([[], []])
                changed.append(im)
            for dl in self.dynamicimlabeltext:
                dl.set_text('')
                changed.append(dl)
            self.timebar.set_text('Timebar')
            changed.append(self.timebar)
            for topaxesinitfunc in self.topaxesinitfuncs:
                if not topaxesinitfunc is None:
                    #topaxesinitfunc()
                    for change in topaxesinitfunc():
                        changed.append(change)
            return tuple(changed)

        self.framegenerator =self.fta.generate_frames()
        self.savepng = savepng
        if codec is None:
            codec = 'h264_yuv420'
        vargs, extension = codecs[codec]
        if vextraargs is None:
            vextraargs = vargs
        filename = basefilename + extension
        fullpath = os.path.join(path, filename)
        if os.path.exists(fullpath):
            if overwrite:
                os.remove(fullpath)
            else:
                raise IOError(
                    "File '{}' exists, use overwrite parameter to overwrite".format(
                        fullpath))
        if self.savepng:
            self.pngfullpath = os.path.join(path, '%s_pngs' % basefilename)
            if not os.path.exists(self.pngfullpath):
                os.makedirs(self.pngfullpath)
        nframes = int(self.fta.ntimesamples)
        ani = animation.FuncAnimation(self.fig, self.update_img, nframes,
                                      init_func=initanim,
                                      interval=self.fta.fps,
                                      blit=blit)

        writer = animation.writers['ffmpeg'](fps=self.fta.fps, bitrate=-1,
                                             extra_args=vextraargs)
        ani.save(fullpath, writer=writer, dpi=self.dpi,
                 savefig_kwargs=self.savefig_kwargs)
        plt.close(self.fig)
        if not (self.sound is None):
            self.add_sound(filename=filename, path=path,
                           keeporiginal=False)


class LetterAxes(object):
    def __init__(self, letters=('AAAAAAA', 'BBBBBBB'), fontsize=11,
                 fontcolor='black',
                 bgcolor='white', axes=None):
        self.letters = letters
        self._calc_positions()
        self.fontsize = fontsize
        self.bgcolor = bgcolor
        self.fontcolor = fontcolor
        self.lettertexts = None
        if not axes is None:
            self.axes = axes
        else:
            self.axes = plt.gca()
        self.axes.set_axis_off()
        self._draw_letters()

    def _calc_positions(self):
        self.nlines = len(self.letters)
        self.lineheight_c = 1 / float(self.nlines)
        self.maxnletters = max([len(line) for line in self.letters])
        self.letterwidth_c = 1 / float(self.maxnletters)

    def _draw_letters(self):

        self.axes.add_patch(
            patches.Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=self.bgcolor,
                              edgecolor=self.bgcolor))
        lettertexts = []
        for lineno, lineletters in enumerate(self.letters[::-1]):
            lettertexts.append([])
            for letterno, letter in enumerate(lineletters):
                x = letterno * self.letterwidth_c + self.letterwidth_c * 0.5
                y = lineno * self.lineheight_c + self.lineheight_c * 0.5
                letter = self.axes.text(x, y, letter,
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        color='black',
                                        fontsize=self.fontsize)
                lettertexts[-1].append(letter)
        self.lettertexts = lettertexts

class TimeIndicatedLetterAxes(LetterAxes):

    def __init__(self, starttime=0., endtime=10.,
                 timeindicators=(5.0,), letters=('AAAAAAA', 'BBBBBBB'),
                 fontsize=11, fontcolor='black', bgcolor='white', linewidth=1,
                 axes=None):

        self.starttime = starttime
        self.endtime = endtime
        self.duration = float(endtime - starttime)
        self.timeindicators = timeindicators
        self.linewidth = linewidth
        super(TimeIndicatedLetterAxes, self).__init__(letters=letters,
                                                      fontsize=fontsize,
                                                      bgcolor=bgcolor,
                                                      fontcolor=fontcolor,
                                                      axes=axes)
        self._draw_timeindicators()

    def _time_to_x(self, time):
        pos = (time - self.starttime) / self.duration
        #print pos
        return pos

    def _draw_timeindicators(self):
        timeindicatorlines = []
        for timeindicator in self.timeindicators:
            x = self._time_to_x(timeindicator)
            l = self.axes.add_line(
                lines.Line2D((x, x), (0., 1.), linewidth=self.linewidth,
                             color='blue'))
            timeindicatorlines.append(l)
        self.timeindicatorlines = timeindicatorlines

    def update(self, time, frameno):
        for til in self.timeindicatorlines:
            x = self._time_to_x(time)
            til.set_data((x, x), (0., 1.))
        return tuple(self.timeindicatorlines)


    def init(self):
        for til, ti in zip(self.timeindicatorlines, self.timeindicators):
            x = self._time_to_x(ti)
            til.set_data((x, x), (0., 1))
        return tuple(self.timeindicatorlines)


class SpecgramAxes:

    def  __init__(self, sound, starttime=None, endtime=None, axes=None,
                  bgcolor='white', nperseg=None,
                  noverlap=None, nfft=None, interpolation='nearest',
                  cmap='gray_r', clim=None, maxfreq=8000.):
        self.sound = sound
        if starttime is not None:
            self.starttime = starttime
        else:
            self.starttime = sound.starttime
        if endtime is not None:
            self.endtime = endtime
        else:
            self.endtime = sound.endtime
        if not axes is None:
            self.axes = axes
        else:
            self.axes = plt.gca()
        self.bgcolor = bgcolor
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.interpolation = interpolation
        self.cmap = cmap
        self.clim = clim
        self.maxfreq = maxfreq
        self.f, self.t, self.specgram = self._calc_specgram()
        self._draw_specgram()

    def _calc_specgram(self):
        samples = self.sound.samples[:]
        fs = self.sound.fs
        f,t,Sxx =  spectrogram(samples, fs=fs, window=('tukey', 0.25),
                               nperseg=self.nperseg, noverlap=self.noverlap,
                               nfft=self.nfft, detrend='constant',
                               return_onesided=True, scaling='density',
                               axis=-1, mode='psd')
        eps = np.finfo(Sxx.dtype).eps
        return f, t, 10*np.log10(Sxx+eps)


    def _draw_specgram(self):
        if self.clim is None:
            max = self.specgram.max()
            clim = (max-40, max)
        else:
            clim = self.clim

        im = self.axes.imshow(self.specgram, interpolation=self.interpolation,
                         cmap=self.cmap, aspect='auto', clim=clim,
                         extent=[self.starttime, self.endtime,
                                 self.f[0], self.f[-1]],
                         origin='lower', zorder=0)
        self.axes.set_ylim((0, self.maxfreq))

class TimeIndicatedSpecgramAxes(SpecgramAxes):

    def __init__(self, sound, starttime=None, endtime=None,
                 axes=None, bgcolor='white', nperseg=None,
                 noverlap=None, nfft=None, interpolation='nearest',
                 cmap='gray_r', clim=None, maxfreq=8000., linewidth=1,
                 indicatorcolor='#0094ff'):

        super(TimeIndicatedSpecgramAxes, self).__init__(sound=sound,
                                                        starttime=starttime,
                                                        endtime=endtime,
                                                        axes=axes,
                                                        bgcolor=bgcolor,
                                                        nperseg=nperseg,
                                                        noverlap=noverlap,
                                                        nfft=nfft,
                                                        interpolation=interpolation,
                                                        cmap=cmap,
                                                        clim=clim,
                                                        maxfreq=maxfreq)

        self.duration = float(self.endtime - self.starttime)
        self.linewidth = linewidth
        self.indicatorcolor = indicatorcolor
        self.indicatorline = None
        self._draw_timeindicator()

    # def _time_to_x(self, time):
    #     pos = (time - self.starttime) / self.duration
    #     return pos

    def _draw_timeindicator(self):
        l = self.axes.add_line(
            lines.Line2D((self.starttime, self.starttime),
                         (self.starttime, self.maxfreq),
                         linewidth=self.linewidth,
                         color=self.indicatorcolor))
        self.timeindicatorline = l

    def update(self, time, frameno):
        # x = self._time_to_x(time)
        self.timeindicatorline.set_data((time, time), (0., self.maxfreq))
        return (self.timeindicatorline, )

    def init(self):
        # x = self._time_to_x(self.starttime)
        self.timeindicatorline.set_data((self.starttime, self.starttime),
                                        (0., self.maxfreq))
        return (self.timeindicatorline, )


