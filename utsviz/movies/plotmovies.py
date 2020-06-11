import numpy as np
import uts
from matplotlib import pyplot as plt
from matplotlib import animation
#from matplotlib.lines import Line2D

__all__ = ['plot_single']

# THIS SHOULD BE CLEANED UP
def plot_single(s, width, height, filename, ticks=False, fps=60.,
                frameduration=1.0, color='000000', lw=1, codec='libx264',
                crf=20, preset='slow', bitrate=5000, blit=True, dpi=None):
    ymin, ymax, sargmin, sargmax = uts.extremes(s)

    #st = s.samplingtimes()
    framesize, stepsize =  int(frameduration*np.round(s.fs)), int(
        np.round(s.fs / float(fps)))
    nframes, newsize, remainder = uts.core.utils.fit_frames(s.nt, framesize,
                                                            stepsize)
    # First set up the figure, the axis, and the plot element we want to animate
    #linewidth = 2.0

    fig, ax = plt.subplots(dpi=dpi)
    # fig.patch.set_alpha(0.0)
    fig.set_size_inches([width, height])

    line, = ax.plot([], [], lw=lw, color=color)
    ax.set_xlim(0, framesize)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,

    x = np.arange(framesize)

    def animate(i):
        line.set_data(x, s.samples[i * stepsize:i * stepsize + framesize])
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nframes, interval=1 / float(fps),
                                   blit=blit)
    extra_args = ['-pix_fmt', 'yuv420p']# '-crf', str(crf), '-preset', preset]
    mywriter = animation.FFMpegWriter(fps=fps, codec=codec, bitrate=bitrate,
                                      extra_args=extra_args)
    savefig_kwargs = {'facecolor': 'white', 'edgecolor': 'white',
                      'alpha':1.0}

    anim.save(filename, writer=mywriter, savefig_kwargs=savefig_kwargs)
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                            frames=nframes, interval=1/float(fps), blit=True)
    # anim.save(filename, fps=fps, codec=codec)
    plt.show()