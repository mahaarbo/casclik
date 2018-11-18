import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
SPINE_COLOR = "gray"


def _to_percent(val, pos):
    """For changing the y ticker to percentage"""
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * val)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def percentage_yaxis(axes):
    axes.yaxis.set_major_formatter(FuncFormatter(_to_percent))


def percentage_xaxis(axes):
    axes.xaxis.set_major_formatter(FuncFormatter(_to_percent))


def latexify(fig_width=None, fig_height=None, columns=2):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """
    # Borrowed from https://nipunbatra.github.io/blog/2014/latexify.html
    # All credit goes to nipunbatra!
    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0     # Aesthetic ratio
        fig_height = fig_width*golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'pdf',
              'text.latex.preamble': ['\usepackage{gensymb}',
                                      '\usepackage{bm}',
                                      '\usepackage{amsmath}',
                                      '\usepackage{amsfonts}'],
              'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 8,
              'font.size': 8,  # was 10
              'legend.fontsize': 8,  # was 10
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'}

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)
    if not isinstance(ax, Axes3D):
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax
