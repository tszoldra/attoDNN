import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'xtick.direction': 'in',
                     'ytick.direction': 'in'})

plt.rcParams.update({
    "text.usetex": True,  # needs apt-get install cm-super
    "font.family": 'STIXGeneral',  # "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size": 16,
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{physics}'  # for \mathbb
})
