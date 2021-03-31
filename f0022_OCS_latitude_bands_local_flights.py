import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams, cycler
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import glob2
import xarray as xr
import pandas as pd
import itertools
import re
import clean.clean_03 as southtrac
from matplotlib import gridspec
from scipy import stats
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

def group(df):
    df = df[['ALT', 'LAT', 'OCS']]
    alt_range = np.arange(9, 14+0.5, 0.5) 
    lat_range = [-70, -60, -50, -40]
    output = df['OCS'].groupby([pd.cut(df['ALT'], alt_range), pd.cut(df['LAT'], lat_range)]).agg(['mean', 'std'])
    return output

df = southtrac.read(strat=1, local=1)
plot_data_p1 = group(df[(df.index.month==9) | (df.index.month==10)])
plot_data_p2 = group(df[(df.index.month==11)])
plot_data_dif = plot_data_p1['mean'] - plot_data_p2['mean']

index = [pd.Interval(-70, -60, closed='right'),
         pd.Interval(-60, -50, closed='right'),
         pd.Interval(-50, -40, closed='right'),]
tro_hs_v = [
    dict(sep = 10.4, nov = 9.1), 
    dict(sep = 9.8, nov = 9.5), 
    dict(sep = 10.1, nov = 10.6), 
    ]
trp_hs = dict(zip(index, tro_hs_v))

def plotting(label=None, df=None, ax=None, shift=0, **kwargs):
    if ax is None:
        ax = plt.gca()   
    ax.errorbar(x=df['mean'], 
                y=[(x.left+x.right)/2+shift for x in group['mean'].index.get_level_values('ALT')],
                xerr=df['std'],
                capsize=4,
                markeredgecolor='dimgrey',
                label=str(key),
                markersize=15,
                **kwargs
                )
    ax.grid()
    return ax

colors_v = ('dodgerblue', 'darkorange', 'forestgreen')
colors = dict(zip(index, colors_v))

fig = plt.figure(figsize=(10, 50))
font = {'size': 20}
plt.rc('font', **font)

spec = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 0.5])#,height_ratios=[15,1]) width_ratios=[9,1]

ax1 = fig.add_subplot(spec[0, 0])
grouped_p1 = plot_data_p1.groupby('LAT')
for i, (key, group) in enumerate(grouped_p1):
    plotting(
        df=group,
        ax=ax1,
        shift=(i-1)/50,
        color=colors[key],
        fmt='-o',
        )
    trp_h = trp_hs[key]['sep']
    ax1.plot([0, 1000], [trp_h]*2, color=colors[key], ls='--')
ax1.legend()
ax1.set(
   xlabel ='OCS / ppt', 
   ylabel ='altitude / km', 
   xlim =(150, 550), 
   ylim =(9, 14), 
   title='phase 1'
   ) 

ax2 = fig.add_subplot(spec[0, 1], sharex=ax1, sharey=ax1)
grouped_p2 = plot_data_p2.groupby('LAT')
for i, (key, group) in enumerate(grouped_p2):
    plotting(
        df=group,
        ax=ax2,
        shift=(i-1)/50,
        color=colors[key],
        fmt='-o'
        )
    trp_h = trp_hs[key]['nov']
    ax2.plot([0, 1000], [trp_h]*2, color=colors[key], ls='--')
ax2.set_xlabel('OCS / ppt')
ax2.set_title('phase 2')

ax3 = fig.add_subplot(spec[0, 2], sharey=ax1)
grouped_dif = plot_data_dif.groupby('LAT')
for i, (key, group) in enumerate(grouped_dif):
    ax3.plot(
        group.values,
        [(i.left+i.right)/2 for i in group.index.get_level_values('ALT')],
        color=colors[key],
        label=str(key),
        marker='s',
        markeredgecolor='dimgrey',
        markersize=15
        )
ax3.set_xlabel('difference in OCS / ppt')
ax3.set_xlim(-100, 100)
ax3.set_title('difference 1-2')
ax3.grid()   

plt.show()