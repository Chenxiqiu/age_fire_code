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

# def get_stats(group):
# #    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.quantile(),'SD': group.std()}
#     return {'count': group.count(), 
#             'mean': group.mean(), 
#             'median': group.quantile(),
#             'SD': group.std(),
#             'p10': group.quantile(0.1),
#             'p25': group.quantile(0.25),
#             'p75': group.quantile(0.75),
#             'p90': group.quantile(0.9)}


def group(df):
    df = df[['ALT', 'LAT', 'OCS']]
    alt_range = np.arange(9, 14+0.5, 0.5) 
    lat_range = [-70, -60, -50, -35]
    output = df['OCS'].groupby([pd.cut(df['ALT'], alt_range), pd.cut(df['LAT'], lat_range)]).describe() #apply(get_stats) .unstack()
    print(output.head())
    return output

df = southtrac.read(strat=1, local=1)
plot_data_p1 = group(df[(df.index.month==9) | (df.index.month==10)])
plot_data_p2 = group(df[(df.index.month==11)])
plot_data_dif = plot_data_p1['mean'] - plot_data_p2['mean']

def plotting(label=None, df=None, ax=None, shift=0, **kwargs):
    if ax is None:
        ax = plt.gca()   
    ax.errorbar(x=df['mean'], 
                y=[(x.left+x.right)/2+shift for x in group['mean'].index.get_level_values('ALT')],
                xerr=df['std'],
                capsize=4,
                ecolor='#BDBDBD',
                markeredgecolor='dimgrey',
                label=str(key),
                **kwargs
                )
    ax.grid()
    return ax

color_range = ('dodgerblue', 'forestgreen', 'darkorchid')
fmt = ('o', "v", "s")

fig = plt.figure(figsize=(10, 50))
spec = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[1, 1, 0.5])#,height_ratios=[15,1]) width_ratios=[9,1]

ax1 = fig.add_subplot(spec[0, 0])
grouped_p1 = plot_data_p1.groupby('LAT')
for i, (key, group) in enumerate(grouped_p1):
    plotting(
        df=group,
        ax=ax1,
        shift=(i-1)/50,
        color=color_range[i],
        fmt='-o'
        )
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
        color=color_range[i],
        fmt='-o'
        )
ax2.set_xlabel('OCS / ppt')
ax2.set_title('phase 2')

ax3 = fig.add_subplot(spec[0, 2], sharey=ax1)
grouped_dif = plot_data_dif.groupby('LAT')
for i, (key, group) in enumerate(grouped_dif):
    ax3.plot(
        group.values,
        [(i.left+i.right)/2 for i in group.index.get_level_values('ALT')],
        color=color_range[i],
        label=str(key),
        marker='s'
        )
ax3.set_xlabel('difference in OCS / ppt')
ax3.set_xlim(-100, 100)
ax3.set_title('difference 1-2')
ax3.grid()   

plt.show()