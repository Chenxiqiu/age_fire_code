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
    lat_range = np.arange(-60, 60+10, 10) 
    output = df['OCS'].groupby([pd.cut(df['LAT'], lat_range)]).describe().reset_index() #apply(get_stats) .unstack()
    print(output.head())
    output['lat'] = output.apply(lambda x: x['LAT'].left+5/2,axis=1)
    return output  

# amica 
df = southtrac.read(strat=0, transfer=1, ALT=11)

def plot(label=None, df=None, ax=None, shift=0, **kwargs):
    if ax is None:
        ax = plt.gca()   
    ax.errorbar(y=df['mean'], 
                x=df['lat']+shift, 
                yerr=df['std'],
                capsize=4,
                label=label,
                fmt='s',
                #ecolor='#BDBDBD',
                **kwargs
                )
    return ax

months = [
    'September', 
    'October', 
    'November',
         ]

groups = [
    group(df[(df.index.month==9)]),
    group(df[(df.index.month==10)]),
    group(df[(df.index.month==11)]),
         ]

colors = [
    ('dodgerblue', 'turquoise'),
    ('darkorange', 'gold'),
    ('forestgreen', 'lawngreen'),
         ]
           
data = dict(zip(months, groups))
color = dict(zip(months, colors))

fig, ax = plt.subplots()

for i, (month, to_plot) in enumerate(data.items()):
    dot_color, cap_color = color[month]
    plot(month,
         to_plot,
         color=dot_color,
         ecolor=cap_color,
         shift=i-1,
         )

ax.set(
       xlabel ='latitude', 
       ylabel ='OCS / ppt', 
       xlim =(-60, 60), 
       ylim =(200, 700), 
       title='alt>11km'
       ) 

ax.grid()
plt.legend()   

    
# plot('local flights: phase 1',
#      df[(df.index.month==9) | (df.index.month==10)],
#      shift=0,
#      color='DarkBlue',
#      ecolor='LightBlue',
#      )

# plot('local flights: phase 2',
#      df[df.index.month==11],
#      shift=0.05,
#      color='darkmagenta',
#      ecolor='violet',
#      )

# plt.legend()

# ax=groups_sep.reset_index().plot(kind = "scatter", 
#                                  x='lat', y='mean',
#                                  yerr = "SD", capsize=10,#,capthick=1,
#                                  legend = 'SEP', 
#                                  title = "Average Avocado Prices",
#                                  ax=ax)

# scatter(x='lat', y='mean', color='DarkBlue', label='SEP',s=80)
# groups_oct.reset_index().plot.scatter(x='lat', y='mean', color='DarkGreen', label='OCT', s=80,
#                                       ax=ax)

# means = pd.concat([groups_sep['mean'], groups_oct['mean'],groups_sep['mean']-groups_oct['mean']], 
#                    axis='columns') 
# means.columns =['sep','oct','dif']

# means.plot.scatter()


# lat_res=5
# lat_range = np.arange(-90,90+lat_res,lat_res) 
# df_sep['AMICA_OCS'].groupby([pd.cut(df_sep.IRS_LAT, lat_range)]).plot.box()
