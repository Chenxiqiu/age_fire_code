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

groupby = 'LAT'
vmin = -90#-90
vmax = 90#90
res = 5#5

def group(df, group='OCS', groupby=groupby, vmin=vmin, vmax=vmax, res=res):
    vrange = np.arange(vmin, vmax+res, res) 
    output = df[group].groupby([pd.cut(df[groupby], vrange)]).describe().reset_index() #apply(get_stats) .unstack()
    output[groupby.casefold()] = output.apply(lambda x: x[groupby].left+res/2,axis=1)
    return output  

# amica 
df = southtrac.read(strat=0,LAT=[-90, 90], ALT=9)
 
df_9oct = df[(df['flight']=='2019-10-09') | (df['flight']=='2019-10-09')] #SAL - OPH super high
ocs_9oct = group(df_9oct)

df_2nov = df[df['flight']=='2019-11-02'] #OPH - SAL super low
ocs_2nov = group(df_2nov)


def plot(df=None, ax=None, shift=0, **kwargs):
    if ax is None:
        ax = plt.gca()   
    ax.errorbar(df[groupby.casefold()]+shift, 
                df['mean'], 
                yerr=df['std'],
                capsize=4,
                **kwargs
                )
    return ax

##############################################################################
fig, ax = plt.subplots()
group_name='FAIRO_O3'
# 9oct
p1 = plot(df=group(df_9oct, group=group_name), 
          ax=ax,
          shift=0.5,
          fmt='x',
          color='DarkBlue', 
          ecolor='LightBlue', 
          label=';;;')
#2nov
p2 = plot(df=group(df_2nov, group=group_name), 
          shift=0.5,
          ax=ax,
          fmt='x',
          color='darkmagenta', 
          ecolor='violet', 
          label=f'2nov {group_name}')

ax2 = ax.twinx()
p3 = plot(df=ocs_9oct, 
          ax=ax2,
          fmt='o',
          color='DarkBlue', 
          ecolor='LightBlue', 
          label='9oct OCS')
#2nov
p4 = plot(df=ocs_2nov, 
          ax=ax2,
          fmt='o',
          color='darkmagenta', 
          ecolor='violet', 
          label='2nov OCS')

fig.legend(loc='upper center', bbox_to_anchor=(0.5,1),ncol=4,
           columnspacing=0.5, frameon=True)

# # added these three lines
# lines = [p1, p2, p3, p4]
# legends = [f'9oct {group_name}',
#            f'2nov {group_name}',
#            '9oct OCS',
#            '2nov OCS'
#            ]
# ax.legend(lines, 
#           legends,
#           loc= 'upper center')

# plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
#             borderaxespad=0, frameon=False)

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
