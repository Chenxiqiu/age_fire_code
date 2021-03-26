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
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
import seaborn as sns

# def get_stats(group):
# #    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.quantile(),'SD': group.std()}
#     return {'count': group.count(), 
#             'mean': group.mean(), 
#             'median': group.quantile(),
#             'SD': group.std(),
#             'p10': group.quantile(0.1),
#             'p25': group.quantile(0.25),
#             'p75': group.quantile(0.75),
#             'p90': group.quantile(0.9)
#             }

plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

lat_max = -53.7
lat_min = -70.3

#df=southtrac.read(res=30,lat_max,lat_min,9000) # amica 
#df=southtrac.read(res=30, LAT = [lat_min, lat_max], ALT=9000)
df=southtrac.read(strat=1, local=1)

def group(df, group, groupby, vmin, vmax, res):
    vrange = np.arange(vmin, vmax+res, res) 
    output = df[group].groupby([pd.cut(df[groupby], vrange)]).describe().reset_index() #apply(get_stats) .unstack()
    output[groupby.casefold()] = output.apply(lambda x: x[groupby].left+res/2,axis=1)
    print('done')
    return output, vrange

target = 'AGE' 
stats, vrange = group(df, 'OCS', target, 0, 40, 5)

###############################################################################
fig = plt.figure(figsize=(10,50))
spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15,1],width_ratios=[9,1])#,height_ratios=[15,1]) width_ratios=[9,1]

ax = fig.add_subplot(spec[0,0])

# main=ax.scatter (x,y,s=5) #c=df[color],cmap='rainbow',
df1 = df[df['air']==1]
df2 = df[df['air']==2]
main=ax.scatter (df1.OCS,
                 df1[target],
                 s=3, 
                 c='tab:orange',
                 label='mixed air in the tropopause layer')
main=ax.scatter (df2.OCS,
                 df2[target],
                 s=3, 
                 c='tab:green',
                 label='mixed air in the tropopause layer')
# ax.errorbar(stats['mean'],age_range[:-1] + age_res / 2, xerr=stats.SD, fmt='o', color='black')
ax.set_xlim(100,600)
ax.set_ylim(0,40)
ax.set_xlabel('OCS / ppt')
ax.set_ylabel(target)

ax = fig.add_subplot(spec[0,1])
ax.barh(vrange[:-1],
        stats['count'].to_numpy(), 
        5,
        align='edge')
ax.set_ylim(0,40)
ax.set_xlabel('#')
ax.axes.yaxis.set_visible(False)

# ax = fig.add_subplot(spec[1,0])
# cbar=plt.colorbar(main, cax=ax,orientation='horizontal')
# cbar.set_label(color) 

plt.show()