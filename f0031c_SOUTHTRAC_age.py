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
#from varname import nameof

import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

def group(df, group, groupby_v, vmin, vmax, res):
    vrange = np.arange(vmin, vmax+res, res) 
    output = df[group].groupby([pd.cut(df[groupby_v], vrange)]).describe().reset_index() #apply(get_stats) .unstack()
    output[groupby_v.casefold()] = output.apply(lambda x: x[groupby_v].left+res/2,axis=1)
    output = output.loc[output['count']>=10]
    print('done')
    return output, vrange

species = 'UMAQS_N2O' #OCS N2O

target = 'MF_24'

vmin = 0
vmax = 100
res = 5

df = southtrac.read(strat=1) #local=1, 
ins_data, age_range = group(
                    df.reset_index(),
                    group=species, 
                    groupby_v=target, 
                    vmin=vmin, 
                    vmax=vmax, 
                    res=res
                    )
###############################################################################
colors = {'MIPAS' : 'firebrick',
          'ACE': 'orangered',
          'SOUTHTRAC': 'teal'
          }

fig = plt.figure(figsize=(10,50))
spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15,1],width_ratios=[9,1])#,height_ratios=[15,1]) width_ratios=[9,1]

ax1 = fig.add_subplot(spec[0,0])
# x = df.OCS
# y = df[target] 
# main=ax.scatter(x,y,s=5) #c=df[color],cmap='rainbow',
# #main=ax.scatter (x,y,s=3, c='orange')
ax1.errorbar(ins_data['mean'], 
             ins_data[target.casefold()], # - res/10, 
             xerr=ins_data['std'], 
             fmt='o', 
             color=colors['SOUTHTRAC'], 
             label='SOUTHTRAC', 
             capsize=3)
ax1.scatter(df[species], df[target], s=1, c='darkorange', label='stratospheric air')
southtrac_mixed = df[df.MF_06>=30]
ax1.scatter(southtrac_mixed[species], southtrac_mixed[target], s=1, c='forestgreen', label='mixed air in tropopause layer')

ax1.set(
        xlim=(0, 600) if species == 'OCS' else (0, 350),
        ylim=(vmin,vmax),
        )
ax1.set_xlabel('OCS / ppt' if species == 'OCS' else 'N2O / ppb')
ax1.set_ylabel(target)
plt.legend()
plt.title('all flights')

ax2 = fig.add_subplot(spec[0,1], sharey=ax1)
x = ins_data.apply(lambda x: x[target].left,axis=1)
y = ins_data['count']
ax2.barh(x,y, res,align='edge')
ax2.set_xlabel('#')
ax2.axes.yaxis.set_visible(False)

# ax = fig.add_subplot(spec[1,0])
# cbar=plt.colorbar(main, cax=ax,orientation='horizontal')
# cbar.set_label(color) 

plt.show()


