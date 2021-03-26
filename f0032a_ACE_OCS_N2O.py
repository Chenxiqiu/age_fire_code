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

species = 'OCS' #OCS N2O
target = 'N2O'

ins_name = 'ACE' #MIPAS ACE
postfix = 'tagged' #DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus

if ins_name == 'MIPAS':
    dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/09_ENVISAT_MIPAS_with_AGEparams_final/'
elif ins_name == 'ACE':
    dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/07_ACE_FTS_with_AGEparams_final/'

df = xr.open_dataset(dircInput1+f'{ins_name}_{species}_{target}_{postfix}.nc').to_dataframe()

df = df[df['air']>=1]

vmin = 0
vmax = 350
res = 25

ins_data, age_range = group(
                df.reset_index(), #_month9-11
                group=species, 
                groupby_v=target, 
                vmin=vmin, 
                vmax=vmax, 
                res=res
                )

# southtrac = southtrac.read(local=1)
# southtrac, _ = group(
#                     southtrac[southtrac.air >= 1].reset_index(),
#                     group='OCS', 
#                     groupby_v=target, 
#                     vmin=vmin, 
#                     vmax=vmax, 
#                     res=res
#                     )
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
ax1.errorbar(ins_data['mean'], ins_data[target.casefold()] - res/10, xerr=ins_data['std'], fmt='o', color=colors[ins_name], 
            label=ins_name)
ax1.scatter(df[species], df.index, s=0.1, c='silver')
ax1.set(
        xlim=(0, 600) if species == 'OCS' else (0, 350),
        ylim=(vmin,vmax),
        )
ax1.set_xlabel('OCS / ppt' if species == 'OCS' else 'N2O / ppb')
ax1.set_ylabel(target)
plt.legend()
plt.title(f'{postfix}')

ax2 = fig.add_subplot(spec[0,1], sharey=ax1)
x = ins_data.apply(lambda x: x[target].left,axis=1)
y = ins_data['count']
ax2.barh(x,y, res,align='edge')
ax2.set_ylim(0,vmax)
ax2.set_xlabel('#')
ax2.axes.yaxis.set_visible(False)

# ax = fig.add_subplot(spec[1,0])
# cbar=plt.colorbar(main, cax=ax,orientation='horizontal')
# cbar.set_label(color) 

plt.show()


