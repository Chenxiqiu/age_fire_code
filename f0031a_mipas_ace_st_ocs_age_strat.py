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

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/09_ENVISAT_MIPAS_with_AGEparams_final/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/07_ACE_FTS_with_AGEparams_final/'

def group(df, group, groupby_v, vmin, vmax, res):
    vrange = np.arange(vmin, vmax+res, res) 
    output = df[group].groupby([pd.cut(df[groupby_v], vrange)]).describe().reset_index() #apply(get_stats) .unstack()
    output[groupby_v.casefold()] = output.apply(lambda x: x[groupby_v].left+res/2,axis=1)
    output = output.loc[output['count']>=10]
    print('done')
    return output, vrange


species = 'OCS' #OCS N2O

target = 'MF_03'

postfix = 'tagged' #DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus

vmin, vmax, res = 0, 60 if target== 'AGE' else 100, 5

mipas = xr.open_dataset(dircInput1+f'MIPAS_OCS_{target}_{postfix}.nc').to_dataframe()
mipas, age_range = group(
                mipas[mipas['air'] >= 1].reset_index(), #_month9-11
                group='OCS', 
                groupby_v=target, 
                vmin=vmin, 
                vmax=vmax, 
                res=res
                )


ace = xr.open_dataset(dircInput2+f'ACE_OCS_{target}_{postfix}.nc').to_dataframe()
ace, _ = group(
               ace[ace['air'] >= 1].reset_index(),
                group='OCS', 
                groupby_v=target, 
                vmin=vmin, 
                vmax=vmax, 
                res=res
                )


southtrac = southtrac.read(strat=1)
southtrac, _ = group(
                    southtrac.reset_index(),
                    group='OCS', 
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
# spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15,1],width_ratios=[9,1])#,height_ratios=[15,1]) width_ratios=[9,1]

ax1 = fig.add_subplot() #spec[0, 0]
# x = df.OCS
# y = df[target] 
# main=ax.scatter(x,y,s=5) #c=df[color],cmap='rainbow',
# #main=ax.scatter (x,y,s=3, c='orange')

def plot(label=None, df=None, ax=None, shift=None, **kwargs):
    if ax is None:
        ax = plt.gca()   
    ax.errorbar(x=df['mean'], 
                y=df[target.casefold()]+shift, 
                xerr=df['std'],
                capsize=3,
                fmt='o', 
                label=label,
                color=colors[label],
                **kwargs
                )
    return ax

plot(label='MIPAS', df=mipas, shift=-res/20)
plot(label='SOUTHTRAC', df=southtrac, shift=0)
plot(label='ACE', df=ace, shift=res/10)

ax1.set(
        xlim=(0, 500) if species == 'OCS' else (0, 350),
        ylim=(vmin,vmax),
        )
ax1.set_xlabel('OCS / ppt' if species == 'OCS' else 'N2O / ppb')
ax1.set_ylabel(target)
plt.legend()
plt.title(f'{postfix}')

# ax = fig.add_subplot(spec[0,1])
# x = age_range[:-1]
# y = southtrac['count'].to_numpy()
# ax.barh(x,y, (age_range[1] - age_range[0]),align='edge')
# ax.set_ylim(0,vmax)
# ax.set_xlabel('#')
# ax.axes.yaxis.set_visible(False)

# ax = fig.add_subplot(spec[1,0])
# cbar=plt.colorbar(main, cax=ax,orientation='horizontal')
# cbar.set_label(color) 

plt.show()


