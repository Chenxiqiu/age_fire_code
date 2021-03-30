import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams, cycler
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob2
#import xarray as xr
import pandas as pd
import clean.clean_03 as southtrac
from matplotlib import gridspec
#from varname import nameof

import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

def get_stats(group):
#    return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.quantile(),'SD': group.std()}
    return {'count': group.count(), 
            'mean': group.mean(), 
            #'median': group.quantile(),
            'SD': group.std(),
            }

def group(df, groupby_v, vmin, vmax, res):
    vrange = np.arange(vmin, vmax+res, res)
    df.dropna(inplace=True)
    df = df.loc[(df['air']>=1)]
    df.drop(columns='air', inplace=True)
    tag = 'stratospheric'
    output = pd.DataFrame(columns=['count', 'mean', 'SD'])
    output['count'] = df[species].groupby([pd.cut(df.index, vrange)]).count()
    output['mean'] = df[species].groupby([pd.cut(df.index, vrange)]).mean()
    output['std'] = df[species].groupby([pd.cut(df.index, vrange)]).std()
    output.reset_index(inplace=True)
    output[groupby_v.casefold()] = output.apply(lambda x: x['index'].left+res/2,axis=1)
    output = output.loc[output['count']>=10]
    print('done')
    return output, vrange, df, tag

ins_name = 'MIPAS' #MIPAS ACE
species = 'OCS' #OCS N2O
target = 'AGE'
postfix = 'tagged' #DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus

vmin = 0
vmax = 100
res = 5

if ins_name == 'MIPAS':
    dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/09_ENVISAT_MIPAS_with_AGEparams_final/'
elif ins_name == 'ACE':
    dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/07_ACE_FTS_with_AGEparams_final/'

#df = xr.open_dataset(dircInput1+f'{ins_name}_{species}_{target}_{postfix}.nc').to_dataframe()

ins_data, age_range, df, tag = group(
                pd.read_pickle(dircInput1 + f'{ins_name}_{species}_{target}_{postfix}.pkl'), #_month9-11 
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
ax1.errorbar(ins_data['mean'], 
             ins_data[target.casefold()], # - res/10, 
             xerr=ins_data['std'], 
             fmt='o', 
             color=colors[ins_name], 
             label=ins_name, 
             capsize=3)
ax1.scatter(df[species], df.index, s=0.1, c='silver')
ax1.set(
        xlim=(0, 500) if species == 'OCS' else (0, 350),
        ylim=(vmin,vmax),
        )
ax1.set_xlabel('OCS / ppt' if species == 'OCS' else 'N2O / ppb')
ax1.set_ylabel(target)
plt.legend()
plt.title(f'{tag}')

ax2 = fig.add_subplot(spec[0,1], sharey=ax1)
x = ins_data[target.casefold()]-res/2
y = ins_data['count']
ax2.barh(x,y, res,align='edge')
ax2.set_ylim(0,vmax)
ax2.set_xlabel('#')
ax2.axes.yaxis.set_visible(False)

# ax = fig.add_subplot(spec[1,0])
# cbar=plt.colorbar(main, cax=ax,orientation='horizontal')
# cbar.set_label(color) 

plt.show()

view = df.head()
