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

postfix = 'JJA_LAT30Sminus_THETA430minus_PV2plus' 
#DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus
target = 'AGE'
vmin = 0
vmax = 60
res = 5

ace = xr.open_dataset(dircInput2+f'ACE_N2O_{target}_{postfix}.nc').to_dataframe()
ace, _ = group(
                ace.reset_index(),
                group='N2O', 
                groupby_v=target, 
                vmin=vmin, 
                vmax=vmax, 
                res=res
              )

southtrac = southtrac.read(local=1)
southtrac, _ = group(
                    southtrac[southtrac.air >= 1].reset_index(),
                    group='UMAQS_N2O', 
                    groupby_v=target, 
                    vmin=vmin, 
                    vmax=vmax, 
                    res=res
                    )
###############################################################################
fig = plt.figure(figsize=(10,50))
spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15,1],width_ratios=[9,1])#,height_ratios=[15,1]) width_ratios=[9,1]

ax = fig.add_subplot(spec[0,0])
# x = df.OCS
# y = df[target] 
# main=ax.scatter(x,y,s=5) #c=df[color],cmap='rainbow',
# #main=ax.scatter (x,y,s=3, c='orange')
# ax.errorbar(mipas['mean'], mipas[target.casefold()] - res/10, xerr=mipas['std'], fmt='o', color='black', 
#             label='mipas')
ax.errorbar(ace['mean'], ace[target.casefold()], xerr=ace['std'], fmt='o', color='orange', 
            label='ace')
ax.errorbar(southtrac['mean'], southtrac[target.casefold()] + res/10, xerr=southtrac['std'], fmt='o', color='blue',
            label='southtrac')
ax.set_xlim(0,600)
ax.set_ylim(0,vmax)
ax.set_xlabel('OCS / ppt')
ax.set_ylabel(target)
plt.legend()
plt.title(f'{postfix}')

# ax = fig.add_subplot(spec[0,1])
# x = southtrac.apply(lambda x: x[target].left,axis=1)
# y = southtrac['count']
# ax.barh(x,y, (age_range[1] - age_range[0]),align='edge')
# ax.set_ylim(0,vmax)
# ax.set_xlabel('#')
# ax.axes.yaxis.set_visible(False)

# ax = fig.add_subplot(spec[1,0])
# cbar=plt.colorbar(main, cax=ax,orientation='horizontal')
# cbar.set_label(color) 


plt.show()


