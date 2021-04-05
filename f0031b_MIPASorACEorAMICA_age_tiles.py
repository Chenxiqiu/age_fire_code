import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams, cycler
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob2
#import xarray as xr
import pandas as pd
import clean.clean_03 as southtrac
import process.constants as c
from matplotlib import gridspec
from datetime import datetime
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
#from varname import nameof
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

ins_name = 'MIPAS' #MIPAS ACE AMICA
species = 'OCS' #OCS N2O
postfix = 'tagged' #DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus


if ins_name not in ('MIPAS', 'ACE', 'AMICA'):
  raise Exception("instrument not recognized!")
if species not in ('OCS', 'N2O'):
  raise Exception("species not recognized!")

if species == 'OCS':
    mrmin, mrmax, mrres = c.OCSMIN_MIPAS if ins_name == 'MIPAS' else c.OCSMIN_ACE, c.OCSMAX, c.OCSRES
if species == 'N2O':
    mrmin, mrmax, mrres = c.N2OMIN_MIPAS if ins_name == 'MIPAS' else c.N2OMIN_ACE, c.N2OMAX, c.N2ORES

mrrange = np.arange(mrmin, mrmax+mrres, mrres)    
    

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/09_ENVISAT_MIPAS_with_AGEparams_final/' \
if ins_name == 'MIPAS' else 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/07_ACE_FTS_with_AGEparams_final/'

def group(df):    
    df.index = pd.cut(df.index, vrange)
    df.rename_axis(target, inplace=True)
    df[species] = pd.cut(df[species], mrrange)
    count = df.pivot_table(index=target, columns=species, values='air', aggfunc='count')
    total_count = count.max(axis=1).rename("total_count")
    relative_count = count.div(total_count, axis=0)
    return relative_count, total_count



def data2tiles_southtrac():
        
    tag = 'stratospheric'
    df = southtrac.read(strat=1).reset_index(target).sort_index()
    
    relative_count, total_count = group(df)
    return relative_count, total_count, tag
    
def plot_age():    
    camps = {'MIPAS' : 'YlGn',
              'ACE': 'OrRd',
              'SOUTHTRAC': ' '
              }
    
    fig = plt.figure(figsize=(10, 50))
    font = {'size': 15}
    plt.rc('font', **font)
    spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15, 1],width_ratios=[9, 1])#,height_ratios=[15,1]) width_ratios=[9,1]
    
    ax1 = fig.add_subplot(spec[0, 0])
    plot_data = relative_count['relative_count'].unstack()
    x = plot_data.index.left.union(plot_data.index.right)
    y = plot_data.columns.left.union(plot_data.columns.right)             
    y, x = np.meshgrid(y, x)
    z = plot_data
    main = ax1.pcolor (x, y, z, cmap=camps[ins_name])
    
    ax1.set(
            xlim=(mrmin, mrmax),
            ylim=(vmin,vmax),
            )
    ax1.set_xlabel('OCS / ppt' if species == 'OCS' else 'N2O / ppb')
    ax1.set_ylabel(target)
    plt.title(f'{ins_name}_{tag}')
    
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)
    plot_data = relative_count['count'].unstack()
    x, y = plot_data.columns.left, plot_data.sum(0)
    ax2.barh(x, y, res, align='edge', color='powderblue')
    ax2.set_xscale('log')
    ax2.set_xlabel('#')
    ax2.set_xlim(0, 1e8)
    ax2.axes.yaxis.set_visible(False)
    
    ax3 = fig.add_subplot(spec[1, 0])
    cbar=plt.colorbar(main, cax=ax3, orientation='horizontal')
    cbar.set_label(f'% of # relative to the highest bin given {target}') 
    
    plt.show()    

for target in c.ALL_AGEV_SOUTHTRAC if ins_name == 'AMICA' else c.ALL_AGEV:
        vrange = c.VRANGE_AGE if target == 'AGE' else c.VRANGE
        vmin, vmax, res = c.VMIN, c.VMAX_AGE if target== 'AGE' else c.VMAX, c.VRES
        relative_count, total_count, tag = data2tiles_southtrac() if  ins_name == 'AMICA' else data2tiles_satellites()
        plot_age()

winsound.Beep(freq, duration)








