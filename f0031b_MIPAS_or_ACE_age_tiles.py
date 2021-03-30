import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rcParams, cycler
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob2
#import xarray as xr
import pandas as pd
import clean.clean_03 as southtrac
import constants as c
from matplotlib import gridspec
from datetime import datetime
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
#from varname import nameof
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

def group(df, groupby_v):    
    df.loc[:, species] = pd.cut(df[species], mrrange)
    df.reset_index(inplace=True)
    df.set_index(species, inplace=True)
    df.sort_index(inplace=True)
    output = pd.DataFrame(columns=['count','relative_count'])
    output.loc[:, 'count'] = df[target].groupby(species).count()#apply(get_stats).unstack() #apply(get_stats) .unstack()
    output.loc[:, 'relative_count'] = output['count'] / np.nanmax(output['count'])
    output.index = output.index.astype('interval').rename(species)
    return output, tag

ins_name = 'ACE' #MIPAS ACE
species = 'OCS' #OCS N2O
postfix = 'tagged' #DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus


if ins_name not in ('MIPAS', 'ACE'):
  raise Exception("instrument not recognized!")
# if target not in ('AGE', 'MF_03', 'MF_06', 'MF_12', 'MF_24', 'MF_48'):
#   raise Exception("age variable not recognized!")
if species not in ('OCS', 'N2O'):
  raise Exception("species not recognized!")

if species == 'OCS':
    mrmin, mrmax, mrres = c.OCSMIN_MIPAS if ins_name == 'MIPAS' else c.OCSMIN_ACE, c.OCSMAX, c.OCSRES
if species == 'N2O':
    mrmin, mrmax, mrres = c.N2OMIN_MIPAS if ins_name == 'MIPAS' else c.N2OMIN_ACE, c.N2OMAX, c.N2ORES

mrrange = np.arange(mrmin, mrmax+mrres, mrres)    
    

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/09_ENVISAT_MIPAS_with_AGEparams_final/' \
if ins_name == 'MIPAS' else 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/07_ACE_FTS_with_AGEparams_final/'

for target in ('AGE', 'MF_03', 'MF_06', 'MF_12', 'MF_24', 'MF_48'):
    vmin, vmax, res = c.VMIN, c.VMAX_AGE if target== 'AGE' else c.VMAX, c.VRES
    frame = []
    for i in range(vmin, vmax, res):
        df = pd.read_pickle(dircInput1 + f'{ins_name}_{species}_{target}_({i}, {i+res}]_{postfix}.pkl')
        df.dropna(inplace=True)
        
        tag = 'stratospheric'
        df = df.loc[df['air']>=1].copy()
        df.drop(columns='air', inplace=True)
        
        if df.empty: break
        ins_data, tag = group(df, groupby_v=target,)
        ins_data.loc[:, target] = pd.Interval(i, i + res)
        ins_data.set_index(target, append=True, inplace=True)
    
        #ins_data.loc[:, target.casefold()] = pd.Interval(i, i + res), i + res/2
        
        frame.append(ins_data)
        print(f'@{datetime.now().strftime("%H:%M:%S")}   {target}: ({i}, {i+res}]')
    ins_data = pd.concat(frame) 
    
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
    
    fig = plt.figure(figsize=(10, 50))
    font = {'size': 15}
    plt.rc('font', **font)
    spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15, 1],width_ratios=[9, 1])#,height_ratios=[15,1]) width_ratios=[9,1]
    
    ax1 = fig.add_subplot(spec[0, 0])
    plot_data = ins_data['relative_count'].unstack()
    x = plot_data.index.left.union(plot_data.index.right)
    y = plot_data.columns.left.union(plot_data.columns.right)             
    y, x = np.meshgrid(y, x)
    z = plot_data
    main = ax1.pcolor (x, y, z, cmap='YlGn' if ins_name == 'MIPAS' else 'OrRd', shading='flat')
    
    ax1.set(
            xlim=(mrmin, mrmax),
            ylim=(vmin,vmax),
            )
    ax1.set_xlabel('OCS / ppt' if species == 'OCS' else 'N2O / ppb')
    ax1.set_ylabel(target)
    plt.title(f'{ins_name}_{tag}')
    
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)
    plot_data = ins_data['count'].unstack()
    x, y = plot_data.columns.left, plot_data.sum(0)
    ax2.barh(x, y, res, align='edge', color='powderblue')
    ax2.set_xlabel(f'# for each {target} bin')
    ax2.axes.yaxis.set_visible(False)
    
    ax3 = fig.add_subplot(spec[1, 0])
    cbar=plt.colorbar(main, cax=ax3, orientation='horizontal')
    cbar.set_label(f'% of # relative to the highest bin given {target}') 
    
    plt.show()
    winsound.Beep(freq, duration)








