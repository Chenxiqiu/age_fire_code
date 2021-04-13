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
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

dircInputs = {'MIPAS': 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/09_ENVISAT_MIPAS_with_AGEparams_final/',
              'ACE': 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/07_ACE_FTS_with_AGEparams_final/'
              }

##########input##########
def input_func(descrpt, psblt):
    while True:
        parameter = input(descrpt+':\n').upper()
        if parameter in psblt:
            break
        else:
            print(f'\nSorry, but {parameter} doesn\'t exist :O')
    return parameter

species = input_func('atmospheric species', ('OCS', 'N2O'))

postfix = 'tagged' #DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus
print(f'\npostfix: {postfix}\n')

##########functions##########
def group(df):
    df = df.copy()
    df.index = pd.cut(df.index, vrange)
    df.rename_axis(target, inplace=True)
    df = df.astype(np.float32)
    return {'mean':df['OCS'].mean(), 'std': df[species].std(), 'count': df[species].count()}

def data2plot(ins_name):
    frame = []
    for i in vrange[:-1]:
        df = pd.read_pickle(dircInputs[ins_name] + f'{ins_name}_{species}_{target}_({i}, {i+res}]_{postfix}.pkl')
        df.dropna(inplace=True)
        
        tag = 'stratospheric'
        df = df.loc[df['air']>=1].copy()
        
        if df.empty: 
            print(f'{target}: ({i}, {i+res}] is empty')
            break
        frame.append(pd.DataFrame(group(df), index=[pd.Interval(i, i+res)]))
        print(f'@{datetime.now().strftime("%H:%M:%S")} {ins_name} {target}: ({i}, {i+res}]')
    return pd.concat(frame), tag

def plot_age():
    
    colors = {'MIPAS' : 'forestgreen',
              'ACE': 'red',
              'AMICA': 'darkorchid'
              }    
    
    def plot(label=None, df=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()   
        ax.fill_betweenx(y=df.index.mid,
                         x1=df['mean']-df['std'], 
                         x2=df['mean']+df['std'],
                         label=label,
                         color=colors[label],
                         alpha=0.5)
        # ax.errorbar(x=df['mean'], 
        #             y=df.index.mid+shift, 
        #             xerr=df['std'],
        #             capsize=3,
        #             fmt='o', 
        #             label=label,
        #             color=colors[label],
        #             **kwargs
                    # )
        return ax
    
    fig = plt.figure(figsize=(10,50))
    font = {'size': 15}
    plt.rc('font', **font)
    ax1 = fig.add_subplot()
    
    plot(label='AMICA', df=stats_amica)
    plot(label='MIPAS', df=stats_mipas)
    plot(label='ACE', df=stats_ace)
    
    ax1.set(
            xlim=(0, 600) if species == 'OCS' else (0, 350),
            ylim=(vmin,vmax),
            )
    ax1.set_xlabel('OCS / ppt' if species == 'OCS' else 'N2O / ppb')
    ax1.set_ylabel(target)
    plt.legend()
    plt.title(f'{tag}')
    plt.show()

##########plotting##########
for target in np.intersect1d(c.ALL_AGEV_SOUTHTRAC, c.ALL_AGEV):
    vrange = c.VRANGE_AGE if target == 'AGE' else c.VRANGE
    vmin, vmax, res = c.VMIN, c.VMAX_AGE if target== 'AGE' else c.VMAX, c.VRES
    ##########data processing##########
    stats_ace, tag = data2plot('ACE')
    stats_mipas, tag = data2plot('MIPAS')
    data_southtrac = southtrac.read(strat=1).set_index(target)[species]
    data_southtrac.dropna(inplace=True)
    data_southtrac.index = pd.IntervalIndex(pd.cut(data_southtrac.index, vrange))
    data_southtrac.rename_axis(target, inplace=True)
    stats_amica = data_southtrac.groupby(target).agg({'mean', 'std', 'count'})
    plot_age()

winsound.Beep(freq, duration)
