import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
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

def input_func(descrpt, psblt):
    while True:
        parameter = input(descrpt+':\n').upper()
        if parameter in psblt:
            break
        else:
            print(f'\nSorry, but {parameter} doesn\'t exist :O')
    return parameter

ins_name = input_func('instrument name', ('MIPAS', 'ACE', 'AMICA'))
species = input_func('atmospheric species', ('OCS', 'N2O'))
target_button = input_func(
    'all parameters(leave empty) or specific parameter (type name)', 
    ('',)+c.ALL_AGEV_SOUTHTRAC+c.ALL_AGEV
    )
                           
postfix = 'tagged' #DJF_LAT30Nplus_THETA430minus_PV2plus JJA_LAT30Sminus_THETA430minus_PV2plus
print(f'\npostfix: {postfix}')

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

def data2tiles_satellites():
    frame_relative_count = []
    frame_total_count = []
    for i in vrange[:-1]:
        df = pd.read_pickle(dircInput1 + f'{ins_name}_{species}_{target}_({i}, {i+res}]_{postfix}.pkl')
        df.dropna(inplace=True)
        
        tag = 'stratospheric'
        df = df.loc[df['air']>=1].copy()
        
        if df.empty: 
            print(f'{target}: ({i}, {i+res}] is empty')
            break
        relative_count, total_count = group(df)
        frame_relative_count.append(relative_count)
        frame_total_count.append(total_count)
        print(f'@{datetime.now().strftime("%H:%M:%S")}   {target}: ({i}, {i+res}]')
    relative_count = pd.concat(frame_relative_count)
    total_count = pd.concat(frame_total_count)
    return relative_count, total_count, tag
    
def data2tiles_southtrac():
        
    tag = 'stratospheric'
    df = southtrac.read(strat=1).set_index(target).sort_index()
    
    relative_count, total_count = group(df)
    return relative_count, total_count, tag
    
def plot_age():    
    camps = {
        'MIPAS' : 'YlGn',
        'ACE': 'OrRd',
        'AMICA': 'BuPu'
        }

    cmap = cm.get_cmap(camps[ins_name])
    barcolor = cmap(0.1)
    
    fig = plt.figure(figsize=(10, 50))
    font = {'size': 18}
    plt.rc('font', **font)
    spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15, 1],width_ratios=[9, 1])#,height_ratios=[15,1]) width_ratios=[9,1]
    
    ax1 = fig.add_subplot(spec[0, 0])
    x = np.append([i.left for i in relative_count.columns], relative_count.columns[-1].right)
    y = np.append([i.left for i in relative_count.index], relative_count.index[-1].right)
    x, y = np.meshgrid(x, y)
    main = ax1.pcolormesh (x, y, relative_count, cmap=cmap, shading='flat')
    
    ax1.set(
            xlim=(mrmin, mrmax),
            ylim=(vmin,vmax),
            )
    ax1.set_xlabel('OCS [ppt]' if species == 'OCS' else 'N2O [ppb]', fontweight='bold')
    ax1.set_ylabel(f'{target} [month]' if target in ['AGE', 'p50'] else f'{target} [%]', fontweight='bold')
    plt.title(f'{ins_name}_{tag}')
    
    ax2 = fig.add_subplot(spec[0, 1], sharey=ax1)
    x = [i.left for i in total_count.index]
    y = total_count.values
    ax2.barh(x, y, res, align='edge', color=barcolor)
    ax2.set_xscale('log')
    ax2.set_xlabel('#')
    ax2.set_xlim(1, 1e3 if ins_name=='AMICA' else 1e8)
    ax2.axes.yaxis.set_visible(False)
    
    ax3 = fig.add_subplot(spec[1, 0])
    cbar=plt.colorbar(main, cax=ax3, orientation='horizontal')
    cbar.set_label(f'% of # relative to the highest bin given {target}') 
  
if target_button != '':
    target = target_button
    vrange = c.VRANGE_AGE if target == 'AGE' else c.VRANGE
    vmin, vmax, res = c.VMIN, c.VMAX_AGE if target== 'AGE' else c.VMAX, c.VRES
    relative_count, total_count, tag = data2tiles_southtrac() if  ins_name == 'AMICA' else data2tiles_satellites()
    plot_age()    
else:
    for target in c.ALL_AGEV_SOUTHTRAC if ins_name == 'AMICA' else c.ALL_AGEV:
        vrange = c.VRANGE_AGE if target == 'AGE' else c.VRANGE
        vmin, vmax, res = c.VMIN, c.VMAX_AGE if target== 'AGE' else c.VMAX, c.VRES
        relative_count, total_count, tag = data2tiles_southtrac() if  ins_name == 'AMICA' else data2tiles_satellites()
        plot_age()

winsound.Beep(freq, duration)








