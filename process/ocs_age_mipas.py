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
from matplotlib import gridspec
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'
import constants as c

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/09_ENVISAT_MIPAS_with_AGEparams_cleaned/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/09_ENVISAT_MIPAS_with_AGEparams_final/'

fn = glob2.glob(dircInput1+'*.nc')

species = 'OCS'

for v in ['AGE', 'MF_24', 'MF_03', 'MF_06', 'MF_12', 'MF_48']: 
    frame = []
    for f in fn:
        df = xr.open_dataset(f).to_dataframe().reset_index()
        df = df[['LAT', species, 'THETA', 'PV', v, 'time']] 
        if v == 'AGE':
            df['AGE'] = df['AGE']*12
        if species == 'OCS':
            df[species] = df[species]*1e12
        
        tag = 'tagged'
        df.loc[:, 'air'] = 0
        strat_filter = (abs(df['PV']) > 2) | (df['THETA'] > 380)
        real_strat_filter = abs(df['PV']) >= 4   
        df.loc[strat_filter, 'air'] = 1 
        df.loc[real_strat_filter, 'air'] = 2          
        
        # tag = 'JJA_LAT30Sminus_THETA430minus_PV2plus' #JJA_LAT30Sminus_THETA430minus_PV2plus DJF_LAT30Nplus_THETA430minus_PV2plus
        # df = df[(df.time.dt.month == 6) | (df.time.dt.month == 7) | (df.time.dt.month == 8)]
        # df = df[df.LAT <= -30]
        # df = df[df.THETA <= 430]
        # df = df[abs(df.PV) >= 2]
        
        # tag = 'stratospheric'
        # df = df.loc[(df['air']>=1)]
        # df.drop(columns='air', inplace=True)
        
        try: 
            df = df[[species, v, 'air']].copy()
        except: 
            df = df[[species, v]].copy()
        df = df.astype(np.float16)
        frame.append(df)
    
    df = pd.concat(frame)            
    df.set_index(v, inplace=True)   
    df.sort_index(inplace=True) 
    vrange = c.VRANGE_AGE if v== 'AGE' else c.VRANGE
    df.dropna(inplace=True)
    
    grouped = df.groupby([pd.cut(df.index, vrange)])
    for name, group in grouped:
        group.to_pickle(dircInput2+f'MIPAS_OCS_{v}_{name}_{tag}.pkl') #df.to_xarray().to_netcdf(dircInput2+f'MIPAS_OCS_{v}_{tag}.nc')
        print(f'MIPAS_OCS_{v}_{name}_{tag}.pkl')