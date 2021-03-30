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
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/07_ACE_FTS_with_AGEparams_cleaned/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/04_final/07_ACE_FTS_with_AGEparams_final/'

fn = tuple(glob2.glob(dircInput1+'*.nc'))

species_list = ('OCS', 'N2O')
v_list = ('AGE', 'MF_03', 'MF_06', 'MF_12', 'MF_24', 'MF_48')

# tag = {
#        'DJF_LAT30Nplus_THETA430minus_PV2plus': lambda df: ((df.time.dt.month == 12) | (df.time.dt.month == 1) | (df.time.dt.month == 2))
#                                                            & (df.LAT >= 30)
#                                                            & (df.THETA <= 430)
#                                                            & (abs(df.PV) >= 2)
#        'tagged'                                                    
#       }

vmin = 0
vmax = 100
res = 5

for species in species_list:
    for v in v_list: 
        frame = []
        for fn in fn:
            df = xr.open_dataset(fn).to_dataframe().reset_index()
            df = df[['LAT', 'LON', species, 'THETA', 'PV', v, 'time']]
            try:
                df['AGE'] = df['AGE']*12
            except:
                pass
            try: 
                df['OCS'] = df['OCS']*1e12
            except: 
                pass
            try: 
                df['N2O'] = df['N2O']*1e9
            except: 
                pass
                        
            # tag = 'DJF_LAT30Nplus_THETA430minus_PV2plus'
            #df = df[(df.time.dt.month == 12) | (df.time.dt.month == 1) | (df.time.dt.month == 2)]
            #df = df[df.LAT >= 30]
            #df = df[df.THETA <= 430]
            #df = df[abs(df.PV) >= 2]
            
            tag = 'tagged'
            df.loc[:, 'air'] = 0
            strat_filter = (abs(df['PV']) > 2) | (df['THETA'] > 380)
            real_strat_filter = abs(df['PV']) >= 4   
            df.loc[strat_filter, 'air'] = 1 
            df.loc[real_strat_filter, 'air'] = 2  
            
            df = df[[species, v, 'air']]
            #df = df.astype(np.float16)
            print (fn)
            frame.append(df)
    
        df = pd.concat(frame)
        #df.rename(columns = {'alt':'ALT'}, inplace = True)    
        df.set_index(v, inplace=True)   
        df.sort_index(inplace=True)
        vrange = c.VRANGE_AGE if v== 'AGE' else c.VRANGE
        grouped = df.groupby([pd.cut(df.index, vrange)])
        for name, group in grouped:
            group.to_pickle(dircInput2+f'ACE_{species}_{v}_{name}_{tag}.pkl') #df.to_xarray().to_netcdf(dircInput2+f'MIPAS_OCS_{v}_{tag}.nc') 
            print(f'@{datetime.now().strftime("%H:%M:%S")}  ACE_OCS_{v}_{name}_{tag}.pkl')  
            
winsound.Beep(freq, duration)