import numpy as np
import matplotlib.pyplot as plt 
import datetime
import glob2
import xarray as xr
import pandas as pd
#plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/02_semi_raw/07_ACE_FTS_with_AGEparams/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/07_ACE_FTS_with_AGEparams_cleaned/'

def open_data(year):
    fn = glob2.glob(dircInput1 + f'fts_v3.6_REN_HN2_ecmwf_3d_24_1.5_{str(year)[2:]}*.nc')
    frame = []
    for filename in fn:
        df = xr.open_dataset(filename).to_dataframe()
        df.reset_index(inplace=True)
        df.set_index('time', inplace = True)
        frame.append(df)    
    return pd.concat(frame)
    
# def flag(df):      
#     df.dropna(subset=[species], inplace=True)
#     return df

def name(df):
    df.rename(columns={
        'alt': 'ALT',
               }, 
              inplace = True)
    return df

def clean():
    df = open_data(year)
    # df = flag(df, )
    df = name(df)
    return df 

###############################################################################

if __name__ == "__main__":
    year = np.arange(2004,2017+1,1)
    for year in year:
        try:
            df = clean()
            df.to_xarray().to_netcdf(dircInput2+f'fts_v3.6_REN_HN2_ecmwf_3d_24_1.5__{year}_cleaned.nc')
            print (year, 'finished')
        except:
            print(f'no data for year {year}')
    