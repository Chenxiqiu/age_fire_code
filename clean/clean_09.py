import numpy as np
import matplotlib.pyplot as plt 
import datetime
import glob2
import xarray as xr
import pandas as pd
#plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/02_semi_raw/09_ENVISAT_MIPAS_with_AGEparams/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/09_ENVISAT_MIPAS_with_AGEparams_cleaned/'

species = 'OCS'

def open_data(year):
    df = xr.open_mfdataset(dircInput1 + f'MIPAS_OCS_REN_HN2_ecmwf_3d_24_1.5_{str(year)[2:]}*.nc', 
                           combine='by_coords').to_dataframe()
    df = df.reset_index()
    return df#df.reset_index(inplace=True)
    
# def flag(df):      
#     df.dropna(subset=[species], inplace=True)
#     return df

def name(df):
    df.rename(columns={
        'alt': 'ALT',
               }, 
              inplace = True)
    return df

def rescale(df):
    df['AGE'] = df['AGE'] * 12
    return df
    
def clean():
    df = open_data(year)
    # df = flag(df, )
    df = name(df)
    df.reset_index('time', inplace=True)
    return df 

###############################################################################

if __name__ == "__main__":
    year = np.arange(2002,2012+1,1)
    for year in year:
        df = clean()
        df.to_xarray().to_netcdf(dircInput2+f'MIPAS_OCS_REN_HN2_ecmwf_3d_24_1.5_{year}_cleaned.nc')
        print (year, 'finished')
    
    df = xr.open_dataset('C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/09_ENVISAT_MIPAS_with_AGEparams_cleaned/'+'MIPAS_OCS_REN_HN2_ecmwf_3d_24_1.5_2004_cleaned.nc').to_dataframe()
    view=df.describe()