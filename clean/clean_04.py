import numpy as np
import matplotlib.pyplot as plt 
import datetime
import glob2
import xarray as xr
import pandas as pd
#plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/02_semi_raw/04_imk_ncdf/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/04_imk_ncdf_cleaned/'

def open_data(year):
    df = xr.open_mfdataset(dircInput1 + f'mipas_imk_{str(year)[2:]}*.nc', 
                           combine='by_coords').to_dataframe().reset_index()
    return df    
    
def flag(df):
    species = []
    na_vals = [0]
    for species in ['O3', 'N2O', 'CO', 'CH4']:
        df.loc[df[f'{species}@VISIBILITY_FLAG'].isin(na_vals), species] = np.nan
    df.dropna(subset=[species], inplace=True)
    return df

def name(df):
    df.rename(columns={
        'alt': 'ALT',
               }, 
              inplace = True)
    return df

def clean():
    df = open_data(year)
    df = flag(df)
    df = name(df)
    return df 

if __name__ == "__main__":
    year = np.arange(2002,2012+1,1)
    for year in year:
        try:
            df = clean()
            df.to_xarray().to_netcdf(dircInput2+f'mipas_imk_{year}_cleaned.nc')
            print (year, 'finished')
        except: 
            print(f'{year} is not available')
    