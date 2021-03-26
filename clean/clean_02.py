import numpy as np
import matplotlib.pyplot as plt 
import datetime
import glob2
import xarray as xr
import pandas as pd
#plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/01_raw/02_ACE-FTS_L2_v4.1_NETCDF/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/02_ACE-FTS_L2_v4.1_NETCDF_cleaned/'


def open_data(species):
    df = xr.open_dataset(dircInput1 + f'ACEFTS_L2_v4p1_{species}.nc').to_dataframe().reset_index()
    return df    
    
def flag(df, species):
    na_vals = [-999, -888]
    df.loc[df[species].isin(na_vals), species] = np.nan
    df.loc[df[f'{species}_error'].isin(na_vals), species] = np.nan
    
    #my own filter for unreasonably large values
    if species == 'OCS':
        df.loc[df[species] >= 1000e-12, species] = np.nan # only very few data larger than 1000ppt 
        df.loc[df[species] <= -500e-12, species] = np.nan # filter out data lower than -500ppt
    # elif species == 'SF6':
    #     df.loc[df[species] < 1e-11, species] = np.nan # all the percentiles are 
        
    df.dropna(subset=[species], inplace=True)
    return df

def name(df):
    df.rename(columns={
        'latitude': 'LAT',
        'longitude': 'LON',
        'altitude': 'ALT',
               }, 
              inplace = True)
    return df

def to_datetime(df):
    df['year'] = df['year'].apply(lambda x: int(x))
    df['month'] = df['month'].apply(lambda x: int(x))
    df['day'] = df['day'].apply(lambda x: int(x))
    df['time'] = pd.to_datetime(df[['year', 'month', 'day']])+df['hour'].apply(lambda x: pd.Timedelta(hours=x))
    return df

def clean(species):
    df = open_data(species)
    df = flag(df, )
    df = name(df)
    df = to_datetime(df)
    return df 

if __name__ == "__main__":
    species = input('species of interest: \n')
    df = clean(species)
    df.to_xarray().to_netcdf(dircInput2+f'ACEFTS_L2_v4p1_{species}_cleaned.nc')
    
