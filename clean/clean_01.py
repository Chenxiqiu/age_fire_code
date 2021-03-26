import numpy as np
import matplotlib.pyplot as plt 
import datetime
import glob2
import xarray as xr
import pandas as pd
#plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/01_raw/01_ACE-FTS_L2_v3.5-6_FLAG/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/01_ACE-FTS_L2_v3.5-6_FLAG_cleaned/'


def open_data(species):
    df = xr.open_dataset(dircInput1 + f'ACEFTS_L2_v3p6_{species}.nc').to_dataframe().reset_index()
    return df
    
def flag(df):
    #values suggesting issues in quality_flag
    na_vals = [9, # Data fill value of −999 (no data)
               8, # Error fill value of −888 (data is scaled a priori)
               7, # Instrument or processing error
               5, # Extreme unnatural outlier detected from EDF
               4, # Moderate unnatural outlier detected from running MeAD 
               2 # Not enough data points in the region to do statistical analysis
               ]   
    df.loc[df['quality_flag'].isin(na_vals), species] = np.nan
    
    #my own filter for unreasonably large values
    if species == 'OCS':
        df.loc[df[species] >= 1000e-12, species] = np.nan # only very few data larger than 1000ppt 
        df.loc[df[species] <= -500e-12, species] = np.nan # filter out data lower than -500ppt
        
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
    df = flag(df)
    df = name(df)
    df = to_datetime(df)
    return df 

if __name__ == "__main__":
    species = input('species of interest: \n')
    df = clean(species)
    df.to_xarray().to_netcdf(dircInput2+f'ACEFTS_L2_v3p6_{species}_cleaned.nc')
    
