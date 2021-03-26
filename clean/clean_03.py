import numpy as np
import datetime
import glob2
import xarray as xr
import pandas as pd
import re
#plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

dircInput1 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/01_raw/03_HALO-DB_mission_116/'
dircInput2 = 'C:/Users/Chenxi/OneDrive/phd/age_and_fire/data/03_cleaned/03_HALO-DB_mission_116_cleaned/'


#file names
fn_amica_database = glob2.glob(dircInput1 + 'AMICA/' + '*.ames')
fn_amica_wiki = glob2.glob(dircInput1 + 'AMICA/' + '*.dat')
fn_bahamas = glob2.glob(dircInput1 + 'BAHAMAS/' + '*.nc')
fn_fairo = glob2.glob(dircInput1 + 'FAIRO/' + '*FAIRO_O3*.ames')
fn_fairoCI = glob2.glob(dircInput1 + 'FAIRO/' + '*FAIROCI_O3*.ames')
fn_fish = glob2.glob(dircInput1 + 'FISH/' + '*FISH_H2O.ames')
fn_umaq = glob2.glob(dircInput1 + 'UMAQ/' + '*.ames')
fn_aeneas = glob2.glob(dircInput1 + 'AENEAS/' + '*.ames')
fn_hagar_li = glob2.glob(dircInput1+'*HAGARV_LI_prelim.ames')
fn_hagar_ecd = glob2.glob(dircInput1+'*HAGARV_ECD_preliminary.ames')
fn_ghost = glob2.glob(dircInput1+'*GhOST_MS_preliminary.ames')


fn_Met_V2 = glob2.glob(dircInput1 + 'CLAMS_Met/' + '*CLAMS_Met_V2.nc')
fn_agespec_HN2 = glob2.glob(dircInput1+'*CLAMS_agespec_HN2.nc')
# fn_backtraj_nr_ST1_cfc_clim = glob2.glob(dircInput1+'clams_at_halo/'+'*backtraj_nr_ST1_cfc_clim.nc')
# fn_backtraj_nr_ST1_clim = glob2.glob(dircInput1+'clams_at_halo/'+'*backtraj_nr_ST1_clim.nc')
fn_sfctracer_F02 = glob2.glob(dircInput1+'*CLAMS_sfctracer_F02.nc')
fn_chem_V1 = glob2.glob(dircInput1+'*CLAMS_chem_V1.nc') 
#fn_gloria_kit = glob2.glob(dircInput1+'*GLORIA_chemistry_mode_KIT.nc')
#fn_gloria_fzj = glob2.glob(dircInput1+'*GLORIAFZJ_L1V0002preL2V00pre.nc')

############################in-situ measurements###############################
def clean_bahamas(res, fn = fn_bahamas):
    frame = []
    for filename in fn:
        df = xr.open_dataset(filename).to_dataframe()
        df.rename(columns = {'TIME': "time", 
                              'IRS_ALT': 'ALT',
                              'IRS_LAT': 'LAT',
                              'IRS_LON': 'LON',
                              }, inplace = True)
        df.set_index('time', inplace = True)
        df = df.resample(f'{res}S').mean()
        df['flight'] = df.index[0].strftime('%Y-%m-%d')
        frame.append(df)    
    output = pd.concat(frame)
    output.sort_index(inplace = True)
    return output[['ALT','LAT','LON','THETA','flight']]

def clean_amica(res, fn = fn_amica_wiki):
    frame = []
    for filename in fn:
        df = pd.read_csv(
                         filename,
                         delimiter = ',',
                         skiprows = [1],
                         header = [0],
                         parse_dates = [0],
                         infer_datetime_format = True,
                         #names=['time','AMICA:OCS','AMICA:CO','AMICA:H2O']
                         )
        df.set_index('time', inplace = True)
        df.sort_index(inplace = True)
        new_names = {
                'CO': 'AMICA_CO',
                'H2O': 'AMICA_H2O',
                }
        df.rename(columns = new_names, inplace = True)
        frame.append(df)    
    output = pd.concat(frame)
    output.sort_index(inplace = True)
    output.rename(columns = {'O3': 'AMICA_O3',}, inplace = True) 
    return output#[list(new_names.values())+ ['OCS', 'AMICA_O3']]

def clean_fairo(res, fn = fn_fairo):
    frame = []
    for filename in fn:
        date = pd.to_datetime(re.findall("(\d+)", filename)[-3])
        df = pd.read_csv(
                         filename,
                         delimiter='\t',
                         skiprows=27,
                         header=0
                         )
        df['time'] = df['UTC_seconds'].apply(lambda x: datetime.timedelta(seconds = x) + date)
        df.drop(columns = ['UTC_seconds'], inplace = True) 
        df.set_index('time', inplace = True)
        df.replace(-9999, np.nan, inplace=True) 
        df = df.resample(f'{res}S').mean()
        df.rename(columns = {'Ozone[ppb]': 'FAIRO_O3'}, inplace = True)
        frame.append(df)
    output = pd.concat(frame)
    output.sort_index(inplace = True)
    
    return output

def clean_fish(res, fn = fn_fish):
    frame=[]
    for filename in fn:
        date=pd.to_datetime(re.findall("(\d+)", filename)[-2])
        df = pd.read_csv(
                         filename,
                         skiprows=19,
                         delimiter=' ',
                         names=['UTC_seconds','H2O_tot','H2O_tot_err']
                         )
        df['time']=df['UTC_seconds'].apply(lambda x: datetime.timedelta(seconds=x)+date )
                         #,parse_dates=[0])
        df.set_index('time', inplace = True) 
        df = df.resample(f'{res}S').mean()
        df.rename(columns = {'H2O_tot': 'FISH_H2O'}, inplace = True)
        frame.append(df)
    output = pd.concat(frame) 
    output.sort_index(inplace = True)
    return output[['FISH_H2O']]

def clean_umaq(res, fn = fn_umaq):
    frame=[]
    for filename in fn:
        date=pd.to_datetime(re.findall("(\d+)", filename)[-5])
        df = pd.read_csv(
                         filename,
                         skiprows=36,
                         delimiter=' ',
                         header=0,
                         )
        df['time']=df['UTC_seconds'].apply(lambda x: datetime.timedelta(seconds=x)+date )
                         #,parse_dates=[0])
        df.drop(columns = ['UTC_seconds'], inplace = True) 
        df.set_index('time', inplace = True) 
        df.replace(99999.00, np.nan, inplace = True) 
        df = df.resample(f'{res}S').mean()
        df.rename(columns = {'UMAQS_CH4_ppbv': 'UMAQS_CH4',
                             'UMAQS_N2O_ppbv': 'UMAQS_N2O',
                             'UMAQS_CO2_ppmv': 'UMAQS_CO2',
                             'UMAQS_CO_ppbv': 'UMAQS_CO',
                             }, inplace = True)        
        frame.append(df)
    output = pd.concat(frame) 
    output.sort_index(inplace = True)
    return output

def clean_aeneas(res, fn = fn_aeneas):
    frame=[]
    for filename in fn:
        date=pd.to_datetime(''.join(re.findall("(\d+)", filename)[-3:-1]))
        df = pd.read_csv(
                         filename,
                         skiprows=20,
                         delimiter=' ',
                         names=['UTC_seconds', 'AENEAS_NO', 'AENEAS_NOy'],
                         )
        df['time']=df['UTC_seconds'].apply(lambda x: datetime.timedelta(seconds=x)+date )
                         #,parse_dates=[0])
        df.drop(columns = ['UTC_seconds'], inplace = True) 
        df.set_index('time', inplace = True) 
        df.replace(-9999, np.nan, inplace = True) 
        df = df.resample(f'{res}S').mean()
        frame.append(df)
    output = pd.concat(frame)  
    output.sort_index(inplace = True)
    return output

#####################################CLaMS#####################################
def clean_Met_V2(res, fn = fn_Met_V2):
    frame=[]
    for filename in fn:
        df = xr.open_dataset(filename).to_dataframe()
        df = df.resample(f'{res}S').mean()
        # df=df.rename(columns={"TIME": "time"})
        # df=df.set_index('time')
        frame.append(df)
    output = pd.concat(frame)
    output.sort_index(inplace = True)
    return output[['PV']]

def clean_agespec_HN2(res, fn = fn_agespec_HN2):
    frame=[]
    for filename in fn:
        df = xr.open_dataset(filename).to_dataframe()
        df = df.resample(f'{res}S').mean()
        frame.append(df)
    output = pd.concat(frame)
    output.sort_index(inplace = True)
    new_names = {
        'AGESPEC_P50': 'P50',
        'AGEFRAC_L06': 'MF_06',
        'AGEFRAC_G24': 'MF_24',
        }
    output.rename(columns = new_names, inplace = True)
    output['MF_06'] = output['MF_06'] * 100
    output['MF_24'] = output['MF_24'] * 100
    return output[list(new_names.values()) + ['AGE']]

def clean_sfctracer_F02(res, fn = fn_sfctracer_F02):
    frame=[]
    for filename in fn:
        df = xr.open_dataset(filename).to_dataframe()
        df = df.resample(f'{res}S').mean()
        frame.append(df)
    output = pd.concat(frame)
    output.sort_index(inplace=True)
    output.drop(columns=['LAT',
                         'LON',
                         'NOONLAT',
                         'NOONLON',
                         'NOONTHETA',
                         'NOONZETA',
                         'PRESS',
                         'THETA',
                         'ZETA',
                         'P0',
                         'P1',
                         'P2',
                         'P3',
                         'P4',
                         'P5',
                         'P6',
                         'P7',
                         'P8',
                            ], inplace=True)
    return output

###############################################################################
def merge(df1, df2, res):
    return pd.merge_asof(df1, df2,
                     left_index=True, 
                     right_index=True,
                     tolerance = pd.Timedelta(f'{res}S'),
                     direction = 'nearest')

def read(strat=0, res=30, LAT = '*', ALT = '*', local=0, transfer=0, PV=[2, 4]):
    """
    read all relevant data from SouthTRAC 
    Parameters:
    strat (boolen): True: only shows stratopheric data. False: all data
    res (int): time resolution in seconds. Default: 30
    LAT ([lower_bound, higher_bound]): latitude range
    ALT (int or float): lower bound of the latitude range
    Returns:
    pd.dataframe
    """
    df = merge(clean_bahamas(res), clean_amica(res), res)
    df = merge(df, clean_fairo(res), res)
    df = merge(df, clean_fish(res), res)
    df = merge(df, clean_umaq(res), res)
    df = merge(df, clean_aeneas(res), res)
    df = merge(df, clean_Met_V2(res), res)
    df = merge(df, clean_agespec_HN2(res), res)
    df = merge(df, clean_sfctracer_F02(res), res)
    
    if strat:
        print (f'abs(PV)>{PV[0]} or theta>380K only')
        df = df[(abs(df['PV'])>PV[0]) | (df['THETA']>380)]
    
    if local:
        print ('local flights only')
        df = df[~df['flight'].isin(['2019-09-08','2019-09-09',
                           '2019-10-07','2019-10-06','2019-10-09',
                           '2019-11-02','2019-11-04','2019-11-06'])]    
                
    if transfer:
        print ('trasnfer flights only')
        df = df[df['flight'].isin(['2019-09-08','2019-09-09',
                           '2019-10-07','2019-10-06','2019-10-09',
                           '2019-11-02','2019-11-04','2019-11-06'])]   
    
    if isinstance(LAT, list):
        df = df[(df['LAT'] > LAT[0]) & (df['LAT'] <= LAT[1])]
        print(f'latitude between {LAT[0]} to {LAT[1]} only')
    else:
        print ('Attention: ALL latitude selected')
    
    df['ALT'] = df['ALT']/1000
    if isinstance(ALT, list):
        df = df[(df['ALT'] > LAT[0]) & (df['ALT'] <= LAT[1])]
        print(f'altitude between {ALT[0]} to {ALT[1]} only')
    elif type(ALT) == int or type(ALT) == float:
        df = df[df['ALT'] > ALT]
        print(f'altitude>{ALT} only')
    else:
        print ('Attention: ALL altitude selected')
    
    df.loc[:, 'air'] = 0
    strat_filter = (abs(df['PV'])>PV[0]) | (df['THETA']>380)
    real_strat_filter = abs(df['PV']) >= PV[1]    
    df.loc[strat_filter, 'air'] = 1 
    df.loc[real_strat_filter, 'air'] = 2 
    
    return df

if __name__ == "__main__":
    #test the module
    res = int (input('time resolution in seconds:\n'))
    df = read(res = res, strat=0)
    print(df.head())
