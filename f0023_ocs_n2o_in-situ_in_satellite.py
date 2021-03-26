import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import rcParams, cycler
import matplotlib.cm as cm
import datetime
import glob2
import xarray as xr
import pandas as pd
import itertools
import re
import southtrac
from matplotlib import gridspec
from scipy import stats
plt.close("all")
pd.options.display.max_columns = None
pd.options.display.max_rows = None


dircInput2 = 'C:/data/ACE-FTS_L2_v4.1_NETCDF/'
dircInput3 = 'C:/data/ENVISAT_MIPAS_by_year/'

lat_max = -53.7#-32.7
lat_min = -70.3
lon_max = -32.6
lon_min = -84

def clean(df,column):
    df.loc[df[column] == -999, column] = np.nan
    df.loc[df[column] == -888, column] = np.nan
    #df.loc[df[column] > 1e-9, column] = np.nan
    df.loc[df[column+'_error']==-888, column] = np.nan
    #df = df.dropna(subset=[column])
    return df

def ace(species,startyear,endyear='',startmonth='',endmonth='',
        lat='',lon='',alt=[10,14.5]):
    filename='ACEFTS_L2_v4p1_'+species+'.nc'
    df = xr.open_dataset(dircInput2+filename)
    if endyear:
        df =df.where((df.year>=startyear) & (df.year<=endyear)) 
    else:
        df =df.where(df.year==startyear) 
    if startmonth and endmonth:
        df =df.where((df.month>=startmonth) & (df.month<=endmonth))
    else:
        print('annual mean')
    #df=df.where(df[species]>-999)
    df['ALT'] = df['altitude']
    df['LAT'] = df['latitude']
    df['LON'] = df['longitude']
    df = df[['year','ALT','LAT','LON',species,species+'_error']]
    df = df.to_dataframe()
    df=df[(df['ALT']>=alt[0]) & (df['ALT']<=alt[1])]
    if lon:
        df = df[(df['LON']>lon[0]) & (df['LON']<lon[1])]
    if lat:
        df = df[(df['LAT']>lat[0]) & (df['LAT']<lat[1])]
    return clean(df,species)

def eqvltN2O (df,baseyear,rate):
    df['eN2O']=df['N2O']+(-baseyear+df['year'])*rate
    return df
    
def process(**kwargs):
    N2O=ace(species='N2O',
          **kwargs)
    OCS = ace(species='OCS',
          **kwargs)
    df = pd.concat([OCS['OCS'],OCS['year'], OCS['LAT'],N2O['N2O']], axis=1, sort=False) \
    .dropna(subset=['OCS']).dropna(subset=['N2O']) \
        .reset_index('altitude') \
            #.drop(['index'],axis=1)
    df['OCS'] = df['OCS'].apply(lambda x: x*1e12) 
    df['N2O'] = df['N2O'].apply(lambda x: x*1e9)
    df = eqvltN2O(df,2019,0.9)
    return df
    
def group (df,y,ymin,ymax,yres,x):
    grange = np.arange(ymin,ymax+yres,yres)      
    output = df[x].groupby([pd.cut(df[y], grange)]).describe().reset_index()
    new_column=y+'_m'
    output[new_column]=output.apply(lambda x1: x1[y].left+yres/2,axis=1)
    return output

#ace_old =process(startyear=2016,endyear=2018,alt=[10,30]) #,lat=[-90,90],lon=[-180,180]

#%%
ace_new =process(startyear=2016,endyear=2019,alt=[8.5,30.5]) #,lat=[-90,90],lon=[-180,180]

# amica 
df=southtrac.read('30S',lat_max,lat_min,10000) \
    .dropna(subset=['UMAQS_N2O_ppbv'])
df_gm=group (df=df,
                  y='UMAQS_N2O_ppbv', ymin=210, ymax=330, yres=10,
                  x='AMICA_OCS')[['UMAQS_.++N2O_ppbv_m','mean','std']]

#plotting
fig = plt.figure(figsize=(10,50))
spec = gridspec.GridSpec(nrows=2, ncols=2,height_ratios=[15,1],width_ratios=[1,1])#,height_ratios=[15,1]) width_ratios=[9,1]

ax = fig.add_subplot(spec[0,0])
ax.errorbar(df_gm['mean'], df_gm['UMAQS_N2O_ppbv_m'], xerr=df_gm['std'],
            fmt='o',color='black',
            capsize=4,ecolor='gray',alpha=0.6,
            label='SouthTRAC')

main=ax.scatter(ace_new['OCS'],ace_new['eN2O'],c=ace_new['LAT'],#ALT LAT
                cmap='rainbow',#plt.cm.get_cmap('rainbow', 4),#'rainbow',plt.cm.get_cmap('cubehelix', 6)
                s=1)
#main.set_clim(2016,2019)

ax.set_xlabel('OCS')
ax.set_ylabel('N2O')
ax.set_xlim(-50,600)
ax.set_ylim(0,350)
plt.title('0023')

ax = fig.add_subplot(spec[1,0])
cbar=plt.colorbar(main, cax=ax,orientation='horizontal')#,ticks=[2016,2017,2018,2019])
cbar.set_label('latitude') 

####################################################################################
ax = fig.add_subplot(spec[0,1])
ax.errorbar(df_gm['mean'], df_gm['UMAQS_N2O_ppbv_m'], xerr=df_gm['std'],
            fmt='o',color='black',
            capsize=4,ecolor='gray',alpha=0.6,
            label='SouthTRAC')

main=ax.scatter(ace_new['OCS'],ace_new['N2O'],c=ace_new['LAT'],#ALT LAT
                cmap='rainbow',#plt.cm.get_cmap('rainbow', 4),#'rainbow',plt.cm.get_cmap('cubehelix', 6)
                s=0.5)
#main.set_clim(2016,2019)

ax.set_xlabel('OCS')
ax.set_ylabel('N2O')
ax.set_xlim(-50,600)
ax.set_ylim(0,350)

ax = fig.add_subplot(spec[1,1])
cbar=plt.colorbar(main, cax=ax,orientation='horizontal')#,ticks=[2016,2017,2018,2019])
cbar.set_label('altitude') 


# def linear_regression(x):
#   return slope * x + intercept

# slope, intercept, r, p, std_err = stats.linregress(df_gm['mean'], df_gm['UMAQS_N2O_ppbv_m'])
# mymodel = df_gm['mean'].apply(linear_regression)
# ax.plot(df_gm['mean'], mymodel,label='slope= {}, r= {}'.format(slope,r))

# #plot 
# ax = fig.add_subplot(spec[1,1])
# main=ax.scatter(ace_new['OCS'],ace_new['N2O'],c=ace_new['year'],#ALT LAT
#                 cmap=plt.cm.get_cmap('rainbow', 4),#'rainbow',plt.cm.get_cmap('cubehelix', 6)
#                 s=3)
# main.set_clim(2016,2019)

# #plot 2019 only
# ax.scatter(ace_old['OCS'],ace_old['N2O'],c=ace_old['year'],#'grey',#ALT LAT
#                 cmap=plt.cm.get_cmap('rainbow', 4),
#                 s=0.5)
# ax.set_xlabel('OCS')
# ax.set_ylabel('N2O')
# ax.set_xlim(0,600)
# ax.set_ylim(175,350)


# ax = fig.add_subplot(spec[1,1])
# cbar=plt.colorbar(main, cax=ax,orientation='horizontal',ticks=[2016,2017,2018,2019])
# cbar.set_label('year') 

# #(-75,-35) (6,15) (-71,-32)

# # slope, intercept, r, p, std_err = stats.linregress(ace_new['OCS'], ace_new['N2O'])
# # mymodel = ace_new['OCS'].apply(linear_regression)
# # ax.plot(ace_new['OCS'], mymodel,label='slope= {}, r= {}'.format(slope,r))


#%%

#function for getting data in shapes to plot
def get_mean(tbp,species,lat_res):
    lat_range = np.arange(-90,90+lat_res,lat_res) 
    return tbp[species].groupby([pd.cut(tbp.LAT, lat_range),tbp.altitude]).mean()

#functions for plotting
def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, 
name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave 
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax) 
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin)) 
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave 
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin)) 
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False), 
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plot(df=None,ax=None,
         xmin=-90, xmax=90, ymin=10, ymax=30,
         cmin=0,cmax=1,cmap='rainbow',
         **kwargs):
    df = df.reset_index()
    #Axises
    x=pd.unique([i.mid for i in df['LAT']])
    y=pd.unique(df['altitude'])
    x, y = np.meshgrid(x,y) 
    z=df.pivot(index ='altitude', columns ='LAT').to_numpy()#,values='mean'
    #Plotting
    if ax is None:
        ax = plt.gca()   
    main=ax.pcolormesh(x,y,z,vmin=cmin, vmax=cmax,cmap=cmap,shading='nearest')
    ax.set(
           xlim=(xmin,xmax), ylim=(ymin,ymax),
           **kwargs
           )
    #ticks and grid
    ax.set_xticks(np.arange(xmin,xmax+1,5) , minor=True)
    ax.set_xticks(np.arange(xmin,xmax+1,10)) 
    ax.set_yticks(np.arange(ymin,ymax+1,1) , minor=True)
    ax.set_yticks(np.arange(ymin,ymax+1,10)) 
    ax.grid(which='both', alpha=0.2, linestyle='--')
    ax.grid(which='minor', alpha=0.2, linestyle='--')
    return(main)

df_ace =process(startyear=2016,endyear=2019,
                startmonth=9,endmonth=11,
                alt=[8.5,30.5])

#get data into shapes to plot
OCS, N2O = get_mean(df_ace, 'OCS', 5), get_mean(df_ace, 'N2O', 5)    

#lay-out
fig= plt.figure(figsize=(50,10))
spec = gridspec.GridSpec(ncols=3, nrows=2, height_ratios=[15,1], width_ratios=[1, 1, 1])

#OCS portion left
ax1 = fig.add_subplot(spec[0,0])
plot(OCS/OCS.max(),ax1, ylabel='altitude/ km',title='f0023 OCS')

#N2O portion left
ax2 = fig.add_subplot(spec[0,1])
main=plot(N2O/N2O.max(),ax2, xlabel='latitude',title='N2O')

#color bar for % of the highest level ax1 or ax2
ax3 = fig.add_subplot(spec[1,0:2])
cbar=plt.colorbar(main, cax=ax3, orientation='horizontal')
cbar.set_label('% of the highest level')

#OCS : N2O 
ax4 = fig.add_subplot(spec[0,2])
cmap=matplotlib.cm.Spectral.reversed()
cmap=remappedColorMap(cmap,start=0, midpoint=2/3, stop=1.0)
main=plot((OCS/OCS.max())/(N2O/N2O.max()),ax4,
     cmin=0, cmax=1.5, cmap=cmap, title='OCS/N2O')

#color bar for the ratio ax4
ax5 = fig.add_subplot(spec[1,2:])
cbar=plt.colorbar(main, cax=ax5,orientation='horizontal')
cbar.set_label('OCS/ N2O') 
#cbar.set_ticks(np.arange(-50,cmax+1,50))
