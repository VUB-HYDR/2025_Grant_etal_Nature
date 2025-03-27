# ---------------------------------------------------------------
# Functions to compute emergence of exposure from noise
# ----------------------------------------------------------------

#               
#%%  ----------------------------------------------------------------
# IMPORT AND PATH 
# ----------------------------------------------------------------

import os
import requests
from zipfile import ZipFile
import io
import xarray as xr
import pickle as pk
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe
import mapclassify as mc
from copy import deepcopy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import regionmask as rm
import geopandas as gpd
from scipy import interpolate
import cartopy.crs as ccrs
import seaborn as sns
import cartopy as cr
import cartopy.feature as feature
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

# %% ----------------------------------------------------------------
# Conceptual plot for emergence in one location,   

def plot_conceptual(
    da_cohort_size,
    countries_mask,
    countries_regions,
    d_isimip_meta,
    flags,
    df_life_expectancy_5,
):
    # get data
    #
    cntry='Belgium'
    city_name='Brussels'
    # cntry='Switzerland'
    # city_name='Zurich'
    # concept_bys = np.arange(1960,2021,30)
    concept_bys = np.arange(1960,2021,1)
    print(cntry)
    da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
    da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages') # cohort relative sizes in sample country
    da_cntry = xr.DataArray(
        np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
        dims=countries_mask.dims,
        coords=countries_mask.coords,
    )
    da_cntry = da_cntry.where(da_cntry,drop=True)
    # weights for latitude (probably won't use but will use population instead)
    lat_weights = np.cos(np.deg2rad(da_cntry.lat))
    lat_weights.name = "weights"   
    # brussels coords  
    city_lat = 50.8476
    city_lon = 4.3572  
    # zurich coords
    # 47.3769° N, 8.5417° E
    # city_lat = 47.3769
    # city_lon = 8.5417

    ds_spatial = xr.Dataset(
        data_vars={
            'cumulative_exposure': (
                ['run','GMT','birth_year','time','lat','lon'],
                np.full(
                    (len(list(d_isimip_meta.keys())),
                    len(GMT_indices_plot),
                    len(concept_bys),
                    len(year_range),
                    len(da_cntry.lat.data),
                    len(da_cntry.lon.data)),
                    fill_value=np.nan,
                ),
            ),
        },
        coords={
            'lat': ('lat', da_cntry.lat.data),
            'lon': ('lon', da_cntry.lon.data),
            'birth_year': ('birth_year', concept_bys),
            'time': ('time', year_range),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_indices_plot)
        }
    )

    # load demography pickle
    with open('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry), 'rb') as f:
        ds_dmg = pk.load(f)                  

    # loop over simulations
    for i in list(d_isimip_meta.keys()): 

        print('simulation {} of {}'.format(i,len(d_isimip_meta)))

        # load AFA data of that run
        with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
            da_AFA = pk.load(f)
            
        # mask to sample country and reduce spatial extent
        da_AFA = da_AFA.where(ds_dmg['country_extent']==1,drop=True)
        
        for step in GMT_indices_plot:
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                
                da_AFA_step = da_AFA.reindex(
                    {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                ).assign_coords({'time':year_range})                     
                                    
                # simple lifetime exposure sum
                da_le = xr.concat(
                    [(da_AFA_step.loc[{'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1)}].cumsum(dim='time') +\
                    da_AFA_step.sel(time=ds_dmg['death_year'].sel(birth_year=by).item()) *\
                    (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item()))\
                    for by in concept_bys],
                    dim='birth_year',
                ).assign_coords({'birth_year':concept_bys})
                
                da_le = da_le.reindex({'time':year_range})
                
                ds_spatial['cumulative_exposure'].loc[{
                    'run':i,
                    'GMT':step,
                    'birth_year':concept_bys,
                    'time':year_range,
                    'lat':ds_dmg['country_extent'].lat.data,
                    'lon':ds_dmg['country_extent'].lon.data,
                }] = da_le.loc[{
                    'birth_year':concept_bys,
                    'time':year_range,
                    'lat':ds_dmg['country_extent'].lat.data,
                    'lon':ds_dmg['country_extent'].lon.data,
                }]

    # mean for brussels            
    da_test_city = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').mean(dim='run')
    da_test_city = da_test_city.rolling(time=5,min_periods=5).mean()

    # standard deviation for brussels
    da_test_city_std = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').std(dim='run')
    da_test_city_std = da_test_city_std.rolling(time=5,min_periods=5).mean()

    # fill in 1st 4 years with 1s
    # first for mean
    for by in da_test_city.birth_year.data:
        for step in GMT_indices_plot:
            da_test_city.loc[{'birth_year':by,'GMT':step,'time':np.arange(by,by+5)}] = da_test_city.loc[{'birth_year':by,'GMT':step}].min(dim='time')
    # then for std        
    for by in da_test_city_std.birth_year.data:
        for step in GMT_indices_plot:
            da_test_city_std.loc[{'birth_year':by,'GMT':step,'time':np.arange(by,by+5)}] = da_test_city_std.loc[{'birth_year':by,'GMT':step}].min(dim='time')        
                
    # load PIC pickles
    with open('./data/{}/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],cntry), 'rb') as f:
        ds_pic = pk.load(f)   
    with open('./data/{}/{}/{}/gridscale_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'rb') as f:
        ds_pic_qntl = pk.load(f)

    # plotting city lat/lon pixel doesn't give smooth kde
    df_pic_city = ds_pic['lifetime_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').to_dataframe().drop(columns=['lat','lon',])         
    da_pic_city_9999 = ds_pic_qntl['99.99'].sel({'lat':city_lat,'lon':city_lon},method='nearest')  

    # concept figure
    # ------------------------------------------------------------------   
    
    # plot building
    from mpl_toolkits.axes_grid1 import inset_locator as inset
    plt.rcParams['patch.linewidth'] = 0.1
    plt.rcParams['patch.edgecolor'] = 'k'
    colors = dict(zip(GMT_indices_plot,['steelblue','darkgoldenrod','darkred']))
    x=5
    y=1
    l = 0
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }

    # ------------------------------------------------------------------   
    # 1960 time series
    f,ax = plt.subplots(
        figsize=(x,y)
    )
    for step in GMT_indices_plot:
        da_test_city.loc[{'birth_year':1960,'GMT':step}].plot.line(
            ax=ax,
            color=colors[step],
            linewidth=1,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':1960,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax,
            color=colors[step],
            linewidth=3,
            zorder=4,
        )
             
    end_year=1960+np.floor(df_life_expectancy_5.loc[1960,cntry])
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_xticks(np.arange(1960,2031,10))
    ax.set_xticklabels([1960,None,1980,None,2000,None,2020,None])
    ax.set_yticks([0,5])
    ax.set_yticklabels([None,5])     
    ax.annotate(
        'Born in 1960',
        (1965,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )    
    ax.set_title(None)
    ax.annotate(
        letters[l],
        (1960,ax.get_ylim()[-1]+2),
        xycoords=ax.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    )    
    l+=1      
      
    ax.set_xlim(
        1960,
        end_year,
    )
    ax.set_ylim(
        0,
        da_pic_city_9999+1,
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.tick_params(colors='gray')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    ax.hlines(
        y=da_pic_city_9999, 
        xmin=1960, 
        xmax=da_test_city.loc[{'birth_year':1960}].time.max()+10, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )

    # 1960 pdf
    ax_pdf_l = end_year+5
    ax_pdf_b = -2
    ax_pdf_w = 20
    ax_pdf_h = ax.get_ylim()[-1]+2
    ax_pdf = ax.inset_axes(
        bounds=(ax_pdf_l, ax_pdf_b, ax_pdf_w, ax_pdf_h),
        transform=ax.transData,
    )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        color='lightgrey',
        discrete = True,
        ax=ax_pdf
    )
    ax_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    for step in GMT_indices_plot:
        ax_pdf.hlines(
            y=da_test_city.loc[{'birth_year':1960,'GMT':step}].max(), 
            xmin=0, 
            xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
            colors=colors[step], 
            linewidth=1, 
            linestyle='-', 
            label=gmt_legend[step], 
            zorder=2
        )
    ax_pdf.spines['right'].set_visible(False)
    ax_pdf.spines['top'].set_visible(False)      
    ax_pdf.set_ylabel(None)
    ax_pdf.set_xlabel(None)
    ax_pdf.set_ylim(-2,ax.get_ylim()[-1])
    ax_pdf.tick_params(colors='gray')
    ax_pdf.spines['left'].set_color('gray')
    ax_pdf.spines['bottom'].set_color('gray')
    ax_pdf.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10,
    )
    l+=1
        
    # ------------------------------------------------------------------       
    # 1990 time series
    ax2_l = 1990
    ax2_b = da_pic_city_9999 *2
    ax2_w = np.floor(df_life_expectancy_5.loc[1990,cntry])
    ax2_h = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())
    ax2 = ax.inset_axes(
        bounds=(ax2_l, ax2_b, ax2_w, ax2_h),
        transform=ax.transData,
    )

    for step in GMT_indices_plot:
        da_test_city.loc[{'birth_year':1990,'GMT':step}].plot.line(
            ax=ax2,
            color=colors[step],
            linewidth=1,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax2,
            color=colors[step],
            linewidth=3,
            zorder=4,
        )    
                  
    end_year=1990+np.floor(df_life_expectancy_5.loc[1990,cntry])
    ax2.set_ylabel(None)
    ax2.set_xlabel(None)
    ax2.set_yticks([0,5,10])
    ax2.set_yticklabels([None,5,10])  
    ax2.set_xticks(np.arange(1990,2071,10))      
    ax2.set_xticklabels([None,2000,None,2020,None,2040,None,2060,None])
    ax2.set_xlim(
        1990,
        end_year,
    )
    ax2.set_ylim(
        0,
        np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)  
    ax2.spines['left'].set_position(('data',1990)) 
    ax2.tick_params(colors='gray')
    ax2.spines['left'].set_color('gray')
    ax2.spines['bottom'].set_color('gray')
    
    ax2.annotate(
        'Born in 1990',
        (1995,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )
    ax2.set_title(None)
    ax2.annotate(
        letters[l],
        (1990,ax2.get_ylim()[-1]),
        xycoords=ax2.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    )     
    l+=1           

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':1990,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        if np.any(da.notnull()): # testing for median
            da_t = da.time.where(da == da.min()).dropna(dim='time').item()
            emergences.append(da_t)
    first_emerge = np.min(emergences)

    ax2.hlines(
        y=da_pic_city_9999, 
        xmin=first_emerge, 
        xmax=end_year, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )        

    # 1990 pdf
    ax2_pdf_l = end_year+5
    ax2_pdf_b = -2
    ax2_pdf_w = 20
    ax2_pdf_h = ax2.get_ylim()[-1]+2
    ax2_pdf = ax2.inset_axes(
        bounds=(ax2_pdf_l, ax2_pdf_b, ax2_pdf_w, ax2_pdf_h),
        transform=ax2.transData,
    )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        color='lightgrey',
        discrete = True,
        ax=ax2_pdf
    )
    ax2_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    for step in GMT_indices_plot:
        ax2_pdf.hlines(
            y=da_test_city.loc[{'birth_year':1990,'GMT':step}].max(), 
            xmin=0, 
            xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
            colors=colors[step], 
            linewidth=1, 
            linestyle='-', 
            label=gmt_legend[step], 
            zorder=2
        )
    ax2_pdf.spines['right'].set_visible(False)
    ax2_pdf.spines['top'].set_visible(False)      
    ax2_pdf.set_ylabel(None)
    ax2_pdf.set_xlabel(None)
    ax2_pdf.set_ylim(-2,ax2.get_ylim()[-1])
    ax2_pdf.tick_params(colors='gray')
    ax2_pdf.spines['left'].set_color('gray')
    ax2_pdf.spines['bottom'].set_color('gray')
    ax2_pdf.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10,
    )
    l+=1   
    
    ax2_pdf.annotate(
        'Unprecedented\nlifetime\nexposure\nfor {} people'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':1990,'lat':city_lat,'lon':city_lon},method='nearest').item(),-3)))),
        (1.1,0.3),
        xycoords=ax2_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        # fontweight='bold',
    )             

    # ------------------------------------------------------------------   
    # 2020 time series
    ax3_l = 2020
    ax3_b = np.round(da_test_city.loc[{'birth_year':1990,'GMT':GMT_indices_plot[-1]}].max()) * 1.5
    ax3_w = np.floor(df_life_expectancy_5.loc[2020,cntry])
    ax3_h = np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())
    ax3 = ax2.inset_axes(
        bounds=(ax3_l, ax3_b, ax3_w, ax3_h),
        transform=ax2.transData,
    )
    # plot mean lines
    for step in GMT_indices_plot:
        da_test_city.loc[{'birth_year':2020,'GMT':step}].plot.line(
            ax=ax3,
            color=colors[step],
            linewidth=1,
        )
        # bold line for emergence
        da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        da.plot.line(
            ax=ax3,
            color=colors[step],
            linewidth=3,
            zorder=4,
        )    

    end_year=2020+np.floor(df_life_expectancy_5.loc[2020,cntry])      
    
    ax3.set_ylabel(None)
    ax3.set_xlabel(None)
    ax3.set_yticks([0,5,10,15,20,25])
    ax3.set_yticklabels([None,5,10,15,20,25])   
    ax3.set_xticks(np.arange(2020,2101,10))      
    ax3.set_xticklabels([2020,None,2040,None,2060,None,2080,None,2100])
    ax3.set_xlim(
        2020,
        end_year,
    )
    ax3.set_ylim(
        0,
        np.round(da_test_city.loc[{'birth_year':2020,'GMT':GMT_indices_plot[-1]}].max())+1,
    )
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)  
    ax3.spines['left'].set_position(('data',2020))  
    ax3.tick_params(colors='gray')
    ax3.spines['left'].set_color('gray')
    ax3.spines['bottom'].set_color('gray')

    # get time of first line to cross PIC thresh
    emergences = []
    for step in GMT_indices_plot:
        da = da_test_city.loc[{'birth_year':2020,'GMT':step}]
        da = da.where(da>da_pic_city_9999)
        if np.any(da.notnull()):
            da_t = da.time.where(da == da.min()).dropna(dim='time').item()
            emergences.append(da_t)
    first_emerge = np.min(emergences)

    ax3.hlines(
        y=da_pic_city_9999, 
        xmin=first_emerge, 
        xmax=end_year, 
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    ax3.annotate(
        'Born in 2020',
        (2025,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        fontweight='bold',
        rotation='horizontal',
        color='gray',
    )
    ax3.set_title(None)
    ax3.annotate(
        letters[l],
        (2020,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=10,
        rotation='horizontal',
        color='k',
        fontweight='bold',
    ) 
    l+=1      

    # 2020 pdf
    ax3_pdf_l = end_year+5
    ax3_pdf_b = -2
    ax3_pdf_w = 20
    ax3_pdf_h = ax3.get_ylim()[-1]+2
    ax3_pdf = ax3.inset_axes(
        bounds=(ax3_pdf_l, ax3_pdf_b, ax3_pdf_w, ax3_pdf_h),
        transform=ax3.transData,
    )
    sns.histplot(
        data=df_pic_city.round(),
        y='lifetime_exposure',
        color='lightgrey',
        discrete = True,
        ax=ax3_pdf
    )
    ax3_pdf.hlines(
        y=da_pic_city_9999, 
        xmin=0, 
        xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
        colors='grey', 
        linewidth=1, 
        linestyle='--', 
        label='99.99%', 
        zorder=1
    )
    for step in GMT_indices_plot:
        ax3_pdf.hlines(
            y=da_test_city.loc[{'birth_year':2020,'GMT':step}].max(), 
            xmin=0, 
            xmax=df_pic_city['lifetime_exposure'][df_pic_city['lifetime_exposure']==0].count(),
            colors=colors[step], 
            linewidth=1, 
            linestyle='-', 
            label=gmt_legend[step], 
            zorder=2
        )
    ax3_pdf.spines['right'].set_visible(False)
    ax3_pdf.spines['top'].set_visible(False)      
    ax3_pdf.set_ylabel(None)
    ax3_pdf.set_xlabel(None)
    ax3_pdf.set_ylim(-2,ax3.get_ylim()[-1])
    ax3_pdf.tick_params(colors='gray')
    ax3_pdf.spines['left'].set_color('gray')
    ax3_pdf.spines['bottom'].set_color('gray')
    ax3_pdf.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10,
    )
    l+=1  
    
    ax3_pdf.annotate(
        'Unprecedented\nlifetime\nexposure\nfor {} people'.format(str(int(np.round(ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item(),-3)))),
        (1.1,0.6),
        xycoords=ax3_pdf.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        # fontweight='bold',
    )                   

    # City name
    ax3.annotate(
        '{}, {}'.format(city_name,cntry),
        (1960,ax3.get_ylim()[-1]),
        xycoords=ax3.transData,
        fontsize=16,
        rotation='horizontal',
        color='gray',
    )

    # axis labels ===================================================================

    # x axis label (time)
    x_i=1950
    y_i=-10
    x_f=2040
    y_f=y_i 
    con = ConnectionPatch(
        xyA=(x_i,y_i),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con)   

    con_arrow_top = ConnectionPatch(
        xyA=(x_f-2,y_f+1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_top)  

    con_arrow_bottom = ConnectionPatch(
        xyA=(x_f-2,y_f-1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_bottom) 
    ax.annotate(
        'Time',
        ((x_i+x_f)/2,y_f+1),
        xycoords=ax.transData,
        fontsize=12,
        color='gray',
    )

    # y axis label (Cumulative heatwave exposure since birth)
    x_i=1950
    y_i=-10
    x_f=x_i
    y_f=y_i + 61
    con = ConnectionPatch(
        xyA=(x_i,y_i),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con)   

    con_arrow_left = ConnectionPatch(
        xyA=(x_f-2,y_f-1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_left)  

    con_arrow_right = ConnectionPatch(
        xyA=(x_f+2,y_f-1),
        xyB=(x_f,y_f),
        coordsA=ax.transData,
        coordsB=ax.transData,
        color='gray',
    )
    ax.add_artist(con_arrow_right) 

    ax.annotate(
        'Cumulative heatwave exposure since birth',
        (x_i-10,(y_i+y_f)/5),
        xycoords=ax.transData,
        fontsize=12,
        rotation='vertical',
        color='gray',
    )

    # legend ===================================================================

    # bbox
    x0 = 1.5
    y0 = 0.5
    xlen = 0.5
    ylen = 0.5

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 10
    legend_lw=2
    
    legendcols = list(colors.values())+['gray']+['lightgrey']
    handles = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[0]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[1]),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=legendcols[2]),
        Line2D([0],[0],linestyle='--',lw=legend_lw,color=legendcols[3]),
        Rectangle((0,0),1,1,color=legendcols[4]),
    ]
    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',
        '99.99% pre-industrial \n lifetime exposure',
        'pre-industrial lifetime \n exposure histogram'
    ]
    ax.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
        loc='upper left',
        ncol=1,
        fontsize=legend_font, 
        labelcolor='gray',
        mode="upper left", 
        borderaxespad=0.,
        frameon=False, 
        columnspacing=0.05, 
    )      

    # population estimates
    ds_dmg['population'].sel({'time':1990,'lat':city_lat,'lon':city_lon},method='nearest').sum(dim='age')

    ds_dmg['by_population_y0'].sel({'birth_year':2020,'lat':city_lat,'lon':city_lon},method='nearest').item()
    
    # getting estimate of all birth years that emerge in 1.5 and 3.5 pathways and how many these cohorts sum to
    valid_bys=da_test_city.birth_year.where(da_test_city.loc[{'GMT':0}].max(dim='time')>da_pic_city_9999)
    y1 = valid_bys.min(dim='birth_year')
    y2 = valid_bys.max(dim='birth_year')
    unprecedented=ds_dmg['by_population_y0'].sel(birth_year=np.arange(y1,y2+1),lat=city_lat,lon=city_lon,method='nearest').sum(dim='birth_year').round().item()    
    print('{} thousand unprecedented born in {} and later under pathway {}'.format(unprecedented/10**3,y1,0))
    
    valid_bys=da_test_city.birth_year.where(da_test_city.loc[{'GMT':20}].max(dim='time')>da_pic_city_9999)
    y1 = valid_bys.min(dim='birth_year')
    y2 = valid_bys.max(dim='birth_year')
    unprecedented=ds_dmg['by_population_y0'].sel(birth_year=np.arange(y1,y2+1),lat=city_lat,lon=city_lon,method='nearest').sum(dim='birth_year').round().item()    
    print('{} thousand unprecedented born in {} and later under pathway {}'.format(unprecedented/10**3,y1,20))        

    f.savefig('./ms_figures/f1_concept_{}_{}.png'.format(flags['version'],cntry),dpi=1000,bbox_inches='tight')
    f.savefig('./ms_figures/f1_concept_{}_{}.pdf'.format(flags['version'],cntry),dpi=1000,bbox_inches='tight')

#%% ----------------------------------------------------------------
# plotting pf heatmaps for grid scale across hazards with and without
# limiting simulations to show ensemble effects on GMT scaling

def plot_heatmaps_allhazards(
    df_GMT_strj,
    da_gs_popdenom,
    flags,
):
    
    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    extremes = [
        'heatwavedarea',     
        'cropfailedarea', 
        'burntarea', 
        'driedarea', 
        'floodedarea', 
        'tropicalcyclonedarea',
    ]
    # extremes_labels = {
    #     'burntarea': '$\mathregular{PF_{Wildfires}}$',
    #     'cropfailedarea': '$\mathregular{PF_{Crop failures}}$',
    #     'driedarea': '$\mathregular{PF_{Droughts}}$',
    #     'floodedarea': '$\mathregular{PF_{Floods}}$',
    #     'heatwavedarea': '$\mathregular{PF_{Heatwaves}}$',
    #     'tropicalcyclonedarea': '$\mathregular{PF_{Tropical cyclones}}$',
    # }    
    extremes_labels = {    
        'heatwavedarea': '$\mathregular{CF_{Heatwaves}}$ [%]',
        'cropfailedarea': '$\mathregular{CF_{Crop failures}}$ [%]',
        'burntarea': '$\mathregular{CF_{Wildfires}}$ [%]',
        'driedarea': '$\mathregular{CF_{Droughts}}$ [%]',
        'floodedarea': '$\mathregular{CF_{Floods}}$ [%]',
        'tropicalcyclonedarea': '$\mathregular{CF_{Tropical cyclones}}$ [%]',
    }      
    unprec_level="unprec_99.99"      

    # labels for GMT ticks
    # GMT_indices_ticks=[6,12,18,24]
    GMT_indices_ticks=[0,5,10,15,20]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    
    levels_hw=np.arange(0,101,10)
    levels_cf=np.arange(0,31,5)
    levels_other=np.arange(0,16,1)
    
    # # --------------------------------------------------------------------
    # # population fractions with simulation limits to avoid dry jumps
    
    # # loop through extremes and concat pop and pop frac
    # list_extrs_pf = []
    # for extr in extremes:
    #     with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
    #         ds_pf_gs_extr = pk.load(file)
    #     with open('./data/{}/{}/isimip_metadata_{}_ar6_new_rm.pkl'.format(flags['version'],extr,extr), 'rb') as file:
    #         d_isimip_meta = pk.load(file)        
    #     sims_per_step = {}
    #     for step in GMT_labels:
    #         sims_per_step[step] = []
    #         print('step {}'.format(step))
    #         for i in list(d_isimip_meta.keys()):
    #             if d_isimip_meta[i]['GMT_strj_valid'][step]:
    #                 sims_per_step[step].append(i)  
    #     if extr != 'cropfailedarea':
    #         p = ds_pf_gs_extr[unprec_level].loc[{
    #             'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
    #             'run':sims_per_step[GMT_labels[-1]]
    #         }].sum(dim='country')
    #     else: # for some reason, cropfailedarea doesn't have 3.5th in earlier v1 pickle run?
    #         p = ds_pf_gs_extr[unprec_level].loc[{
    #             'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
    #             'run':sims_per_step[GMT_labels[-1]]
    #         }].sum(dim='country')            
    #     p = p.where(p!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country') *100
    #     list_extrs_pf.append(p)
        
    # ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})
    
    # # plot
    # mpl.rcParams['xtick.labelcolor'] = 'gray'
    # mpl.rcParams['ytick.labelcolor'] = 'gray'
    # x=14
    # y=7
    # f,axes = plt.subplots(
    #     nrows=2,
    #     ncols=3,
    #     figsize=(x,y),
    # )

    # for ax,extr in zip(axes.flatten(),extremes):
    #     if extr == 'heatwavedarea':
    #         p = ds_pf_gs_extrs.loc[{
    #             'hazard':extr,
    #             'birth_year':np.arange(1960,2021),
    #         }].plot.contourf(
    #             x='birth_year',
    #             y='GMT',
    #             ax=ax,
    #             add_labels=False,
    #             # levels=10,
    #             levels=levels_hw,
    #             cmap='Reds',
    #             cbar_kwargs={'ticks':np.arange(0,101,20)}
    #         ) 
    #     elif extr == 'cropfailedarea':
    #         p = ds_pf_gs_extrs.loc[{
    #             'hazard':extr,
    #             'birth_year':np.arange(1960,2021),
    #         }].plot.contourf(
    #             x='birth_year',
    #             y='GMT',
    #             ax=ax,
    #             add_labels=False,
    #             # levels=10,
    #             levels=levels_cf,
    #             cmap='Reds',
    #             cbar_kwargs={'ticks':np.arange(0,31,5)}
    #         )         
    #     else:
    #         p = ds_pf_gs_extrs.loc[{
    #             'hazard':extr,
    #             'birth_year':np.arange(1960,2021),
    #         }].plot.contourf(
    #             x='birth_year',
    #             y='GMT',
    #             ax=ax,
    #             add_labels=False,
    #             # levels=10,
    #             levels=levels_other,
    #             cmap='Reds',
    #             cbar_kwargs={'ticks':np.arange(0,16,3)}
    #         )  
            
    #     ax.set_yticks(
    #         ticks=GMT_indices_ticks,
    #         labels=gmts2100,
    #         color='gray',
    #     )
    #     ax.set_xticks(
    #         ticks=np.arange(1960,2025,10),
    #         color='gray',
    #     )                  
        
    # # ax stuff
    # l=0
    # for n,ax in enumerate(axes.flatten()):
    #     ax.set_title(
    #         extremes_labels[extremes[n]],
    #         loc='center',
    #         fontweight='bold',
    #         color='gray',
    #         fontsize=12,
    #     )
    #     ax.set_title(
    #         letters[l],
    #         loc='left',
    #         fontweight='bold',
    #         fontsize=10,
    #     )  
    #     l+=1                  
    #     ax.spines['right'].set_color('gray')
    #     ax.spines['top'].set_color('gray')
    #     ax.spines['left'].set_color('gray')
    #     ax.spines['bottom'].set_color('gray')

    #     if not np.isin(n,[0,3]):
    #         ax.yaxis.set_ticklabels([])
    #     if n == 0:
    #         ax.annotate(
    #                 'GMT warming by 2100 [°C]',
    #                 (-.3,-0.6),
    #                 xycoords=ax.transAxes,
    #                 fontsize=12,
    #                 rotation='vertical',
    #                 color='gray',
    #                 # fontweight='bold',        
    #             )            
    #     if n <= 2:
    #         ax.tick_params(labelbottom=False)    
    #     if n >= 3:
    #         ax.set_xlabel('Birth year',fontsize=12,color='gray')         
 
    # f.savefig('./ms_figures/pf_heatmap_combined_simlim_{}.png'.format(flags['version']),dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/pf_heatmap_combined_simlim_{}.eps'.format(flags['version']),format='eps',bbox_inches='tight')
    # plt.show()     
    
    # --------------------------------------------------------------------
    # population fractions with all simulations
    
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        p = ds_pf_gs_extr[unprec_level].loc[{
            'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
        }].sum(dim='country')       
        p = p.where(p!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country') *100
        list_extrs_pf.append(p)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})

    # plot
    mpl.rcParams['xtick.labelcolor'] = 'gray'
    mpl.rcParams['ytick.labelcolor'] = 'gray'
    x=14
    y=7
    f,axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(x,y),
    )
    for ax,extr in zip(axes.flatten(),extremes):
        if extr == 'heatwavedarea':
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_hw,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,101,20)}
            ) 
        elif extr == 'cropfailedarea':
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_cf,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,31,5)}
            )         
        else:
            p = ds_pf_gs_extrs.loc[{
                'hazard':extr,
                'birth_year':np.arange(1960,2021),
            }].plot.contourf(
                x='birth_year',
                y='GMT',
                ax=ax,
                add_labels=False,
                # levels=10,
                levels=levels_other,
                cmap='Reds',
                cbar_kwargs={'ticks':np.arange(0,16,3)}
            )         
        
        ax.set_yticks(
            ticks=GMT_indices_ticks,
            labels=gmts2100,
            color='gray',
        )
        ax.set_xticks(
            ticks=np.arange(1960,2025,10),
            color='gray',
        )    
        
    # ax stuff
    l=0
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(
            extremes_labels[extremes[n]],
            loc='center',
            fontweight='bold',
            color='gray',
            fontsize=12,
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            fontsize=10,
        )  
        l+=1                  
        ax.spines['right'].set_color('gray')
        ax.spines['top'].set_color('gray')
        ax.spines['left'].set_color('gray')
        ax.spines['bottom'].set_color('gray')
        if not np.isin(n,[0,3]):
            ax.yaxis.set_ticklabels([])
        else:
            pass
        if n == 0:
            ax.annotate(
                    'GMT warming by 2100 [°C]',
                    (-.3,-0.6),
                    xycoords=ax.transAxes,
                    fontsize=12,
                    rotation='vertical',
                    color='gray',
                    # fontweight='bold',        
                )            
        if n <= 2:
            ax.tick_params(labelbottom=False)    
        if n >= 3:
            ax.set_xlabel('Birth year',fontsize=12,color='gray')
    
    # f.savefig('./ms_figures/pf_heatmap_combined_allsims_{}.png'.format(flags['version']),dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/pf_heatmap_combined_allsims_{}.pdf'.format(flags['version']),dpi=500,bbox_inches='tight')    
    # f.savefig('./ms_figures/pf_heatmap_combined_allsims_{}.eps'.format(flags['version']),format='eps',bbox_inches='tight')
    plt.show()         

#%% ----------------------------------------------------------------
# plot of locations of emergence

def plot_emergence_union(
    grid_area,
    da_emergence_mean,
):
    
    x=11
    y=8.1
    markersize=10
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = '0'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    
    density=6  # density of hatched lines showing frac of sims with emergence
    sim_frac=0.50  # fraction of sims to show emergence (0.25 for original one)
    gmt = 17 # gmt index to compare multihazard pf
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }  
    colors=[
        mpl.colors.to_rgb('steelblue'),
        mpl.colors.to_rgb('darkgoldenrod'),
        mpl.colors.to_rgb('peru'),
    ]
    cmap_list = mpl.colors.ListedColormap(colors,N=len(colors))

    cmap_reds = plt.cm.get_cmap('Reds')

    colors_union = [
        'white',
        cmap_reds(0.25),
        cmap_reds(0.50),
        cmap_reds(0.75),
    ]
    cmap_list_union = mpl.colors.ListedColormap(colors_union,N=len(colors_union))
    cmap_list_union.set_over(cmap_reds(0.99))
    levels = np.arange(0.5,3.6,1)
    union_levels = np.arange(-0.5,3.6,1)
    norm=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(8,2)
    gs0.update(wspace=0.25,hspace=0.2)
    ax0 = f.add_subplot(gs0[5:8,0:1],projection=ccrs.Robinson())

    # left side for 1960
    # masp per hazard
    gsn0 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,0:1],
        wspace=0,
        hspace=0,
    )
    ax00 = f.add_subplot(gsn0[0],projection=ccrs.Robinson())
    ax10 = f.add_subplot(gsn0[1],projection=ccrs.Robinson())
    ax20 = f.add_subplot(gsn0[2],projection=ccrs.Robinson()) 

    ax01 = f.add_subplot(gsn0[3],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gsn0[4],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gsn0[5],projection=ccrs.Robinson())       

    # colorbars
    pos0 = ax0.get_position()
    # colorbar for emergence
    cax_e = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.075,
        pos0.width*.15,
        pos0.height*0.25
    ])
    # colorbar for union
    cax_u = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.175,
        pos0.width*1.5,
        pos0.height*0.15
    ])

    # right side for 2020
    ax1 = f.add_subplot(gs0[5:8,1:2],projection=ccrs.Robinson()) # map of emergence union
    gsn1 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,1:2],
        wspace=0,
        hspace=0,
    )
    ax02 = f.add_subplot(gsn1[0],projection=ccrs.Robinson())
    ax12 = f.add_subplot(gsn1[1],projection=ccrs.Robinson())
    ax22 = f.add_subplot(gsn1[2],projection=ccrs.Robinson()) 

    ax03 = f.add_subplot(gsn1[3],projection=ccrs.Robinson())
    ax13 = f.add_subplot(gsn1[4],projection=ccrs.Robinson())
    ax23 = f.add_subplot(gsn1[5],projection=ccrs.Robinson())     

    # plot 1960
    i=0
    l=0

    ax00.annotate(
        '1960 birth cohort',
        (0.55,1.3),
        xycoords=ax00.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )          

    i+=1

    # raster of falses to populate based on regions where 25% of projections emerge
    template_1960 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=gmt,birth_year=1960),
        False
    )

    for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p1960 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':1960,
        }]
        ax.contourf(
            p1960.lon.data,
            p1960.lat.data,
            p1960.where(p1960>sim_frac).notnull(),
            levels=[.5,1.5],
            colors='none',
            transform=ccrs.PlateCarree(),
            hatches=[density*'/',density*'/'],
            rasterized=True,
            zorder=10
        )       
        template_1960 = template_1960+p1960.where(p1960>sim_frac).notnull()
        p1960 = xr.where(p1960>0,1,0)
        p1960 = p1960.where(p1960).where(mask.notnull())*3
        p1960.plot(
            ax=ax,
            cmap=cmap_list,
            levels=levels,
            add_colorbar=False,
            add_labels=False,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=5
        )    
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray'
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1    
        i+=1
        
    p_u1960 = template_1960.where(mask.notnull())

    ax0.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax0.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
    p_u1960.plot(
        ax=ax0,
        cmap=cmap_list_union,
        levels=union_levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1                 
    i+=1

    # 2020 birth cohort
    ax02.annotate(
        '2020 birth cohort',
        (0.55,1.3),
        xycoords=ax02.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )       

    # list of hatched booleans for 2020
    template_2020 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=gmt,birth_year=2020),
        False
    )

    for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
        
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        ax.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white',linewidth=0.25))
        p2020 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':2020,
        }]
        ax.contourf(
            p2020.lon.data,
            p2020.lat.data,
            p2020.where(p2020>sim_frac).notnull(),
            levels=[.5,1.5],
            colors='none',
            transform=ccrs.PlateCarree(),
            hatches=[density*'/',density*'/'],
            rasterized=True,
            zorder=10
        )      
        # test = p2020.where(p2020>sim_frac).notnull()
        # ax.contourf(
        #     p2020.lon.data,
        #     p2020.lat.data,
        #     test,
        #     levels=[.5,1.5],
        #     colors='k',
        #     transform=ccrs.PlateCarree(),
        #     rasterized=True,
        #     zorder=10
        # )          
        template_2020 = template_2020+p2020.where(p2020>sim_frac).notnull()           
        p2020 = xr.where(p2020>0,1,0)
        p2020 = p2020.where(p2020).where(mask.notnull())*3
        p2020.plot(
            ax=ax,
            cmap=cmap_list,
            levels=levels,
            add_colorbar=False,
            add_labels=False,
            rasterized=True,
            transform=ccrs.PlateCarree(),
            zorder=5
        )    
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray',
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1          
        i+=1  
        
    p_u2020 = template_2020.where(mask.notnull())
    ax1.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    ax1.add_feature(feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='white'))
    p_u2020.plot(
        ax=ax1,
        cmap=cmap_list_union,
        levels=union_levels,
        add_colorbar=False,
        add_labels=False,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=5
    )
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1  

    # colorbar for emergence per extreme
    colors_emergence_cb = ['white','peru']
    cmap_list_emergence_cb = mpl.colors.ListedColormap(colors_emergence_cb,N=len(colors_emergence_cb))
    levels_emergence_cb = np.arange(-0.5,1.6,1)
    norm_e=mpl.colors.BoundaryNorm(levels_emergence_cb,ncolors=len(levels_emergence_cb)-1)
    cb_e = mpl.colorbar.ColorbarBase(
        ax=cax_e, 
        cmap=cmap_list_emergence_cb,
        # norm=norm,
        orientation='vertical',
        spacing='uniform',
        ticks=[0.25,0.75],
        drawedges=False,
    )
    cb_e.ax.set_yticklabels(['No emergence in ensemble','Emergence in ensemble'])

    cb_e.ax.set_title(
        '(' + r"$\bf{a}$" + '-' + r"$\bf{f}$" + ',' + r"$\bf{h}$" + '-' + r"$\bf{m}$"+')',
        fontsize=10,
    )
    cb_e.ax.xaxis.set_label_position('top')
    cb_e.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=10,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )                 

    # colorbar for union of emergence across extremes
    cb_u = mpl.colorbar.ColorbarBase(
        ax=cax_u, 
        cmap=cmap_list_union,
        norm=norm,
        orientation='horizontal',
        extend='max',
        spacing='uniform',
        ticks=np.arange(0,7).astype('int'),
        drawedges=False,
    )
    cb_u.ax.set_title(
        '(' + r"$\bf{g}$" + ',' + r"$\bf{n}$" + ')',
        y=1.06,
        fontsize=10,
        loc='left',
    )

    cb_u.set_label(
        'Number of emerged extremes',
        fontsize=10,
        labelpad=8,
        color='gray',
    )
    cb_u.ax.xaxis.set_label_position('top')
    cb_u.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   

    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    # eu=da_emergence_union.loc[{'GMT':17,'birth_year':2020}].where(mask.notnull())
    la_frac_eu_gteq3_2020 = xr.where(p_u2020>=3,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100
    la_frac_eu_gteq3_1960 = xr.where(p_u1960>=3,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100
    print('2020 percentage of land area \n with emergence of 3 extremes is {}'.format(la_frac_eu_gteq3_2020.item()))    
    print('1960 percentage of land area \n with emergence of 3 extremes {}'.format(la_frac_eu_gteq3_1960.item()))  
        
    f.savefig('./ms_figures/emergence_union_new_gmt17_50perc.png',dpi=1000,bbox_inches='tight')
    # f.savefig('./ms_figures/emergence_union_new_25perc.pdf',dpi=500,bbox_inches='tight')

# %% ----------------------------------------------------------------
# Combined plot for heatwaves showing box plots of 1.5, 2.5 and 3.5, 
# t series of 2020 by and maps of 2020 by for 1.5, 2.5 and 3.5         

def plot_combined(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
):
        
    x=12
    y=10
    markersize=10
    # cbar stuff
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   
    by=2020

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(10,2)
    # gs0.update(hspace=0.8,wspace=0.8)
    ax00 = f.add_subplot(gs0[0:8,0:1]) # box plots
    # ax10 = f.add_subplot(gs0[2:,0:2]) # tseries for 2020 by
    gs00 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[0:7,1:],
        # top=0.8
    )
    ax01 = f.add_subplot(gs00[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs00[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs00[2],projection=ccrs.Robinson()) 
    pos00 = ax21.get_position()
    cax00 = f.add_axes([
        pos00.x0-0.0775,
        pos00.y0-0.075,
        pos00.width*2.2,
        pos00.height*0.2
    ])

    l = 0 # letter indexing

    # colorbar stuff ------------------------------------------------------------
    cmap_whole = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,101,10)
    norm = mpl.colors.BoundaryNorm(levels,cmap_list_frac.N)   

    # pop frac box plot ----------------------------------------------------------
    GMT_indices_ticks=[6,12,18,24]
    gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    

    # levels = np.arange(0,1.01,0.05)
    levels = np.arange(0,101,5)
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)

    # get data
    df_list_gs = []
    extr='heatwavedarea'
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
        ds_pf_gs_plot = pk.load(file)
    da_p_gs_plot = ds_pf_gs_plot['unprec'].loc[{
        'GMT':GMT_indices_plot,
        'birth_year':sample_birth_years,
    }]
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        print('step {}'.format(step))
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)    

    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
        df_pf_gs_plot_step['hazard'] = extr
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)

    # pf boxplot
    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    p = sns.boxplot(
        data=df_pf_gs_plot[df_pf_gs_plot['hazard']==extr],
        x='birth_year',
        y='pf',
        hue='GMT_label',
        palette=colors,
        showcaps=False,
        showfliers=False,
        boxprops={
            'linewidth':0,
            'alpha':0.5
        },        
        ax=ax00,
    )
    p.legend_.remove()                  
    ax00.spines['right'].set_visible(False)
    ax00.spines['top'].set_visible(False)      
    ax00.tick_params(colors='gray')
    ax00.set_ylim(0,100)
    ax00.spines['left'].set_color('gray')
    ax00.spines['bottom'].set_color('gray')      
    ax00.set_ylabel('$\mathregular{PF_{Heatwaves}}$',color='gray',fontsize=14)
    ax00.set_xlabel('Birth year',color='gray',fontsize=14)       
    ax00.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    l+=1 

    # bbox
    x0 = 0.075
    y0 = 0.7
    xlen = 0.2
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 10
    legend_lw=3.5   

    legendcols = list(colors.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2])
    ]

    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',    
    ]

    ax00.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=1,
        fontsize=legend_font, 
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )            
    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------     

    # gmt_indices_123 = [19,10,0]
    gmt_indices_152535 = [24,15,6]
    map_letters = {24:'d',15:'c',6:'b'}
    da_p_gs_plot = ds_pf_gs['unprec'].loc[{
        'GMT':gmt_indices_152535,
        'birth_year':by,
    }]

    # since wer're looking at country level means across runs, denominator is important and 0s need to be accounted for in non-emergence
    # so we only take sims or runs valid per GMT level and make sure nans are 0
    df_list_gs = []
    for step in gmt_indices_152535:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
        da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    for ax,step in zip((ax01,ax11,ax21),gmt_indices_152535):
        gdf_p['pf']=df_p_gs_plot['pf'][df_p_gs_plot['GMT']==step].values
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        gdf_p.to_crs(robinson).plot(
            ax=ax,
            column='pf',
            cmap=cmap_list_frac,
            norm=norm,
            cax=cax00,
            zorder=2,
        )

        gdf.to_crs(robinson).plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
            zorder=3,
        )
        
        ax.set_title(
            '{} °C'.format(gmt_legend[step]),
            loc='center',
            fontweight='bold',
            fontsize=12,
            color='gray',       
        )
        
        ax.set_title(
            map_letters[step],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    
        l+=1          
        
        # pointers connecting 2020, GMT step pixel in heatmap to map panels ------------------
        if step == gmt_indices_152535[0]:
            x_h=1 
        elif step == gmt_indices_152535[1]:
            x_h=0.95                      
        elif step == gmt_indices_152535[-1]:
            x_h=0.9
        y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
        x_m=0
        y_m=0.5
        con = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m),
            coordsA=ax00.transAxes,
            coordsB=ax.transAxes,
            color='gray'
        )
        ax00.add_artist(con)          
        
    cb = mpl.colorbar.ColorbarBase(
        ax=cax00, 
        cmap=cmap_list_frac,
        norm=norm,
        orientation='horizontal',
        spacing='uniform',
        ticks=ticks,
        drawedges=False,
    )

    cb.set_label(
        # 'Population % living unprecedented exposure to heatwaves',
        # '$PF_HW$',
        '$\mathregular{PF_{Heatwaves}}$ for 2020 birth cohort',
        fontsize=14,
        color='gray'
    )
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)   
    cax00.xaxis.set_label_position('top')                   

    # f.savefig('./ms_figures/combined_plot_{}.png'.format(flags['extr']),dpi=1000,bbox_inches='tight')

    plt.show()            
    
#%% ----------------------------------------------------------------
# plot tseries, pie charts and maps of heatwave PF
# ------------------------------------------------------------------        

def plot_combined_piechart(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
    df_countries,
):
    x=12
    y=7
    markersize=10
    # cbar stuff
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   
    by=2020
    
    plot_var='unprec_99.99'
    gmt_indices_152535 = [20,10,0]
    map_letters = {20:'g',10:'f',0:'e'}

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(10,3)
    # gs0.update(hspace=0.8,wspace=0.8)

    # box plots
    ax0 = f.add_subplot(gs0[0:5,0:2]) 

    # pie charts
    gs10 = gridspec.GridSpecFromSubplotSpec(
        1,
        3, 
        subplot_spec=gs0[7:,0:2],
        # top=0.8
    )
    ax00 = f.add_subplot(gs10[0])
    ax10 = f.add_subplot(gs10[1])
    ax20 = f.add_subplot(gs10[2]) 

    # maps
    gs01 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[0:9,2:],
        # top=0.8
    )
    ax01 = f.add_subplot(gs01[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs01[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs01[2],projection=ccrs.Robinson()) 
    pos00 = ax21.get_position()
    cax00 = f.add_axes([
        pos00.x0-0.05,
        pos00.y0-0.1,
        pos00.width*1.95,
        pos00.height*0.2
    ])

    l = 0 # letter indexing

    # colorbar stuff ------------------------------------------------------------
    cmap_whole = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,101,10)
    norm = mpl.colors.BoundaryNorm(levels,cmap_list_frac.N)   

    # pop frac box plot ----------------------------------------------------------
    # GMT_indices_ticks=[6,12,18,24]
    # GMT_indices_ticks
    # gmts2100 = np.round(df_GMT_strj.loc[2100,GMT_indices_ticks].values,1)    

    # levels = np.arange(0,1.01,0.05)
    levels = np.arange(0,101,5)
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)

    # get data
    df_list_gs = []
    extr='heatwavedarea'
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
        ds_pf_gs_plot = pk.load(file)
    da_p_gs_plot = ds_pf_gs_plot[plot_var].loc[{
        'GMT':GMT_indices_plot,
        'birth_year':sample_birth_years,
    }]
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)    

    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
        df_pf_gs_plot_step['hazard'] = extr
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)

    # pf boxplot
    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    p = sns.boxplot(
        data=df_pf_gs_plot[df_pf_gs_plot['hazard']==extr],
        x='birth_year',
        y='pf',
        hue='GMT_label',
        palette=colors,
        showcaps=False,
        showfliers=False,
        boxprops={
            'linewidth':0,
            'alpha':0.5
        },        
        ax=ax0,
    )
    p.legend_.remove()                  
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)      
    ax0.tick_params(colors='gray')
    ax0.set_ylim(0,100)
    ax0.spines['left'].set_color('gray')
    ax0.spines['bottom'].set_color('gray')      
    # ax0.set_ylabel('$\mathregular{PF_{Heatwaves}}$',color='gray',fontsize=14)
    ax0.set_ylabel('$\mathregular{CF_{Heatwaves}}$ [%]',color='gray',fontsize=14)
    ax0.set_xlabel('Birth year',color='gray',fontsize=14)       
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    l+=1 

    # bbox
    x0 = 0.075
    y0 = 0.7
    xlen = 0.2
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 10
    legend_lw=3.5   

    legendcols = list(colors.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2])
    ]

    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',    
    ]

    ax0.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=1,
        fontsize=legend_font, 
        labelcolor='gray',
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )            
    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways ----------------------------------------------------------     

    # gmt_indices_123 = [19,10,0]
    # gmt_indices_152535 = [24,15,6]
    # map_letters = {24:'g',15:'f',6:'e'}
    da_p_gs_plot = ds_pf_gs[plot_var].loc[{
        'GMT':gmt_indices_152535,
        'birth_year':by,
    }]

    # since wer're looking at country level means across runs, denominator is important and 0s need to be accounted for in non-emergence
    # so we only take sims or runs valid per GMT level and make sure nans are 0
    df_list_gs = []
    for step in gmt_indices_152535:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
        da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    for ax,step in zip((ax01,ax11,ax21),gmt_indices_152535):
        gdf_p['pf']=df_p_gs_plot['pf'][df_p_gs_plot['GMT']==step].values
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        gdf_p.to_crs(robinson).plot(
            ax=ax,
            column='pf',
            cmap=cmap_list_frac,
            norm=norm,
            cax=cax00,
            zorder=2,
            rasterized=True,
        )

        gdf.to_crs(robinson).plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
            zorder=3,
        )
        
        ax.set_title(
            '{} °C'.format(gmt_legend[step]),
            loc='center',
            fontweight='bold',
            fontsize=12,
            color='gray',       
        )
        
        ax.set_title(
            map_letters[step],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    
        # l+=1          
        
        # pointers connecting 2020, GMT step box plot to map panels ------------------
        # if step == gmt_indices_152535[0]:
        #     x_h=1 
        # elif step == gmt_indices_152535[1]:
        #     x_h=0.95                      
        # elif step == gmt_indices_152535[-1]:
        #     x_h=0.9
        # y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
        # x_m=0
        # y_m=0.5
        # con = ConnectionPatch(
        #     xyA=(x_h,y_h),
        #     xyB=(x_m,y_m),
        #     coordsA=ax0.transAxes,
        #     coordsB=ax.transAxes,
        #     color='gray'
        # )
        # ax0.add_artist(con)   
        
        # triangles showing connections between GMT step box plot to map panels ---------------
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        if step == gmt_indices_152535[0]:
            x_h=1 
        elif step == gmt_indices_152535[1]:
            x_h=0.95                      
        elif step == gmt_indices_152535[-1]:
            x_h=0.9
        y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
        x_m=0
        y_m_i=0
        y_m_f=1.0
        con_low = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m_i),
            coordsA=ax0.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        con_hi = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m_f),
            coordsA=ax0.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )   
        con_vert = ConnectionPatch(
            xyA=(x_m,y_m_i),
            xyB=(x_m,y_m_f),
            coordsA=ax.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        
        line_low = con_low.get_path().vertices
        line_hi = con_hi.get_path().vertices
        
        ax0.add_artist(con_low)  
        ax0.add_artist(con_hi) 
        
        tri_coords = np.stack(
            (con_low.get_path().vertices[0],con_low.get_path().vertices[-1],con_hi.get_path().vertices[-1]),
            axis=0,
        )
        triangle = plt.Polygon(tri_coords,ec='lightgray',fc='lightgray',alpha=0.5,zorder=0,clip_on=False)    
        ax0.add_artist(triangle)           
        
        # triangles showing connections between GMT step box plot to map panels ---------------
             
        
    cb = mpl.colorbar.ColorbarBase(
        ax=cax00, 
        cmap=cmap_list_frac,
        norm=norm,
        orientation='horizontal',
        spacing='uniform',
        ticks=ticks,
        drawedges=False,
    )

    cb.set_label(
        '$\mathregular{CF_{Heatwaves}}$ for 2020 birth cohort [%]',
        fontsize=14,
        color='gray'
    )
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)   
    cax00.xaxis.set_label_position('top')   

    # gmt_indices_sample = [24,15,6]
    gmt_indices_sample=gmt_indices_152535
    gmt_legend={
        gmt_indices_sample[0]:'1.5',
        gmt_indices_sample[1]:'2.5',
        gmt_indices_sample[2]:'3.5',
    }
    # colors = dict(zip([6,15,24],['steelblue','darkgoldenrod','darkred']))
    colors = dict(zip([0,10,20],['steelblue','darkgoldenrod','darkred']))

    by_sample = [1960,1990,2020]
    incomegroups = df_countries['incomegroup'].unique()
    income_countries = {}
    for category in incomegroups:
        income_countries[category] = list(df_countries.index[df_countries['incomegroup']==category])
    ig_dict = {
        'Low income':'LI',
        'Lower middle income': 'LMI',
        'Upper middle income': 'UMI',
        'High income': 'HI',
    }


    income_unprec = {}
    da_unprec = ds_pf_gs[plot_var]
    for category in list(income_countries.keys()):
        income_unprec[category] = da_unprec.loc[{
            'country':income_countries[category],
            'GMT':gmt_indices_sample,
        }]

        
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)
            
    pi_totals = {}
    pi_ratios = {}
    for by in by_sample:
        
        # populate each birth year with totals per income group
        pi_totals[by] = [da_gs_popdenom.loc[{
            'country':income_countries[category],
            'birth_year':by,
        }].sum(dim='country').item() for category in list(income_countries.keys())]
        
        
        pi_ratios[by] = {}
        for category in list(income_countries.keys()):
            pi_ratios[by][category] = {}
            for step in gmt_indices_sample:
                unprec = income_unprec[category].loc[{
                    'GMT':step,
                    'run':sims_per_step[step],
                    'birth_year':by
                }].sum(dim='country').mean(dim='run').item()
                pi_ratios[by][category][step] = unprec / da_gs_popdenom.loc[{
                    'country':income_countries[category],
                    'birth_year':by,
                }].sum(dim='country').item()
        

    for by,ax in zip(by_sample,[ax00,ax10,ax20]):
        for i,category in enumerate(list(income_countries.keys())):
            order = np.argsort(list(pi_ratios[by][category].values())) # indices that would sort pf across gmt steps
            order = order[::-1] # want decreasing pf order, so we reverse the indices
            ordered_gmts = [gmt_indices_sample[o] for o in order]
            for step in ordered_gmts:
                colors_list =['None']*4
                colors_list[i] = 'white'
                # first paint white where we want actual color (semi transparent, white makes it look good) per category with nothing in other categories
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }     
                )
                # then paint actual color and radius
                colors_list[i] = colors[step]
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }
                )                     
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    
        ax.set_title(
            by,
            loc='center',
            fontweight='bold',
            fontsize=12,
            color='gray',       
        )        
        l+=1             
        percents = ['25%','50%','75%','100%']
        for i,r in enumerate(np.arange(0.25,1.01,0.25)):
            if r < 1:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    }
                )      
            else:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    labels=list(ig_dict.values()),
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    },
                    textprops={
                        'color':'0.5'
                    }
                )        
            
            if by == 1960:
                ax.annotate(
                    percents[i],
                    xy=(0,r+0.05),
                    color='gray',
                    ha='center',
                    fontsize=6
                )
                
        # triangles showing connections between box plot years on x axis to pie chart years ---------------
        if ax == ax00:
            x_b=0.075 
            x_p_i=x_b - x_b/2
            x_p_f=x_b + x_b*2
        elif ax == ax10:
            x_b=0.5   
            x_p_i=0.425
            x_p_f=0.575                         
        elif ax == ax20:
            x_b=0.925
            x_p_i=x_b - 0.075*2
            x_p_f=x_b + 0.075/2
            
        y_b=-0.1
        y_p=-0.35

        con_le = ConnectionPatch(
            xyA=(x_b,y_b),
            xyB=(x_p_i,y_p),
            coordsA=ax0.transAxes,
            coordsB=ax0.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        con_ri = ConnectionPatch(
            xyA=(x_b,y_b),
            xyB=(x_p_f,y_p),
            coordsA=ax0.transAxes,
            coordsB=ax0.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )   
        con_ho = ConnectionPatch(
            xyA=(x_p_i,y_p),
            xyB=(x_p_f,y_p),
            coordsA=ax0.transAxes,
            coordsB=ax0.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=0,
        )
        
        line_le = con_le.get_path().vertices
        line_ri = con_ri.get_path().vertices
        line_ho = con_ho.get_path().vertices
        
        ax0.add_artist(con_le)  
        ax0.add_artist(con_ri) 
        ax0.add_artist(con_ho)  
        
        pi_tri_coords = np.stack(
            (con_le.get_path().vertices[0].copy(),
            con_le.get_path().vertices[-1].copy(),
            con_ri.get_path().vertices[-1].copy()),
            axis=0,
        )
        pi_tri = plt.Polygon(pi_tri_coords,ec='lightgray',fc='lightgray',alpha=0.5,zorder=0,clip_on=False)    
        ax0.add_artist(pi_tri)                    
            
    for i,k in enumerate(list(ig_dict.keys())):
        ax0.annotate(
            '{}: {}'.format(ig_dict[k],k),
            xy=(0,-1.1-i*0.1),
            color='gray',
            # ha='center',
            xycoords=ax0.transAxes
        )
            
    f.savefig('./ms_figures/combined_plot_piecharts_new_{}.png',dpi=1000,bbox_inches='tight'.format(flags['version']))
    # f.savefig('./ms_figures/combined_plot_piecharts_50.pdf',dpi=50,bbox_inches='tight')
    # f.savefig('./ms_figures/combined_plot_piecharts_500.pdf',dpi=500,bbox_inches='tight')

#%% ----------------------------------------------------------------
# combined plot showing absolute cohort sizes (f2 after review)
# ------------------------------------------------------------------

def plot_combined_population(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
    df_countries,
):

    # plot characteristics
    plot_var='unprec_99.99'
    x=12
    y=7
    markersize=10
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   
    by=2020
    cmap_whole = plt.cm.get_cmap('Reds')
    levels_cmap = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels_cmap[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,101,10)
    levels = np.arange(0,101,5)
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)
    l = 0 # letter indexing
    gmt_indices_152535 = [20,10,0]
    map_letters = {20:'e',10:'d',0:'c'}

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(10,3)

    # box plots
    # ax0 = f.add_subplot(gs0[0:4,0:2]) 
    ax0 = f.add_subplot(gs0[0:5,0:2]) 

    # pop totals
    # ax1 = f.add_subplot(gs0[4:,0:2],sharex=ax0)
    ax1 = f.add_subplot(gs0[5:,0:2],sharex=ax0)

    gs0.update(hspace=0)

    # maps
    gs01 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[0:9,2:],
    )
    ax01 = f.add_subplot(gs01[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs01[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs01[2],projection=ccrs.Robinson())  
    
    pos00 = ax21.get_position()
    cax00 = f.add_axes([
        pos00.x0-0.05,
        pos00.y0-0.1,
        pos00.width*1.95,
        pos00.height*0.2
    ])

    # get data
    df_list_gs = []
    extr='heatwavedarea'
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
        ds_pf_gs_plot = pk.load(file)
    da_p_gs_plot = ds_pf_gs_plot[plot_var].loc[{
        'GMT':GMT_indices_plot,
        'birth_year':sample_birth_years,
    }]
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)  
                
    # --------------------------------------------------------------------
    # pf boxplot time series
    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
        df_pf_gs_plot_step['hazard'] = extr
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)

    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    p = sns.boxplot(
        data=df_pf_gs_plot[df_pf_gs_plot['hazard']==extr],
        x='birth_year',
        y='pf',
        hue='GMT_label',
        palette=colors,
        whis=(0,100),
        showcaps=False,
        showfliers=False,
        boxprops={
            'linewidth':0,
            'alpha':0.5
        },        
        ax=ax0,
    )
    p = sns.stripplot(
        data=df_pf_gs_plot[df_pf_gs_plot['hazard']==extr],
        x='birth_year',
        y='pf',
        hue='GMT_label',
        palette=colors,
        dodge=True,
        alpha=0.4,
        # whis=(0,100),
        # showcaps=False,
        # showfliers=False,
        # boxprops={
        #     'linewidth':0,
        #     'alpha':0.5
        # },        
        ax=ax0,
    )    
    p.legend_.remove()                  
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)      
    ax0.tick_params(colors='gray')
    ax0.set_ylim(0,100)
    ax0.spines['left'].set_color('gray')
    ax0.spines['bottom'].set_color('gray')      
    ax0.set_ylabel('$\mathregular{CF_{Heatwaves}}$ [%]',color='gray',fontsize=14)
    ax0.set_xlabel('Birth year',color='gray',fontsize=14)       
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    l+=1 

    # bbox
    x0 = 0.025
    y0 = 0.7
    xlen = 0.2
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 10
    legend_lw=3.5   

    legendcols = list(colors.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2])
    ]

    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',    
    ]

    ax0.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=1,
        fontsize=legend_font, 
        labelcolor='gray',
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )   

    # --------------------------------------------------------------------
    # populations

    ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)      
    ax1.tick_params(colors='gray')
    # ax1.set_ylim(0,100)
    ax1.spines['top'].set_color('gray')   
    ax1.spines['left'].set_color('gray')
    ax1.spines['bottom'].set_color('gray')    

    incomegroups = df_countries['incomegroup'].unique()
    income_countries = {}
    for category in incomegroups:
        income_countries[category] = list(df_countries.index[df_countries['incomegroup']==category])

    heights={}
    for category in incomegroups:
        heights[category] = da_gs_popdenom.loc[{
            'birth_year':np.arange(1960,2021,10),'country':income_countries[category]
        }].sum(dim='country').values / 10**6
    testdf = pd.DataFrame(heights)    
    testdf['birth_year'] = np.arange(1960,2021,10)
    testdf = testdf.set_index('birth_year')             
    testdf['total'] = testdf.sum(axis=1)
    p1 = testdf['total'].plot(
        kind='bar',
        # column='total',
        # stacked=True,
        color='gray',      
        ax=ax1,
        legend=False,
        rot=0.5
    ) 
    ax1.invert_yaxis()
    ax1.set_xlabel('Birth year',color='gray',fontsize=14)
    ax1.set_ylabel('Global cohort \n totals [in millions]',color='gray',fontsize=14)  
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )  
    # plot cohort sizes as text inside bars
    for i,by in enumerate(testdf.index.values):
        ax1.text(
            x=i,
            y=np.round(testdf['total'].loc[by]) - 10,
            s=str(int(np.round(testdf['total'].loc[by]))),
            horizontalalignment='center',
            verticalalignment='center',
            # transform=ax1.transData,
            fontsize=8,
            color='white',
        )
    
   
    
    l+=1
    # --------------------------------------------------------------------
    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways

    da_p_gs_plot = ds_pf_gs[plot_var].loc[{
        'GMT':gmt_indices_152535,
        'birth_year':by,
    }]
    df_list_gs = []
    for step in gmt_indices_152535:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
        da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    for ax,step in zip((ax01,ax11,ax21),gmt_indices_152535):
        gdf_p['pf']=df_p_gs_plot['pf'][df_p_gs_plot['GMT']==step].values
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        gdf_p.to_crs(robinson).plot(
            ax=ax,
            column='pf',
            cmap=cmap_list_frac,
            norm=norm,
            cax=cax00,
            zorder=2,
            rasterized=True,
        )

        gdf.to_crs(robinson).plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
            zorder=3,
        )
        
        gdf_robinson_bounds = gdf_p.to_crs(robinson).total_bounds # (minx,miny,maxx,maxy)
        
        # ax.set_global() # checking to see if this makes projections fully showing antarctica
        
        ax.set_title(
            '{} °C'.format(gmt_legend[step]),
            loc='center',
            fontweight='bold',
            fontsize=12,
            color='gray',       
        )
        
        ax.set_title(
            map_letters[step],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    

        # triangles showing connections between GMT step box plot to map panels ---------------
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        if step == gmt_indices_152535[0]:
            x_h=1 
        elif step == gmt_indices_152535[1]:
            x_h=0.95                      
        elif step == gmt_indices_152535[-1]:
            x_h=0.9
        y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
        x_m=0
        y_m_i=0
        y_m_f=1.0
        con_low = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m_i),
            coordsA=ax0.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=5,
        )
        con_hi = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m_f),
            coordsA=ax0.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=5,
        )   
        con_vert = ConnectionPatch(
            xyA=(x_m,y_m_i),
            xyB=(x_m,y_m_f),
            coordsA=ax.transAxes,
            coordsB=ax.transAxes,
            color='lightgray',
            alpha=0.5,
            zorder=5,
        )
        
        line_low = con_low.get_path().vertices
        line_hi = con_hi.get_path().vertices
        
        ax0.add_artist(con_low)  
        ax0.add_artist(con_hi) 
        
        tri_coords = np.stack(
            (con_low.get_path().vertices[0],con_low.get_path().vertices[-1],con_hi.get_path().vertices[-1]),
            axis=0,
        )
        triangle = plt.Polygon(tri_coords,ec='lightgray',fc='lightgray',alpha=0.5,zorder=10,clip_on=False)    
        ax0.add_artist(triangle)           
        
        # triangles showing connections between GMT step box plot to map panels ---------------        
        
    cb = mpl.colorbar.ColorbarBase(
        ax=cax00, 
        cmap=cmap_list_frac,
        norm=norm,
        orientation='horizontal',
        spacing='uniform',
        ticks=ticks,
        drawedges=False,
    )

    cb.set_label(
        '$\mathregular{CF_{Heatwaves}}$ for 2020 birth cohort [%]',
        fontsize=14,
        color='gray'
    )
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)   
    cax00.xaxis.set_label_position('top')   

    f.savefig('./ms_figures/f2_combined_plot_popsizes.png',dpi=1000)
    f.savefig('./ms_figures/f2_combined_plot_popsizes.pdf',dpi=1000)
    return gdf_robinson_bounds
    
#%% ----------------------------------------------------------------
# combined plot showing absolute cohort sizes and pie charts
# ------------------------------------------------------------------

def plot_combined_population_piechart(
    df_GMT_strj,
    ds_pf_gs,
    da_gs_popdenom,
    gdf_country_borders,
    sims_per_step,
    flags,
    df_countries,
):

    # plot characteristics
    x=12
    y=7
    markersize=10
    labelfontsize=12
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = 'gray'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors   
    by=2020
    cmap_whole = plt.cm.get_cmap('Reds')
    levels_cmap = np.arange(0,1.01,0.05)
    colors = [cmap_whole(i) for i in levels_cmap[:-1]]
    cmap_list_frac = mpl.colors.ListedColormap(colors,N=len(colors))
    ticks = np.arange(0,101,10)
    levels = np.arange(0,101,5)
    norm=mpl.colors.BoundaryNorm(levels,ncolors=len(levels)-1)
    l = 0 # letter indexing
    gmt_indices_152535 = [24,15,6]
    map_letters = {24:'g',15:'f',6:'e'}

    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }     

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(12,3)

    # box plots
    ax0 = f.add_subplot(gs0[0:4,0:2]) 

    # pop totals
    # ax1 = f.add_subplot(gs0[3:7,0:2],sharex=ax0)
    ax1 = f.add_subplot(gs0[4:7,0:2])

    gs0.update(hspace=0)

    # pie charts
    gs10 = gridspec.GridSpecFromSubplotSpec(
        1,
        3, 
        subplot_spec=gs0[9:,0:2],
        # top=0.8
    )
    ax00 = f.add_subplot(gs10[0])
    ax10 = f.add_subplot(gs10[1])
    ax20 = f.add_subplot(gs10[2]) 

    # maps
    gs01 = gridspec.GridSpecFromSubplotSpec(
        3,
        1, 
        subplot_spec=gs0[0:10,2:],
    )
    ax01 = f.add_subplot(gs01[0],projection=ccrs.Robinson())
    ax11 = f.add_subplot(gs01[1],projection=ccrs.Robinson())
    ax21 = f.add_subplot(gs01[2],projection=ccrs.Robinson()) 
    pos00 = ax21.get_position()
    cax00 = f.add_axes([
        pos00.x0-0.05,
        pos00.y0-0.15,
        pos00.width*1.95,
        pos00.height*0.2
    ])

    # get data
    df_list_gs = []
    extr='heatwavedarea'
    with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
        d_isimip_meta = pk.load(file)              
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
        ds_pf_gs_plot = pk.load(file)
    da_p_gs_plot = ds_pf_gs_plot[plot_var].loc[{
        'GMT':GMT_indices_plot,
        'birth_year':sample_birth_years,
    }]
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)  
                
    # --------------------------------------------------------------------
    # pf boxplot time series
    for step in GMT_indices_plot:
        da_pf_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].fillna(0).sum(dim='country') / da_gs_popdenom.sum(dim='country') * 100
        df_pf_gs_plot_step = da_pf_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_pf_gs_plot_step['GMT_label'] = df_pf_gs_plot_step['GMT'].map(gmt_legend)       
        df_pf_gs_plot_step['hazard'] = extr
        df_list_gs.append(df_pf_gs_plot_step)
    df_pf_gs_plot = pd.concat(df_list_gs)

    colors = dict(zip(list(gmt_legend.values()),['steelblue','darkgoldenrod','darkred']))
    p = sns.boxplot(
        data=df_pf_gs_plot[df_pf_gs_plot['hazard']==extr],
        x='birth_year',
        y='pf',
        hue='GMT_label',
        palette=colors,
        showcaps=False,
        showfliers=False,
        boxprops={
            'linewidth':0,
            'alpha':0.5
        },        
        ax=ax0,
    )
    p.legend_.remove()                  
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)      
    ax0.tick_params(colors='gray')
    ax0.set_ylim(0,100)
    ax0.spines['left'].set_color('gray')
    ax0.spines['bottom'].set_color('gray')      
    ax0.set_ylabel('$\mathregular{CF_{Heatwaves}}$ [%]',color='gray',fontsize=labelfontsize)
    ax0.set_xlabel('Birth year',color='gray',fontsize=labelfontsize)       
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )    
    l+=1 

    # bbox
    x0 = 0.01
    y0 = 0.7
    xlen = 0.2
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75

    legend_font = 8
    legend_lw=3.5   

    legendcols = list(colors.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2])
    ]

    labels= [
        '1.5 °C GMT warming by 2100',
        '2.5 °C GMT warming by 2100',
        '3.5 °C GMT warming by 2100',    
    ]

    ax0.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=1,
        fontsize=legend_font, 
        labelcolor='gray',
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )   

    # --------------------------------------------------------------------
    # populations

    ax1.spines['right'].set_visible(False)
    # ax1.spines['top'].set_visible(False)      
    ax1.tick_params(colors='gray')
    # ax1.set_ylim(0,100)
    ax1.spines['top'].set_color('gray')   
    ax1.spines['left'].set_color('gray')
    ax1.spines['bottom'].set_color('gray')    

    colors_pop = {
        'Low income': plt.cm.get_cmap('tab20b')(1),
        'Lower middle income': plt.cm.get_cmap('tab20b')(5),
        'Upper middle income': plt.cm.get_cmap('tab20b')(9),
        'High income': plt.cm.get_cmap('tab20b')(13),
    }
    incomegroups = df_countries['incomegroup'].unique()
    income_countries = {}
    for category in incomegroups:
        income_countries[category] = list(df_countries.index[df_countries['incomegroup']==category])

    heights={}
    for category in incomegroups:
        heights[category] = da_gs_popdenom.loc[{
            'birth_year':np.arange(1960,2021,10),'country':income_countries[category]
        }].sum(dim='country').values / 10**6
    testdf = pd.DataFrame(heights)    
    testdf['birth_year'] = np.arange(1960,2021,10)
    testdf = testdf.set_index('birth_year')             
    p1 = testdf.plot(
        kind='bar',
        stacked=True,
        color=colors_pop,      
        ax=ax1,
        legend=False,
        rot=0.5
    ) 
    ax1.invert_yaxis()
    ax1.set_xlabel('Birth year',color='gray',fontsize=labelfontsize)
    ax1.set_ylabel('Global cohort \n totals [in millions]',color='gray',fontsize=labelfontsize)  
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        fontsize=10
    )  

    # bbox for legend
    # x0 = 0.01
    # y0 = 0.08
    # xlen = 0.2
    # ylen = 0.3
    x0 = 0.01
    y0 = -0.1
    xlen = 0.3
    ylen = 0.3

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75
    # legend_font = 10
    legend_lw=3.5   

    legendcols = list(colors_pop.values())
    handles = [
        Rectangle((0,0),1,1,color=legendcols[0]),\
        Rectangle((0,0),1,1,color=legendcols[1]),\
        Rectangle((0,0),1,1,color=legendcols[2]),\
        Rectangle((0,0),1,1,color=legendcols[3])
    ]

    # labels= list(colors_pop.keys())
    labels = list(ig_dict.values())

    ax1.legend(
        handles, 
        labels, 
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'upper left',
        ncol=4,
        fontsize=legend_font, 
        labelcolor='gray',
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        columnspacing=0.05, 
        handlelength=legend_entrylen, 
        handletextpad=legend_entrypad
    )       
    
    l+=1

    # --------------------------------------------------------------------
    # pie charts

    gmt_indices_sample = [24,15,6]
    gmt_legend={
        gmt_indices_sample[0]:'1.5',
        gmt_indices_sample[1]:'2.5',
        gmt_indices_sample[2]:'3.5',
    }
    colors = dict(zip([6,15,24],['steelblue','darkgoldenrod','darkred']))

    by_sample = [1960,1990,2020]
    incomegroups = df_countries['incomegroup'].unique()
    income_countries = {}
    for category in incomegroups:
        income_countries[category] = list(df_countries.index[df_countries['incomegroup']==category])
    ig_dict = {
        'Low income':'LI',
        'Lower middle income': 'LMI',
        'Upper middle income': 'UMI',
        'High income': 'HI',
    }        

    income_unprec = {}
    da_unprec = ds_pf_gs[plot_var]
    for category in list(income_countries.keys()):
        income_unprec[category] = da_unprec.loc[{
            'country':income_countries[category],
            'GMT':gmt_indices_sample,
        }]

        
    sims_per_step = {}
    for step in GMT_labels:
        sims_per_step[step] = []
        for i in list(d_isimip_meta.keys()):
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                sims_per_step[step].append(i)
            
    pi_totals = {}
    pi_ratios = {}
    for by in by_sample:
        
        # populate each birth year with totals per income group
        pi_totals[by] = [da_gs_popdenom.loc[{
            'country':income_countries[category],
            'birth_year':by,
        }].sum(dim='country').item() for category in list(income_countries.keys())]
        
        
        pi_ratios[by] = {}
        for category in list(income_countries.keys()):
            pi_ratios[by][category] = {}
            for step in gmt_indices_sample:
                unprec = income_unprec[category].loc[{
                    'GMT':step,
                    'run':sims_per_step[step],
                    'birth_year':by
                }].sum(dim='country').mean(dim='run').item()
                pi_ratios[by][category][step] = unprec / da_gs_popdenom.loc[{
                    'country':income_countries[category],
                    'birth_year':by,
                }].sum(dim='country').item()
        

    for by,ax in zip(by_sample,[ax00,ax10,ax20]):
        for i,category in enumerate(list(income_countries.keys())):
            order = np.argsort(list(pi_ratios[by][category].values())) # indices that would sort pf across gmt steps
            order = order[::-1] # want decreasing pf order, so we reverse the indices
            ordered_gmts = [gmt_indices_sample[o] for o in order]
            for step in ordered_gmts:
                colors_list =['None']*4
                colors_list[i] = 'white'
                # first paint white where we want actual color (semi transparent, white makes it look good) per category with nothing in other categories
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }     
                )
                # then paint actual color and radius
                colors_list[i] = colors[step]
                ax.pie(
                    x=pi_totals[by],
                    colors=colors_list,
                    radius=pi_ratios[by][category][step],
                    wedgeprops={
                        'width':pi_ratios[by][category][step], 
                        'edgecolor':'None',
                        'linewidth': 0.5,
                        'alpha':0.5,
                    }
                )                 
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    
        ax.set_title(
            '{} CF'.format(by),
            loc='center',
            fontweight='bold',
            fontsize=labelfontsize,
            color='gray',       
        )        
        l+=1             
        percents = ['25%','50%','75%','100%']
        for i,r in enumerate(np.arange(0.25,1.01,0.25)):
            if r < 1:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    }
                )      
            else:
                ax.pie(
                    x=pi_totals[by],
                    colors=['None']*4,
                    radius=r,
                    labels=list(ig_dict.values()),
                    wedgeprops={
                        'width':r, 
                        'edgecolor':'0.5',
                        'linewidth': 0.5,
                    },
                    textprops={
                        'color':'0.5'
                    }
                )        
            
            if by == 1990:
                ax.annotate(
                    percents[i],
                    xy=(0,r+0.05),
                    color='gray',
                    ha='center',
                    fontsize=6
                )
            
    for i,k in enumerate(list(ig_dict.keys())):
        ax1.annotate(
            '{}: {}'.format(ig_dict[k],k),
            # xy=(0,-1.1-i*0.1),
            xy=(0,-2.15-i*0.1),
            color='gray',
            # ha='center',
            xycoords=ax0.transAxes
        )
    # --------------------------------------------------------------------
    # maps of pop frac emergence for countries at 1, 2 and 3 deg pathways

    da_p_gs_plot = ds_pf_gs[plot_var].loc[{
        'GMT':gmt_indices_152535,
        'birth_year':by,
    }]
    df_list_gs = []
    for step in gmt_indices_152535:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
        da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())
    robinson = ccrs.Robinson().proj4_init

    for ax,step in zip((ax01,ax11,ax21),gmt_indices_152535):
        gdf_p['pf']=df_p_gs_plot['pf'][df_p_gs_plot['GMT']==step].values
        ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
        gdf_p.to_crs(robinson).plot(
            ax=ax,
            column='pf',
            cmap=cmap_list_frac,
            norm=norm,
            cax=cax00,
            zorder=2,
            rasterized=True,
        )

        gdf.to_crs(robinson).plot(
            ax=ax,
            color='none', 
            edgecolor='black',
            linewidth=0.25,
            zorder=3,
        )
        
        ax.set_title(
            '{} °C'.format(gmt_legend[step]),
            loc='center',
            fontweight='bold',
            fontsize=labelfontsize,
            color='gray',       
        )
        
        ax.set_title(
            map_letters[step],
            loc='left',
            fontweight='bold',
            fontsize=10
        )    
        # l+=1          
        
        # pointers connecting 2020, GMT step pixel in heatmap to map panels ------------------
        if step == gmt_indices_152535[0]:
            x_h=1 
        elif step == gmt_indices_152535[1]:
            x_h=0.95                      
        elif step == gmt_indices_152535[-1]:
            x_h=0.9
        y_h= df_pf_gs_plot[(df_pf_gs_plot['birth_year']==by)&(df_pf_gs_plot['GMT']==step)]['pf'].median() / 100
        x_m=0
        y_m=0.5
        con = ConnectionPatch(
            xyA=(x_h,y_h),
            xyB=(x_m,y_m),
            coordsA=ax0.transAxes,
            coordsB=ax.transAxes,
            color='gray'
        )
        ax0.add_artist(con)          
        
    cb = mpl.colorbar.ColorbarBase(
        ax=cax00, 
        cmap=cmap_list_frac,
        norm=norm,
        orientation='horizontal',
        spacing='uniform',
        ticks=ticks,
        drawedges=False,
    )

    cb.set_label(
        '$\mathregular{CF_{Heatwaves}}$ for 2020 \n birth cohort [%]',
        fontsize=labelfontsize,
        color='gray'
    )
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(
        labelcolor=col_cbticlbl,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb.outline.set_edgecolor(col_cbedg)
    cb.outline.set_linewidth(cb_edgthic)   
    cax00.xaxis.set_label_position('top')   

    f.savefig('./ms_figures/combined_plot_popsizes_piecharts.png',dpi=1000)
                        
#%% ----------------------------------------------------------------
# plot ar6 hexagons with landfrac per extreme and multi extreme panels
    
def plot_hexagon_landfrac_union(
    d_global_emergence,
):                
    gdf_ar6_hex = gpd.read_file('./data/shapefiles/zones.gpkg').rename(columns={'label': 'Acronym'})
    gdf_ar6_hex = gdf_ar6_hex.set_index('Acronym').drop(['id','Continent','Name'],axis=1)
    gdf_ar6_hex = gdf_ar6_hex.drop(labels=['GIC'],axis=0)

    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    from matplotlib.patches import ConnectionPatch
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection

    x=11
    y=8.1
    markersize=10
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = '0'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    
    density=6  # density of hatched lines showing frac of sims with emergence
    landfrac_threshold = 10
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }  

    # colorbar stuff for landfrac emerging

    cmap_landfrac = plt.cm.get_cmap('Reds')
    levels = np.arange(0,1.01,0.05)
    colors_landfrac = [cmap_landfrac(i) for i in levels[:-1]]
    cmap_list_landfrac = mpl.colors.ListedColormap(colors_landfrac,N=len(colors_landfrac))
    ticks_landfrac = np.arange(0,101,10)
    norm_landfrac = mpl.colors.BoundaryNorm(levels*100,cmap_list_landfrac.N)   

    # colorbar stuff for union
    cmap_reds = plt.cm.get_cmap('Reds')
    colors_union = [
        'white',
        cmap_reds(0.25),
        cmap_reds(0.50),
        cmap_reds(0.75),
    ]
    cmap_list_union = mpl.colors.ListedColormap(colors_union,N=len(colors_union))
    cmap_list_union.set_over(cmap_reds(0.99))
    levels = np.arange(0.5,3.6,1)
    union_levels = np.arange(-0.5,3.6,1)
    norm_union=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(8,2)
    gs0.update(wspace=0.25,hspace=0.2)
    ax0 = f.add_subplot(gs0[5:8,0:1])

    # left side for 1960
    # masp per hazard
    gsn0 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,0:1],
        wspace=0,
        hspace=0,
    )
    ax00 = f.add_subplot(gsn0[0])
    ax10 = f.add_subplot(gsn0[1])
    ax20 = f.add_subplot(gsn0[2]) 

    ax01 = f.add_subplot(gsn0[3])
    ax11 = f.add_subplot(gsn0[4])
    ax21 = f.add_subplot(gsn0[5])       

    # colorbars
    pos0 = ax0.get_position()

    # colorbar for landfrac of emergence
    cax_landfrac = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.0575,
        pos0.width*1.5,
        pos0.height*0.15
    ])
    # colorbar for union
    cax_union = f.add_axes([
        pos0.x0-0.07,
        pos0.y0-0.2,
        pos0.width*1.5,
        pos0.height*0.15
    ])

    # right side for 2020
    ax1 = f.add_subplot(gs0[5:8,1:2]) # map of emergence union
    gsn1 = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=gs0[0:5,1:2],
        wspace=0,
        hspace=0,
    )
    ax02 = f.add_subplot(gsn1[0])
    ax12 = f.add_subplot(gsn1[1])
    ax22 = f.add_subplot(gsn1[2]) 

    ax03 = f.add_subplot(gsn1[3])
    ax13 = f.add_subplot(gsn1[4])
    ax23 = f.add_subplot(gsn1[5])     

    # plot 1960
    i=0
    l=0

    ax00.annotate(
        '1960 birth cohort',
        (0.55,1.3),
        xycoords=ax00.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )          

    i+=1

    # new gdf for union
    gdf_ar6_hex_union_1960 = gdf_ar6_hex.copy()
    gdf_ar6_hex_union_1960['union'] = 0

    for ax,extr in zip((ax00,ax10,ax20,ax01,ax11,ax21),extremes):
        
        ds_global_emergence = d_global_emergence[extr]
        gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
        # gdf_hex_landrac_plottable = gdf_ar6_emerged_landfrac.merge(gdf_ar6_hex,left_index=True,right_index=True)
        gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
        gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==1960]
        gdf_hex_landrac_plottable_1960['landfrac'] = gdf_hex_landrac_plottable_1960['landfrac'] * 100 
        gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable_1960[['geometry','birth_year','region','names','landfrac']]
        gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'] + (gdf_hex_landrac_plottable_1960['landfrac'] > landfrac_threshold)
        gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'].astype(float)
        
        gdf_hex_landrac_plottable_1960.plot(
            column='landfrac',
            cmap=cmap_list_landfrac,
            norm=norm_landfrac,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray'
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1    
        i+=1
        
    gdf_ar6_hex_union_1960.plot(
        column='union',
        cmap=cmap_list_union,
        norm=norm_union,
        edgecolor='gray',
        ax=ax0,
    )
    ax0.set_axis_off()
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1                 
    i+=1

    # 2020 birth cohort
    ax02.annotate(
        '2020 birth cohort',
        (0.55,1.3),
        xycoords=ax02.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )       

    # new gdf for union
    gdf_ar6_hex_union_2020 = gdf_ar6_hex.copy()
    gdf_ar6_hex_union_2020['union'] = 0

    for ax,extr in zip((ax02,ax12,ax22,ax03,ax13,ax23),extremes):
        
        ds_global_emergence = d_global_emergence[extr]
        gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
        gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
        gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
        gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==2020]
        gdf_hex_landrac_plottable_2020['landfrac'] = gdf_hex_landrac_plottable_2020['landfrac'] * 100     
        gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable_2020[['geometry','birth_year','region','names','landfrac']]
        gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'] + (gdf_hex_landrac_plottable_2020['landfrac'] > landfrac_threshold)
        gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'].astype(float)
        
        gdf_hex_landrac_plottable_2020.plot(
            column='landfrac',
            cmap=cmap_list_landfrac,
            norm=norm_landfrac,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            extremes_labels[extr],
            loc='center',
            fontweight='bold',
            fontsize=9,
            color='gray',
        )
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        l+=1          
        i+=1  
        
    gdf_ar6_hex_union_2020.plot(
        column='union',
        cmap=cmap_list_union,
        norm=norm_union,
        edgecolor='gray',
        ax=ax1,
    )
    ax1.set_axis_off()
    ax1.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    )    
    l+=1  

    # colorbar for median landfrac emergence per extreme
    cb_landfrac = mpl.colorbar.ColorbarBase(
        ax=cax_landfrac, 
        cmap=cmap_list_landfrac,
        norm=norm_landfrac,
        orientation='horizontal',
        extend='neither',
        spacing='uniform',
        ticks=ticks_landfrac,
        drawedges=False,
    )
    cb_landfrac.ax.set_title(
        '(' + r"$\bf{g}$" + ',' + r"$\bf{n}$" + ')',
        y=1.06,
        fontsize=10,
        loc='left',
    )
    cb_landfrac.set_label(
        'Median percent of land area emerging'.format(str(landfrac_threshold)),
        fontsize=10,
        labelpad=8,
        color='gray',
    )
    cb_landfrac.ax.xaxis.set_label_position('top')
    cb_landfrac.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   

    # colorbar for union of emergence across extremes
    cb_u = mpl.colorbar.ColorbarBase(
        ax=cax_union, 
        cmap=cmap_list_union,
        norm=norm_union,
        orientation='horizontal',
        extend='max',
        spacing='uniform',
        ticks=np.arange(0,7).astype('int'),
        drawedges=False,
    )
    cb_u.ax.set_title(
        '(' + r"$\bf{g}$" + ',' + r"$\bf{n}$" + ')',
        y=1.06,
        fontsize=10,
        loc='left',
    )

    cb_u.set_label(
        'Number of extremes emerging in at least {}% of region'.format(str(landfrac_threshold)),
        fontsize=10,
        labelpad=8,
        color='gray',
    )
    cb_u.ax.xaxis.set_label_position('top')
    cb_u.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   

    f.savefig('./ms_figures/emergence_landfrac_union_hexagons_{}.png'.format(landfrac_threshold),dpi=1000,bbox_inches='tight')

#%% ----------------------------------------------------------------

def plot_hexagon_multithreshold(
    d_global_emergence,
):                   
    gdf_ar6_hex = gpd.read_file('./data/shapefiles/zones.gpkg').rename(columns={'label': 'Acronym'})
    gdf_ar6_hex = gdf_ar6_hex.set_index('Acronym').drop(['id','Continent','Name'],axis=1)
    gdf_ar6_hex = gdf_ar6_hex.drop(labels=['GIC'],axis=0)

    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    from matplotlib.patches import ConnectionPatch
    from matplotlib.patches import Circle, Wedge, Polygon
    from matplotlib.collections import PatchCollection

    x=8
    y=8.1
    markersize=10
    col_cbticlbl = 'gray'   # colorbar color of tick labels
    col_cbtic = 'gray'   # colorbar color of ticks
    col_cbedg = '0'   # colorbar color of edge
    cb_ticlen = 3.5   # colorbar length of ticks
    cb_ticwid = 0.4   # colorbar thickness of ticks
    cb_edgthic = 0   # colorbar thickness of edges between colors    
    density=6  # density of hatched lines showing frac of sims with emergence
    landfrac_threshold = 10
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }

    # colorbar stuff for union
    cmap_reds = plt.cm.get_cmap('Reds')
    colors_union = [
        'white',
        cmap_reds(0.25),
        cmap_reds(0.50),
        cmap_reds(0.75),
    ]
    cmap_list_union = mpl.colors.ListedColormap(colors_union,N=len(colors_union))
    cmap_list_union.set_over(cmap_reds(0.99))
    levels = np.arange(0.5,3.6,1)
    union_levels = np.arange(-0.5,3.6,1)
    norm_union=mpl.colors.BoundaryNorm(union_levels,ncolors=len(union_levels)-1)

    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(9,2)
    gs0.update(wspace=0.25,hspace=0.2)

    # left side 1960
    ax0 = f.add_subplot(gs0[0:2,0:1],projection=ccrs.Robinson())
    ax1 = f.add_subplot(gs0[3:5,0:1])
    ax2 = f.add_subplot(gs0[5:7,0:1])
    ax3 = f.add_subplot(gs0[7:9,0:1])

    # colorbars
    pos0 = ax3.get_position()

    # colorbar for union
    cax_union = f.add_axes([
        pos0.x0+0.15,
        pos0.y0-0.1,
        pos0.width*1.5,
        pos0.height*0.15
    ])

    # right side for 2020
    ax4 = f.add_subplot(gs0[0:2,1:2])
    ax5 = f.add_subplot(gs0[3:5,1:2])
    ax6 = f.add_subplot(gs0[5:7,1:2])
    ax7 = f.add_subplot(gs0[7:9,1:2])   
    
    # plot 1960
    i=0
    l=0

    ax0.annotate(
        '1960 birth cohort',
        (0.15,1.3),
        xycoords=ax1.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )          

    i+=1

    # plot ar6 key
    text_kws = dict(
        bbox=dict(color="none"),
        path_effects=[pe.withStroke(linewidth=2, foreground="w")],
        color="gray", 
        fontsize=4, 
    )
    line_kws=dict(lw=0.5)
    ar6_polys = rm.defined_regions.ar6.land
    ar6_polys = ar6_polys[np.arange(1,44).astype(int)]
    ar6_polys.plot(ax=ax0,label='abbrev',line_kws=line_kws,text_kws=text_kws)
    ax0.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    ) 
    l+=1

    # plot blank hexagons
    gdf_ar6_hex_blank = gdf_ar6_hex.copy()
    gdf_ar6_hex_blank = gdf_ar6_hex_blank.drop(labels='PAC',axis=0)
    gdf_ar6_hex_blank['label'] = list(gdf_ar6_hex_blank.index)
    gdf_ar6_hex_blank.plot(
        ax=ax4,
        color='w',
        edgecolor='gray',
    )
    # ar6_polys.abbrevs NEED TO GET THESE IN HEXAGONS
    gdf_ar6_hex_blank.apply(
        lambda x: ax4.annotate(
            text=x['label'], 
            xy=x.geometry.centroid.coords[0], 
            ha='center',
            color="gray", 
            fontsize=4, 
        ), 
        axis=1
    )
    ax4.set_axis_off()
    ax4.set_title(
        letters[l],
        loc='left',
        fontweight='bold',
        color='k',
        fontsize=10,
    ) 
    l+=1

    for ax,thresh in zip((ax1,ax2,ax3),(10,20,30)):
        
        # new gdf for union
        gdf_ar6_hex_union_1960 = gdf_ar6_hex.copy()
        gdf_ar6_hex_union_1960['union'] = 0    
        
        for extr in extremes:
            
            ds_global_emergence = d_global_emergence[extr]
            gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
            gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
            gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==1960]
            gdf_hex_landrac_plottable_1960['landfrac'] = gdf_hex_landrac_plottable_1960['landfrac'] * 100 
            gdf_hex_landrac_plottable_1960 = gdf_hex_landrac_plottable_1960[['geometry','birth_year','region','names','landfrac']]
            gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'] + (gdf_hex_landrac_plottable_1960['landfrac'] > thresh)
            gdf_ar6_hex_union_1960['union'] = gdf_ar6_hex_union_1960['union'].astype(float)
        
        gdf_ar6_hex_union_1960.plot(
            column='union',
            cmap=cmap_list_union,
            norm=norm_union,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )    
        
        # label for threshold
        ax.annotate(
            'X = {}'.format(thresh),
            (-0.1,0.4),
            xycoords=ax.transAxes,
            fontsize=10,
            rotation='vertical',
            color='gray',
        )           
        
        l+=1                 
        i+=1

    # 2020 birth cohort
    ax4.annotate(
        '2020 birth cohort',
        (0.15,1.3),
        xycoords=ax5.transAxes,
        fontsize=14,
        rotation='horizontal',
        color='gray',
        fontweight='bold',
    )       

    # for ax,thresh in zip((ax4,ax5,ax6,ax7),(10,20,30,40)):
    # for ax,thresh in zip((ax4,ax5,ax6),(10,20,30)):
    for ax,thresh in zip((ax5,ax6,ax7),(10,20,30)):
        
        # new gdf for union
        gdf_ar6_hex_union_2020 = gdf_ar6_hex.copy()
        gdf_ar6_hex_union_2020['union'] = 0
        
        for extr in extremes:
        
            ds_global_emergence = d_global_emergence[extr]
            gdf_ar6_emerged_landfrac = ds_global_emergence['emerged_area_ar6_landfrac_{}'.format(extr)].median(dim='run').to_dataframe().reset_index()
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.rename(mapper={'abbrevs':'Acronym','emerged_area_ar6_landfrac_{}'.format(extr):'landfrac'},axis=1).set_index('Acronym')
            gdf_ar6_emerged_landfrac = gdf_ar6_emerged_landfrac.drop(labels=['WAN','EAN','GIC',],axis=0)
            gdf_hex_landrac_plottable = gdf_ar6_hex.copy().merge(gdf_ar6_emerged_landfrac,left_index=True,right_index=True)
            gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable[gdf_hex_landrac_plottable['birth_year']==2020]
            gdf_hex_landrac_plottable_2020['landfrac'] = gdf_hex_landrac_plottable_2020['landfrac'] * 100     
            gdf_hex_landrac_plottable_2020 = gdf_hex_landrac_plottable_2020[['geometry','birth_year','region','names','landfrac']]
            gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'] + (gdf_hex_landrac_plottable_2020['landfrac'] > thresh)
            gdf_ar6_hex_union_2020['union'] = gdf_ar6_hex_union_2020['union'].astype(float)
        
        gdf_ar6_hex_union_2020.plot(
            column='union',
            cmap=cmap_list_union,
            norm=norm_union,
            edgecolor='gray',
            ax=ax,
        )
        ax.set_axis_off()
        ax.set_title(
            letters[l],
            loc='left',
            fontweight='bold',
            color='k',
            fontsize=10,
        )        
        
        l+=1          
        i+=1  
        

    # colorbar for union of emergence across extremes
    cb_u = mpl.colorbar.ColorbarBase(
        ax=cax_union, 
        cmap=cmap_list_union,
        norm=norm_union,
        orientation='horizontal',
        extend='max',
        spacing='uniform',
        ticks=np.arange(0,7).astype('int'),
        drawedges=False,
    )

    cb_u.set_label(
        'Extremes with median emergence in >X% of region',
        fontsize=10,
        labelpad=8,
        color='gray',
    )
    cb_u.ax.xaxis.set_label_position('top')
    cb_u.ax.tick_params(
        labelcolor=col_cbticlbl,
        labelsize=12,
        color=col_cbtic,
        length=cb_ticlen,
        width=cb_ticwid,
        direction='out'
    )   
    cb_u.outline.set_color('gray')

    # f.savefig('./ms_figures/emergence_union_hexagons_multithresh.png',dpi=1000,bbox_inches='tight')
    
#%% ----------------------------------------------------------------    
# preparing data for pramid plots    
def pyramid_setup(
    flags,
    ds_gdp,
    ds_grdi,
    da_cohort_size_1960_2020,
    ds_vulnerability,
):

    # GDP vulnerability quantiles ===========================================================
    extremes = [
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    
    GMT_integers = [0,10,12,17,20]
    gmt_low=0
    gmt_high=20
    
    if not os.path.isfile('./data/{}/pyramid_data_gdp.pkl'.format(flags['version'])):
    
        # also plot 1960 vs 2020 for gdp (grdi only has 2020)
        population_quantiles_10poorest = []
        population_quantiles_10richest = []
        
        population_quantiles_20poorest = []
        population_quantiles_20richest = []
        
        for by in birth_years:
            
            gdp = ds_gdp['gdp_isimip_rcp26_mean'].sel(birth_year=by)
            pop = da_cohort_size_1960_2020.sel(birth_year=by)

            gdp = gdp.where(pop.notnull())
            pop = pop.where(gdp.notnull())

            vulnerability = xr.DataArray(gdp.values.flatten())
            vulnerability = vulnerability.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(vulnerability))}) # have to do this so the coords are traceable back to the 2-D layout
            vulnerability_ranks = vulnerability.rank(dim='gridcell_number').round()

            pop_flat = xr.DataArray(pop.values.flatten())
            pop_flat = pop_flat.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(pop_flat))}) # have to do this so the coords are traceable back to the 2-D layout
            sorted_pop = pop_flat.sortby(vulnerability_ranks) # failed because gdp and pop need common mask
            sorted_pop_nonans = sorted_pop[sorted_pop.notnull()]
            sorted_pop_nonans_cumsum = sorted_pop_nonans.cumsum()
            sorted_pop_nonans_cumsum_pct = sorted_pop_nonans_cumsum / sorted_pop_nonans.sum()

            # gather pop totals for plotting for each birth year
            population_quantiles_10poorest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()/10**6) # groups all even population!!!    
            population_quantiles_10richest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item()/10**6) # groups all even population!!!  
            
            population_quantiles_20poorest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()+
                    sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[1].item())/10**6
            )
            population_quantiles_20richest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-2].item()+
                    sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item())/10**6
            )
            
        d_pyramid_plot_gdp = {}
        
        for e in extremes:
            
            v='gdp_q_by_p'
            df_vulnerability = ds_vulnerability.to_dataframe().reset_index()      
            df_vulnerability_e = df_vulnerability.loc[:,['run','GMT','qntl','vulnerability_index','birth_year',e]]
            df_vulnerability_e.loc[:,e] = df_vulnerability_e.loc[:,e] / 10**6 # convert to millions of people                
            d_pyramid_plot_gdp[e] = {}
            
            for GMT in GMT_integers: #GMT_current_policies
                
                d_pyramid_plot_gdp[e][GMT] = {}
                
                # ensemble mean unprecedented population for poor and rich quantiles
                unprec_pop_quantiles_10poorest = []
                unprec_pop_quantiles_10richest = []
                
                unprec_pop_quantiles_20poorest = []
                unprec_pop_quantiles_20richest = []     
                
                # errorbar info (+/- std) for the above quantiles
                unprec_pop_std_10poorest = []
                unprec_pop_std_10richest = []
                
                unprec_pop_std_20poorest = []
                unprec_pop_std_20richest = []         
                
                # ttest results ("*_poor" means that we test for poor quantiles to be greater. "*_rich" for rich quantiles to be greater)
                ttest_10pc_pvals_poor = []       
                ttest_20pc_pvals_poor = []       
                ttest_10pc_pvals_rich = []       
                ttest_20pc_pvals_rich = []       
                
                for by in birth_years:
                    
                    # gather unprec totals for plotting
                    # poorest 10 percent
                    poor_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10poorest.append(poor_unprec_10pc.mean())
                    unprec_pop_std_10poorest.append(poor_unprec_10pc.std())
                    
                    # richest 10 percent
                    rich_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10richest.append(rich_unprec_10pc.mean())
                    unprec_pop_std_10richest.append(rich_unprec_10pc.std())
                    
                    # t test for difference between rich and poor samples
                    ttest_10pc_poor = ttest_rel( # here test that poor is sig greater than rich
                        a=poor_unprec_10pc[poor_unprec_10pc.notnull()].values,
                        b=rich_unprec_10pc[rich_unprec_10pc.notnull()].values,
                        alternative='greater'
                    )
                    ttest_10pc_pvals_poor.append(ttest_10pc_poor.pvalue)
                    ttest_10pc_rich = ttest_rel( # here test that rich is sig greater than poor
                        a=rich_unprec_10pc[rich_unprec_10pc.notnull()].values,
                        b=poor_unprec_10pc[poor_unprec_10pc.notnull()].values,
                        alternative='greater',
                    )
                    ttest_10pc_pvals_rich.append(ttest_10pc_rich.pvalue)                   
                    
                    # poorest 20 percent
                    poor_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    poor_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==1)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20poorest.append(poor_unprec_20pci.mean() +poor_unprec_20pcii.mean())       
                    unprec_pop_std_20poorest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']<=1)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )
                    
                    # richest 20 percent
                    rich_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==8)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    rich_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20richest.append(rich_unprec_20pci.mean()+rich_unprec_20pcii.mean())
                    unprec_pop_std_20richest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']>=8)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )  
                    
                    # t test for difference between rich and poor samples
                    ttest_20pc_poor = ttest_rel(
                        a=np.concatenate((poor_unprec_20pci[poor_unprec_20pci.notnull()].values,poor_unprec_20pcii[poor_unprec_20pcii.notnull()].values)),
                        b=np.concatenate((rich_unprec_20pci[rich_unprec_20pci.notnull()].values,rich_unprec_20pcii[rich_unprec_20pcii.notnull()].values)),
                        alternative='greater'
                    )  
                    ttest_20pc_pvals_poor.append(ttest_20pc_poor.pvalue)
                    ttest_20pc_rich = ttest_rel(
                        a=np.concatenate((rich_unprec_20pci[rich_unprec_20pci.notnull()].values,rich_unprec_20pcii[rich_unprec_20pcii.notnull()].values)),
                        b=np.concatenate((poor_unprec_20pci[poor_unprec_20pci.notnull()].values,poor_unprec_20pcii[poor_unprec_20pcii.notnull()].values)),
                        alternative='greater'
                    )                      
                    ttest_20pc_pvals_rich.append(ttest_20pc_rich.pvalue)
                    
                # add stuff to plotting dictionary for pickling
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_quantiles_10poorest'] = unprec_pop_quantiles_10poorest
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_std_10poorest'] = unprec_pop_std_10poorest
                d_pyramid_plot_gdp[e][GMT]['population_quantiles_10poorest'] = population_quantiles_10poorest
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_quantiles_10richest'] = unprec_pop_quantiles_10richest
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_std_10richest'] = unprec_pop_std_10richest
                d_pyramid_plot_gdp[e][GMT]['population_quantiles_10richest'] = population_quantiles_10richest
                d_pyramid_plot_gdp[e][GMT]['ttest_10pc_pvals_poor'] = ttest_10pc_pvals_poor
                d_pyramid_plot_gdp[e][GMT]['ttest_10pc_pvals_rich'] = ttest_10pc_pvals_rich
                
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_quantiles_20poorest'] = unprec_pop_quantiles_20poorest
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_std_20poorest'] = unprec_pop_std_20poorest
                d_pyramid_plot_gdp[e][GMT]['population_quantiles_20poorest'] = population_quantiles_20poorest
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_quantiles_20richest'] = unprec_pop_quantiles_20richest
                d_pyramid_plot_gdp[e][GMT]['unprec_pop_std_20richest'] = unprec_pop_std_20richest
                d_pyramid_plot_gdp[e][GMT]['population_quantiles_20richest'] = population_quantiles_20richest
                d_pyramid_plot_gdp[e][GMT]['ttest_20pc_pvals_poor'] = ttest_20pc_pvals_poor
                d_pyramid_plot_gdp[e][GMT]['ttest_20pc_pvals_rich'] = ttest_20pc_pvals_rich        
                
            # then run the pyramid data selection for the low vs high gmt panels (this is just to get the pvalues for these comparisons, the other data is already there)
            # note that GMT integers as string "0_20" are used in the 2nd dictionary key level (instead of "GMT" integer above)
            d_pyramid_plot_gdp[e]['0_20'] = {} # i.e. "0" is the integer for the 1.5 degree pathway, "20" is for 3.5

            ttest_20pc_pvals_poor_0_vs_20 = [] # significance tests on poor (or high deprivation) across GMT pathways
            ttest_20pc_pvals_rich_0_vs_20 = [] # significance tests on rich (or low deprivation) across GMT pathways

            for by in birth_years:
                
                # poorest 20 percent, low gmt
                poor_unprec_gmt0_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==0)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]

                poor_unprec_gmt0_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==1)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]   
                
                # poorest 20 percent, high gmt
                poor_unprec_gmt20_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==0)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]

                poor_unprec_gmt20_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==1)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]                       

                # richest 20 percent, low gmt
                rich_unprec_gmt0_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==8)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]

                rich_unprec_gmt0_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==9)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]  
                
                # richest 20 percent, high gmt
                rich_unprec_gmt20_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==8)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]

                rich_unprec_gmt20_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==9)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]                      

                # t test for difference between low and high gmt pathways for poor/rich populations
                ttest_20pc_poor = ttest_ind(
                    a=np.concatenate((poor_unprec_gmt20_20pci[poor_unprec_gmt20_20pci.notnull()].values,poor_unprec_gmt20_20pcii[poor_unprec_gmt20_20pcii.notnull()].values)),
                    b=np.concatenate((poor_unprec_gmt0_20pci[poor_unprec_gmt0_20pci.notnull()].values,poor_unprec_gmt0_20pcii[poor_unprec_gmt0_20pcii.notnull()].values)),
                    alternative='greater'
                )  
                ttest_20pc_pvals_poor_0_vs_20.append(ttest_20pc_poor.pvalue)
                ttest_20pc_rich = ttest_ind(
                    a=np.concatenate((rich_unprec_gmt20_20pci[rich_unprec_gmt20_20pci.notnull()].values,rich_unprec_gmt20_20pcii[rich_unprec_gmt20_20pcii.notnull()].values)),
                    b=np.concatenate((rich_unprec_gmt0_20pci[rich_unprec_gmt0_20pci.notnull()].values,rich_unprec_gmt0_20pcii[rich_unprec_gmt0_20pcii.notnull()].values)),
                    alternative='greater'
                )                      
                ttest_20pc_pvals_rich_0_vs_20.append(ttest_20pc_rich.pvalue) 
                
            d_pyramid_plot_gdp[e]['0_20']['ttest_20pc_pvals_poor'] = ttest_20pc_pvals_poor_0_vs_20
            d_pyramid_plot_gdp[e]['0_20']['ttest_20pc_pvals_rich'] = ttest_20pc_pvals_rich_0_vs_20                    
                    
        with open('./data/{}/pyramid_data_gdp.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(d_pyramid_plot_gdp,f)   
            
    else:
        
        print("gdp pickle already there, delete pyramid_data_gdp.pkl and rerun function if you want to renew this data")
                    
    # grdi vulnerability quantiles ===========================================================
    if not os.path.isfile('./data/{}/pyramid_data_grdi.pkl'.format(flags['version'])):
    
        # also plot 1960 vs 2020 for gdp (grdi only has 2020)
        population_quantiles_10poorest = []
        population_quantiles_10richest = []
        
        population_quantiles_20poorest = []
        population_quantiles_20richest = []
        
        for by in birth_years:
            
            grdi = ds_grdi['grdi']
            pop = da_cohort_size_1960_2020.sel(birth_year=by)

            grdi = grdi.where(pop.notnull())
            pop = pop.where(grdi.notnull())
            
            vulnerability = xr.DataArray(grdi.values.flatten())
            vulnerability = vulnerability.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(vulnerability))}) # have to do this so the coords are traceable back to the 2-D layout
            vulnerability_ranks = vulnerability.rank(dim='gridcell_number').round()

            pop_flat = xr.DataArray(pop.values.flatten())
            pop_flat = pop_flat.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(pop_flat))}) # have to do this so the coords are traceable back to the 2-D layout
            sorted_pop = pop_flat.sortby(vulnerability_ranks) # failed because gdp and pop need common mask
            sorted_pop_nonans = sorted_pop[sorted_pop.notnull()]
            sorted_pop_nonans_cumsum = sorted_pop_nonans.cumsum()
            sorted_pop_nonans_cumsum_pct = sorted_pop_nonans_cumsum / sorted_pop_nonans.sum()

            # gather pop totals for plotting for each birth year
            population_quantiles_10poorest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item()/10**6) # groups all even population!!!    
            population_quantiles_10richest.append(sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()/10**6) # groups all even population!!!  
            
            population_quantiles_20poorest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-2].item()+
                    sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[-1].item())/10**6
            )
            population_quantiles_20richest.append(
                (sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[0].item()+
                    sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum()[1].item())/10**6
            )
        
        d_pyramid_plot_grdi = {}
         
        for e in extremes:
            
            # first run the pyramid data selection for the rich vs poor panel 
            v='grdi_q_by_p'
            df_vulnerability = ds_vulnerability.to_dataframe().reset_index()      
            df_vulnerability_e = df_vulnerability.loc[:,['run','GMT','qntl','vulnerability_index','birth_year',e]]
            df_vulnerability_e.loc[:,e] = df_vulnerability_e.loc[:,e] / 10**6 # convert to millions of people 
            d_pyramid_plot_grdi[e] = {}             
            
            for GMT in GMT_integers: 
                
                d_pyramid_plot_grdi[e][GMT] = {}        
                
                # ensemble mean unprecedented population for poor and rich quantiles
                unprec_pop_quantiles_10poorest = []
                unprec_pop_quantiles_10richest = []
                
                unprec_pop_quantiles_20poorest = []
                unprec_pop_quantiles_20richest = []     
                
                # errorbar info (+/- std) for the above quantiles
                unprec_pop_std_10poorest = []
                unprec_pop_std_10richest = []
                
                unprec_pop_std_20poorest = []
                unprec_pop_std_20richest = []         
                
                # ttest results ("*_poor" means that we test for poor quantiles to be greater. "*_rich" for rich quantiles to be greater)
                ttest_10pc_pvals_poor = []       
                ttest_20pc_pvals_poor = []       
                ttest_10pc_pvals_rich = []       
                ttest_20pc_pvals_rich = []                        
                
                for by in birth_years:            
                    
                    # gather unprec totals for plotting
                    # poorest 10 percent
                    poor_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10poorest.append(poor_unprec_10pc.mean())
                    unprec_pop_std_10poorest.append(poor_unprec_10pc.std())
                    
                    # richest 10 percent
                    rich_unprec_10pc = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    unprec_pop_quantiles_10richest.append(rich_unprec_10pc.mean())
                    unprec_pop_std_10richest.append(rich_unprec_10pc.std())
                    
                    # t test for difference between rich and poor samples
                    ttest_10pc_poor = ttest_rel( # here test that poor is sig greater than rich
                        a=poor_unprec_10pc[poor_unprec_10pc.notnull()].values,
                        b=rich_unprec_10pc[rich_unprec_10pc.notnull()].values,
                        alternative='greater'
                    )
                    ttest_10pc_pvals_poor.append(ttest_10pc_poor.pvalue)
                    ttest_10pc_rich = ttest_rel( # here test that rich is sig greater than poor
                        a=rich_unprec_10pc[rich_unprec_10pc.notnull()].values,
                        b=poor_unprec_10pc[poor_unprec_10pc.notnull()].values,
                        alternative='greater',
                    )
                    ttest_10pc_pvals_rich.append(ttest_10pc_rich.pvalue)                   
                    
                    # poorest 20 percent
                    poor_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==8)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    poor_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==9)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20poorest.append(poor_unprec_20pci.mean() +poor_unprec_20pcii.mean())       
                    unprec_pop_std_20poorest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']>=8)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )
                    
                    # richest 20 percent
                    rich_unprec_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==0)&\
                                (df_vulnerability_e['GMT']==GMT)][e]
                    
                    rich_unprec_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']==1)&\
                                (df_vulnerability_e['GMT']==GMT)][e]                   
                    unprec_pop_quantiles_20richest.append(rich_unprec_20pci.mean()+rich_unprec_20pcii.mean())
                    unprec_pop_std_20richest.append(
                        df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                        (df_vulnerability_e['birth_year']==by)&\
                            (df_vulnerability_e['qntl']<=1)&\
                                    (df_vulnerability_e['GMT']==GMT)][e].std()
                    )  
                    
                    # t test for difference between rich and poor samples
                    ttest_20pc_poor = ttest_rel(
                        a=np.concatenate((poor_unprec_20pci[poor_unprec_20pci.notnull()].values,poor_unprec_20pcii[poor_unprec_20pcii.notnull()].values)),
                        b=np.concatenate((rich_unprec_20pci[rich_unprec_20pci.notnull()].values,rich_unprec_20pcii[rich_unprec_20pcii.notnull()].values)),
                        alternative='greater'
                    )  
                    ttest_20pc_pvals_poor.append(ttest_20pc_poor.pvalue)
                    ttest_20pc_rich = ttest_rel(
                        a=np.concatenate((rich_unprec_20pci[rich_unprec_20pci.notnull()].values,rich_unprec_20pcii[rich_unprec_20pcii.notnull()].values)),
                        b=np.concatenate((poor_unprec_20pci[poor_unprec_20pci.notnull()].values,poor_unprec_20pcii[poor_unprec_20pcii.notnull()].values)),
                        alternative='greater'
                    )                      
                    ttest_20pc_pvals_rich.append(ttest_20pc_rich.pvalue)       
                
                # add stuff to plotting dictionary for pickling
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_quantiles_10poorest'] = unprec_pop_quantiles_10poorest
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_std_10poorest'] = unprec_pop_std_10poorest
                d_pyramid_plot_grdi[e][GMT]['population_quantiles_10poorest'] = population_quantiles_10poorest
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_quantiles_10richest'] = unprec_pop_quantiles_10richest
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_std_10richest'] = unprec_pop_std_10richest
                d_pyramid_plot_grdi[e][GMT]['population_quantiles_10richest'] = population_quantiles_10richest
                d_pyramid_plot_grdi[e][GMT]['ttest_10pc_pvals_poor'] = ttest_10pc_pvals_poor
                d_pyramid_plot_grdi[e][GMT]['ttest_10pc_pvals_rich'] = ttest_10pc_pvals_rich
                
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_quantiles_20poorest'] = unprec_pop_quantiles_20poorest
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_std_20poorest'] = unprec_pop_std_20poorest
                d_pyramid_plot_grdi[e][GMT]['population_quantiles_20poorest'] = population_quantiles_20poorest
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_quantiles_20richest'] = unprec_pop_quantiles_20richest
                d_pyramid_plot_grdi[e][GMT]['unprec_pop_std_20richest'] = unprec_pop_std_20richest
                d_pyramid_plot_grdi[e][GMT]['population_quantiles_20richest'] = population_quantiles_20richest
                d_pyramid_plot_grdi[e][GMT]['ttest_20pc_pvals_poor'] = ttest_20pc_pvals_poor
                d_pyramid_plot_grdi[e][GMT]['ttest_20pc_pvals_rich'] = ttest_20pc_pvals_rich   
        
        
            # then run the pyramid data selection for the low vs high gmt panels (this is just to get the pvalues for these comparisons, the other data is already there)
            # note that GMT integers as string "0_20" are used in the 2nd dictionary key level (instead of "GMT" integer above)
            d_pyramid_plot_grdi[e]['0_20'] = {} # i.e. "0" is the integer for the 1.5 degree pathway, "20" is for 3.5

            ttest_20pc_pvals_poor_0_vs_20 = [] # significance tests on poor (or high deprivation) across GMT pathways
            ttest_20pc_pvals_rich_0_vs_20 = [] # significance tests on rich (or low deprivation) across GMT pathways

            for by in birth_years:
                
                # poorest 20 percent, low gmt
                poor_unprec_gmt0_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==8)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]

                poor_unprec_gmt0_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==9)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]   
                
                # poorest 20 percent, high gmt
                poor_unprec_gmt20_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==8)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]

                poor_unprec_gmt20_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==9)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]                       

                # richest 20 percent, low gmt
                rich_unprec_gmt0_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==0)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]

                rich_unprec_gmt0_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==1)&\
                            (df_vulnerability_e['GMT']==gmt_low)][e]  
                
                # richest 20 percent, high gmt
                rich_unprec_gmt20_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==0)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]

                rich_unprec_gmt20_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
                    (df_vulnerability_e['birth_year']==by)&\
                        (df_vulnerability_e['qntl']==1)&\
                            (df_vulnerability_e['GMT']==gmt_high)][e]                      

                # t test for difference between low and high gmt pathways for poor/rich populations
                ttest_20pc_poor = ttest_ind(
                    a=np.concatenate((poor_unprec_gmt20_20pci[poor_unprec_gmt20_20pci.notnull()].values,poor_unprec_gmt20_20pcii[poor_unprec_gmt20_20pcii.notnull()].values)),
                    b=np.concatenate((poor_unprec_gmt0_20pci[poor_unprec_gmt0_20pci.notnull()].values,poor_unprec_gmt0_20pcii[poor_unprec_gmt0_20pcii.notnull()].values)),
                    alternative='greater'
                )  
                ttest_20pc_pvals_poor_0_vs_20.append(ttest_20pc_poor.pvalue)
                ttest_20pc_rich = ttest_ind(
                    a=np.concatenate((rich_unprec_gmt20_20pci[rich_unprec_gmt20_20pci.notnull()].values,rich_unprec_gmt20_20pcii[rich_unprec_gmt20_20pcii.notnull()].values)),
                    b=np.concatenate((rich_unprec_gmt0_20pci[rich_unprec_gmt0_20pci.notnull()].values,rich_unprec_gmt0_20pcii[rich_unprec_gmt0_20pcii.notnull()].values)),
                    alternative='greater'
                )                      
                ttest_20pc_pvals_rich_0_vs_20.append(ttest_20pc_rich.pvalue) 
                
            d_pyramid_plot_grdi[e]['0_20']['ttest_20pc_pvals_poor'] = ttest_20pc_pvals_poor_0_vs_20
            d_pyramid_plot_grdi[e]['0_20']['ttest_20pc_pvals_rich'] = ttest_20pc_pvals_rich_0_vs_20        
                    
        with open('./data/{}/pyramid_data_grdi.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(d_pyramid_plot_grdi,f)   
                
    else:
        
        print("grdi pickle already there, delete pyramid_data_grdi.pkl and rerun function if you want to renew this data")                         
            
                        
#%% ----------------------------------------------------------------
# actual function for plotting pyramid stuff
def pyramid_plot(
    flags,
    df_GMT_strj,
    vln_type,
):
    
    per_x=5 # every how many years do we plot (i.e. 1960,1970,1980,...2020 on y axis would be "10")
    height=4 # thickness of bars
    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_cp=12 # "cp" for "current pathway"; 12 or 17
    GMT_low=0
    GMT_high=20
    # plot type (will get removed and looped outside function)
    # vln_type='grdi'
    fontcolor='gray'
    # bbox for legend
    x0 = 0.1
    y0 = 0.8
    xlen = 0.2
    ylen = 0.3
    cmap = ['darkgoldenrod','forestgreen']  
    # space between legend entries
    legend_entrypad = 0.5
    # length per legend entry
    legend_entrylen = 0.75
    # more legend stuff
    legend_font = 10
    legend_lw=3.5      
    unit='pple'
    
    
    # pick GMT
    GMT = GMT_cp 
    
    # start with GDP ====================================================================================
    with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
        d_pyramid_plot = pk.load(f)    
    
    for e in extremes:    
        # for qntl_range in ('10', '20'):
        qntl_range = '20'
        
        poor_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}poorest'.format(qntl_range)]) # "_a" for panel "a"
        poor_std = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_std_{}poorest'.format(qntl_range)])
        poor_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}poorest'.format(qntl_range)])
        rich_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
        rich_std = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_std_{}richest'.format(qntl_range)])
        rich_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}richest'.format(qntl_range)])
        pvalues_poor = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_poor'.format(qntl_range)])
        pvalues_rich = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_rich'.format(qntl_range)])
        
        # labels
        if qntl_range == '10':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-4,-8,-12]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["4","8","12"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [4,8,12]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]                               
            
        elif qntl_range == '20':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-5,-10,-15,-20,-25]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["5","10","15","20","25"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [5,10,15,20,25]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]  
                
        print('{}-{}'.format(e,str(df_GMT_strj.loc[2100,GMT])))
        f,(ax1,ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(6,6),
            sharey=True,
        )
        f.subplots_adjust(wspace=0)
        # full population poor quantile (gdp)
        # if unit == 'pple':
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_pop[::-1*per_x]],
            height=height,
            color='darkgoldenrod',
            zorder=1,
            alpha=0.3
        )
            
        # unprec population poor quantile (gdp)
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_unprec[::-1*per_x]],
            height=height,
            color='darkgoldenrod',
            zorder=1,
            xerr=[i * -1 for i in poor_std[::-1*per_x]],
        )    
        # # ax1.set_xlabel('Millions of people')
        # if unit == 'pple':
        variable = 'Millions of people'
        # else:
        #     variable = 'Percentage of cohort'
        ax1.text(
            x=1,y=-0.1,
            s=variable,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax1.transAxes,
            fontsize=10,
            color='gray',
        )
        ax1.set_xticks(
            ticks=ax_xts['ax1_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
        )
        ax1.set_ylabel(
            "Birth year",
            fontsize=10,
            labelpad=5,
            color=fontcolor,
        )
        # ax1.text(
        #     x=0.5,y=1.1,
        #     s=ax1_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax1.transAxes,
        #     fontsize=10
        # )    
        
        # full population rich quantile (grdi)
        # if unit == 'pple':
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=rich_pop[::-1*per_x],
            height=height,
            color='forestgreen',
            zorder=1,
            alpha=0.3
        ) 
        # unprec population rich quantile (grdi)
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=rich_unprec[::-1*per_x],
            height=height,
            color='forestgreen',
            zorder=1,
            xerr=[i * -1 for i in rich_std[::-1*per_x]],
            
        )     
        ax2.tick_params(left=False)
        ax2.set_xticks(
            ticks=ax_xts['ax2_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
            color=fontcolor
        )     
        
        # legend stuff
        if vln_type == 'gdp':
            ax1_title = 'Poorest {}% in \n lifetime mean GDP'.format(qntl_range)
            ax2_title = 'Richest {}% in \n lifetime mean GDP'.format(qntl_range)
        elif vln_type == 'grdi':
            ax1_title = '{}% highest deprivation'.format(qntl_range) 
            ax2_title = '{}% lowest deprivation'.format(qntl_range)
        # ax2.text(
        #     x=0.5,y=1.1,
        #     s=ax2_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax2.transAxes,
        #     fontsize=10
        # )        
        
        # legend stuff
        legendcols = cmap
        handles = [
            Rectangle((0,0),1,1,color=legendcols[0]),
            Rectangle((0,0),1,1,color=legendcols[1]),
        ]

        labels= [
            ax1_title,
            ax2_title
        ]

        # ax1.legend(
        #     handles, 
        #     labels, 
        #     bbox_to_anchor=(x0, y0, xlen, ylen), 
        #     loc='upper right',
        #     ncol=1,
        #     fontsize=legend_font, 
        #     labelcolor=fontcolor,
        #     mode="expand", 
        #     borderaxespad=0.,\
        #     frameon=False, 
        #     columnspacing=0.05, 
        #     handlelength=legend_entrylen, 
        #     handletextpad=legend_entrypad
        # )                
        
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        for ax in (ax1,ax2):
            ax.tick_params(colors='gray')
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
        
        # plot asterisk or star 1/4 through the bar (0.25 * unprec number as position) if significant differenc and if given side is bigger
        # also plot percentage numbers midway through the bars
        poor_unprec_pct = poor_unprec / poor_pop * 100
        # poor_std_pct = poor_std / poor_pop * 100
        rich_unprec_pct = rich_unprec / rich_pop * 100
        # rich_std_pct = rich_std / rich_pop * 100
        
        for i,by in enumerate(birth_years[::-1*per_x]):
            # asterisks for significance
            if pvalues_poor[::-1*per_x][i] < sl:
                ax1.plot(
                    poor_unprec[::-1*per_x][i]/4 * -1,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,
                    color='k',
                )
            if pvalues_rich[::-1*per_x][i] < sl:
                ax2.plot(
                    rich_unprec[::-1*per_x][i]/4,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,  
                    color='k',                  
                )             
            # percentages
            # left side / poor
            if poor_unprec_pct[::-1*per_x][i] > 50:
                x_poor = poor_unprec[::-1*per_x][i] * -2/3 #+ poor_std[::-1*per_x][i]*4
            elif poor_unprec_pct[::-1*per_x][i] < 50:
                x_poor = poor_pop[::-1*per_x][i] * -0.6
            ax1.text(
                x=x_poor,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transData,
                fontsize=9,
                color='k'
            ) 
            # xerr=[i * -1 for i in poor_std[::-1*per_x]]
            # right side / rich
            if rich_unprec_pct[::-1*per_x][i] > 50:
                x_rich = rich_unprec[::-1*per_x][i] * 2/3#- rich_std[::-1*per_x][i]*2
            elif rich_unprec_pct[::-1*per_x][i] < 50:
                x_rich = rich_pop[::-1*per_x][i] * 0.6#- rich_std[::-1*per_x][i]*2
            ax2.text(
                x=x_rich,
                y=by,
                s='{}%'.format(str(int(np.round(rich_unprec_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transData,
                fontsize=9,
                color='k'
            )       
            # xerr=[i * -1 for i in rich_std[::-1*per_x]]      
        ax1.invert_yaxis() # only have to do this once because because y axis are shared
        f.savefig(
            './figures/pyramid/inverted/vln_pyramid_{}_{}_{}_{}_{}.png'.format(vln_type,e,str(df_GMT_strj.loc[2100,GMT]),qntl_range,unit),
            dpi=1000,
            bbox_inches='tight',
        )
        plt.show()

#%% ----------------------------------------------------------------
# actual function for plotting pyramid stuff
def pyramid_subplot(
    flags,
    df_GMT_strj,
    vln_type,
    f,
    ax1,
    ax2
):
    
    ax1.set_title(
        'b',
        fontweight='bold',
        loc='left'
    )    
    ax1.set_title(
        'Current policies',
        fontweight='bold',
        loc='right',
        color='gray',
        x=1.35
    )      
    
    per_x=5 # every how many years do we plot (i.e. 1960,1970,1980,...2020 on y axis would be "10")
    height=4 # thickness of bars
    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_cp=12 # "cp" for "current pathway"; 12 or 17
    GMT_low=0
    GMT_high=20
    # plot type (will get removed and looped outside function)
    # vln_type='grdi'
    fontcolor='gray'
    # bbox for legend
    x0 = 0.1
    y0 = 0.8
    xlen = 0.2
    ylen = 0.3
    cmap = ['darkgoldenrod','forestgreen']  
    # space between legend entries
    legend_entrypad = 0.5
    # length per legend entry
    legend_entrylen = 0.75
    # more legend stuff
    legend_font = 10
    legend_lw=3.5      
    unit='pple'
    
    
    # pick GMT
    GMT = GMT_cp 
    
    # start with GDP ====================================================================================
    with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
        d_pyramid_plot = pk.load(f)    
    
    for e in extremes:    
        # for qntl_range in ('10', '20'):
        qntl_range = '20'
        
        poor_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}poorest'.format(qntl_range)]) # "_a" for panel "a"
        poor_std = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_std_{}poorest'.format(qntl_range)])
        poor_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}poorest'.format(qntl_range)])
        rich_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
        rich_std = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_std_{}richest'.format(qntl_range)])
        rich_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}richest'.format(qntl_range)])
        pvalues_poor = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_poor'.format(qntl_range)])
        pvalues_rich = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_rich'.format(qntl_range)])
        
        # labels
        if qntl_range == '10':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-4,-8,-12]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["4","8","12"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [4,8,12]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]                               
            
        elif qntl_range == '20':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-5,-10,-15,-20,-25]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["5","10","15","20","25"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [5,10,15,20,25]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]  
                
        print('{}-{}'.format(e,str(df_GMT_strj.loc[2100,GMT])))
        # f,(ax1,ax2) = plt.subplots(
        #     nrows=1,
        #     ncols=2,
        #     figsize=(6,6),
        #     sharey=True,
        # )
        # f.subplots_adjust(wspace=0)
        # full population poor quantile (gdp)
        # if unit == 'pple':
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_pop[::-1*per_x]],
            height=height,
            color='darkgoldenrod',
            zorder=1,
            alpha=0.3
        )
            
        # unprec population poor quantile (gdp)
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_unprec[::-1*per_x]],
            height=height,
            color='darkgoldenrod',
            zorder=1,
            xerr=[i * -1 for i in poor_std[::-1*per_x]],
        )    
        # # ax1.set_xlabel('Millions of people')
        # if unit == 'pple':
        variable = 'Millions of people'
        # else:
        #     variable = 'Percentage of cohort'
        ax1.text(
            x=1,y=-0.15,
            s=variable,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax1.transAxes,
            fontsize=10,
            color='gray',
        )
        
        ax1.tick_params(left=False,right=False)
        ax1.set_xticks(
            ticks=ax_xts['ax1_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
        )
        ax2.set_ylabel(
            "Birth year",
            fontsize=10,
            labelpad=5,
            color=fontcolor,
        )
        
        # full population rich quantile (grdi)
        # if unit == 'pple':
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=rich_pop[::-1*per_x],
            height=height,
            color='forestgreen',
            zorder=1,
            alpha=0.3
        ) 
        # unprec population rich quantile (grdi)
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=rich_unprec[::-1*per_x],
            height=height,
            color='forestgreen',
            zorder=1,
            xerr=[i * -1 for i in rich_std[::-1*per_x]],
            
        )     
        ax2.tick_params(left=False,right=True)
        ax2.set_xticks(
            ticks=ax_xts['ax2_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
            color=fontcolor
        )    
        ax2.set_yticks(
            birth_years[::-1*per_x*2]
        )
        ax1.set_yticks([])        
        ax1.set_yticklabels([])        
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')                
        
        # legend stuff
        if vln_type == 'gdp':
            ax1_title = 'Poorest {}% in \n lifetime mean GDP'.format(qntl_range)
            ax2_title = 'Richest {}% in \n lifetime mean GDP'.format(qntl_range)
        elif vln_type == 'grdi':
            ax1_title = '{}% highest deprivation'.format(qntl_range) 
            ax2_title = '{}% lowest deprivation'.format(qntl_range)
        # ax2.text(
        #     x=0.5,y=1.1,
        #     s=ax2_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax2.transAxes,
        #     fontsize=10
        # )        
        
        # legend stuff
        legendcols = cmap
        handles = [
            Rectangle((0,0),1,1,color=legendcols[0]),
            Rectangle((0,0),1,1,color=legendcols[1]),
        ]

        labels= [
            ax1_title,
            ax2_title
        ]

        # ax1.legend(
        #     handles, 
        #     labels, 
        #     bbox_to_anchor=(x0, y0, xlen, ylen), 
        #     loc='upper right',
        #     ncol=1,
        #     fontsize=legend_font, 
        #     labelcolor=fontcolor,
        #     mode="expand", 
        #     borderaxespad=0.,\
        #     frameon=False, 
        #     columnspacing=0.05, 
        #     handlelength=legend_entrylen, 
        #     handletextpad=legend_entrypad
        # )                
        
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # for ax in (ax1,ax2):
        ax1.tick_params(colors='gray')
        ax2.tick_params(colors='gray')
        ax1.spines['bottom'].set_color('gray')
        ax2.spines['bottom'].set_color('gray')
        ax2.spines['right'].set_color('gray')
        ax2.spines['left'].set_color('gray')
        
        # plot asterisk or star 1/4 through the bar (0.25 * unprec number as position) if significant differenc and if given side is bigger
        # also plot percentage numbers midway through the bars
        poor_unprec_pct = poor_unprec / poor_pop * 100
        # poor_std_pct = poor_std / poor_pop * 100
        rich_unprec_pct = rich_unprec / rich_pop * 100
        # rich_std_pct = rich_std / rich_pop * 100
        
        for i,by in enumerate(birth_years[::-1*per_x]):
            # asterisks for significance
            if pvalues_poor[::-1*per_x][i] < sl:
                ax1.plot(
                    poor_unprec[::-1*per_x][i]/4 * -1,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,
                    color='k',
                )
            if pvalues_rich[::-1*per_x][i] < sl:
                ax2.plot(
                    rich_unprec[::-1*per_x][i]/4,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,  
                    color='k',                  
                )             
            # percentages
            # left side / poor
            if poor_unprec_pct[::-1*per_x][i] > 50:
                x_poor = poor_unprec[::-1*per_x][i] * -2/3 #+ poor_std[::-1*per_x][i]*4
            elif poor_unprec_pct[::-1*per_x][i] < 50:
                x_poor = poor_pop[::-1*per_x][i] * -0.6
            ax1.text(
                x=x_poor,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transData,
                fontsize=9,
                color='k'
            ) 
            # xerr=[i * -1 for i in poor_std[::-1*per_x]]
            # right side / rich
            if rich_unprec_pct[::-1*per_x][i] > 50:
                x_rich = rich_unprec[::-1*per_x][i] * 2/3#- rich_std[::-1*per_x][i]*2
            elif rich_unprec_pct[::-1*per_x][i] < 50:
                x_rich = rich_pop[::-1*per_x][i] * 0.6#- rich_std[::-1*per_x][i]*2
            ax2.text(
                x=x_rich,
                y=by,
                s='{}%'.format(str(int(np.round(rich_unprec_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transData,
                fontsize=9,
                color='k'
            )       
            # xerr=[i * -1 for i in rich_std[::-1*per_x]]      
        ax1.invert_yaxis() # only have to do this once because because y axis are shared
        ax2.invert_yaxis() # only have to do this once because because y axis are shared
        # f.savefig(
        #     './figures/pyramid/inverted/vln_pyramid_{}_{}_{}_{}_{}.png'.format(vln_type,e,str(df_GMT_strj.loc[2100,GMT]),qntl_range,unit),
        #     dpi=1000,
        #     bbox_inches='tight',
        # )
        # plt.show()


# %% ==============================================================================================
# d_pyramid_plot_grdi needs cross-GMT p-values (to see if 1.5 vs 3.5 has significantly higher arithmetic mean)
# I think this was just a testing cell, so I comment it out (ds_vulnerability was causing errors because it isn't read in)
# v='grdi_q_by_p'
# e='heatwavedarea'
# gmt_low=0
# gmt_high=20
# df_vulnerability = ds_vulnerability.to_dataframe().reset_index()      
# df_vulnerability_e = df_vulnerability.loc[:,['run','GMT','qntl','vulnerability_index','birth_year',e]]
# df_vulnerability_e.loc[:,e] = df_vulnerability_e.loc[:,e] / 10**6 # convert to millions of people 

# # load pyramid plot data
# with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
#     d_pyramid_plot = pk.load(f)    
# d_pyramid_plot_grdi = d_pyramid_plot

   
# d_pyramid_plot_grdi[e]['0_20'] = {} # i.e. "0" is the integer for the 1.5 degree pathway, "20" is for 3.5

# ttest_20pc_pvals_poor_0_vs_20 = [] # significance tests on poor (or high deprivation) 
# ttest_20pc_pvals_rich_0_vs_20 = [] # significance tests on rich (or low deprivation) 

# for by in birth_years:
    
#     # poorest 20 percent, low gmt
#     poor_unprec_gmt0_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==8)&\
#                 (df_vulnerability_e['GMT']==gmt_low)][e]

#     poor_unprec_gmt0_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==9)&\
#                 (df_vulnerability_e['GMT']==gmt_low)][e]   
    
#     # poorest 20 percent, high gmt
#     poor_unprec_gmt20_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==8)&\
#                 (df_vulnerability_e['GMT']==gmt_high)][e]

#     poor_unprec_gmt20_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==9)&\
#                 (df_vulnerability_e['GMT']==gmt_high)][e]                       

#     # richest 20 percent, low gmt
#     rich_unprec_gmt0_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==0)&\
#                 (df_vulnerability_e['GMT']==gmt_low)][e]

#     rich_unprec_gmt0_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==1)&\
#                 (df_vulnerability_e['GMT']==gmt_low)][e]  
    
#     # richest 20 percent, high gmt
#     rich_unprec_gmt20_20pci = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==0)&\
#                 (df_vulnerability_e['GMT']==gmt_high)][e]

#     rich_unprec_gmt20_20pcii = df_vulnerability_e[(df_vulnerability_e['vulnerability_index']==v)&\
#         (df_vulnerability_e['birth_year']==by)&\
#             (df_vulnerability_e['qntl']==1)&\
#                 (df_vulnerability_e['GMT']==gmt_high)][e]                      

#     # t test for difference between low and high gmt pathways for poor/rich populations
#     ttest_20pc_poor = ttest_ind(
#         a=np.concatenate((poor_unprec_gmt20_20pci[poor_unprec_gmt20_20pci.notnull()].values,poor_unprec_gmt20_20pcii[poor_unprec_gmt20_20pcii.notnull()].values)),
#         b=np.concatenate((poor_unprec_gmt0_20pci[poor_unprec_gmt0_20pci.notnull()].values,poor_unprec_gmt0_20pcii[poor_unprec_gmt0_20pcii.notnull()].values)),
#         alternative='greater'
#     )  
#     ttest_20pc_pvals_poor_0_vs_20.append(ttest_20pc_poor.pvalue)
#     ttest_20pc_rich = ttest_rel(
#         a=np.concatenate((rich_unprec_gmt20_20pci[rich_unprec_gmt20_20pci.notnull()].values,rich_unprec_gmt20_20pcii[rich_unprec_gmt20_20pcii.notnull()].values)),
#         b=np.concatenate((rich_unprec_gmt0_20pci[rich_unprec_gmt0_20pci.notnull()].values,rich_unprec_gmt0_20pcii[rich_unprec_gmt0_20pcii.notnull()].values)),
#         alternative='greater'
#     )                      
#     ttest_20pc_pvals_rich_0_vs_20.append(ttest_20pc_rich.pvalue) 
    
# d_pyramid_plot_grdi[e][GMT]['ttest_20pc_pvals_poor'] = ttest_20pc_pvals_poor_0_vs_20
# d_pyramid_plot_grdi[e][GMT]['ttest_20pc_pvals_rich'] = ttest_20pc_pvals_rich_0_vs_20    

# %%            
# map testing for panel showing all the quantiles
# in grdi, poor is high integers (8 & 9), in gdp, rich is low integers (0 & 1)
def pyramid_map(
    vln_type,
    ds_grdi_qntls,
    ds_gdp_qntls,
    da_cohort_size_1960_2020,
    gdf_robinson_bounds,
):
    # vln_type = 'gdp'
    fontcolor='gray'
    if vln_type == 'grdi':
        qp_i = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=8,birth_year=2020) #"qp" for "quantile poor", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qp_i = xr.where(qp_i.notnull(),1,0)
        qp_ii = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=9,birth_year=2020)
        qp_ii = xr.where(qp_ii.notnull(),1,0)
        qp = qp_i + qp_ii
        qp = qp.where(qp!=0) # poor == 1

        qr_i = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=0,birth_year=2020) #"qr" for "quantile rich", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qr_i = xr.where(qr_i.notnull(),1,0)
        qr_ii = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=1,birth_year=2020)
        qr_ii = xr.where(qr_ii.notnull(),1,0).squeeze()
        qr = qr_i + qr_ii    
        qr = qr.where(qr!=0)*2 # rich == 2
    elif vln_type == 'gdp':
        qp_i = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=0,birth_year=2020) #"qp" for "quantile poor", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qp_i = xr.where(qp_i.notnull(),1,0)
        qp_ii = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=1,birth_year=2020)
        qp_ii = xr.where(qp_ii.notnull(),1,0)
        qp = qp_i + qp_ii
        qp = qp.where(qp!=0) # poor == 1

        qr_i = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=8,birth_year=2020) #"qr" for "quantile rich", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qr_i = xr.where(qr_i.notnull(),1,0)
        qr_ii = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=9,birth_year=2020)
        qr_ii = xr.where(qr_ii.notnull(),1,0).squeeze()
        qr = qr_i + qr_ii    
        qr = qr.where(qr!=0)*2 # rich == 2    
        
    # get population sizes per pixel for each quantile group. Will add this to geodataframe
    da_cohort_size_1960_2020_q_r = da_cohort_size_1960_2020.loc[{'birth_year':2020}].where(qr.notnull())
    da_cohort_size_1960_2020_q_p = da_cohort_size_1960_2020.loc[{'birth_year':2020}].where(qp.notnull())
    all_data = da_cohort_size_1960_2020.loc[{'birth_year':2020}]
    vmin = 1
    vmax = all_data.max()
    lognorm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)    

    # should convert pixels to points via geodataframe
    # first do for "poor"
    df_p = qp.to_dataframe().reset_index()
    gdf_p = gpd.GeoDataFrame(
        df_p['{}_q_by_p'.format(vln_type)], geometry=gpd.points_from_xy(df_p.lon,df_p.lat)
    )
    gdf_p.set_crs(epsg = "4326",inplace=True)
    gdf_p.loc[:,'{}_q_by_p'.format(vln_type)] = da_cohort_size_1960_2020_q_p.to_dataframe().reset_index()['mask']
    
    # then do for "rich"
    df_r = qr.to_dataframe().reset_index()
    gdf_r = gpd.GeoDataFrame(
        df_r['{}_q_by_p'.format(vln_type)], geometry=gpd.points_from_xy(df_r.lon,df_r.lat)
    )
    gdf_r.set_crs(epsg = "4326",inplace=True)     
    gdf_r.loc[:,'{}_q_by_p'.format(vln_type)] = da_cohort_size_1960_2020_q_r.to_dataframe().reset_index()['mask']   
    
    # get bounds
    robinson = ccrs.Robinson().proj4_init
    gdf_robinson_bounds_v1 = gdf_p.to_crs(robinson).total_bounds # (minx,miny,maxx,maxy) will use this for xlim
    gdf_p = gdf_p.dropna()
    gdf_r = gdf_r.dropna()
    
    # plot
    f,ax = plt.subplots(
        ncols=1,
        nrows=1,
        subplot_kw={'projection':ccrs.Robinson()},
        transform=ccrs.PlateCarree()
    )
    ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    p_poor=gdf_p.to_crs(robinson).plot(
        ax=ax,
        column='{}_q_by_p'.format(vln_type),
        cmap='YlOrBr',
        zorder=5,
        norm=lognorm,
        markersize=gdf_p['{}_q_by_p'.format(vln_type)]/1000,
    )    
    p_rich=gdf_r.to_crs(robinson).plot(
        ax=ax,
        column='{}_q_by_p'.format(vln_type),
        cmap='Greens',
        zorder=4,
        norm=lognorm,
        markersize=gdf_r['{}_q_by_p'.format(vln_type)]/1000,
    )       
    ax.set_xlim(gdf_robinson_bounds_v1[0],gdf_robinson_bounds_v1[2])
    ax.set_ylim(gdf_robinson_bounds[1],gdf_robinson_bounds[3])      

    cmap_browns = plt.cm.get_cmap('YlOrBr')
    cmap_greens = plt.cm.get_cmap('Greens') 

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75
    legend_font = 10
    legend_lw=3.5   

    # new handles
    legend_sizes = [1000, 10000, 100000]
    population_labels = ['1k', '10k', '100k population']

    handles = [
        (Line2D(
            [0], [0],
            marker='o', color='w', 
            markerfacecolor=cmap_type(lognorm(1000)), 
            markersize=np.sqrt(1000/1000),
            markeredgecolor=cmap_type(lognorm(1000)),
        ),
        Line2D(
            [0], [0],
            marker='o', color='w', 
            markerfacecolor=cmap_type(lognorm(1000*10)), 
            markersize=np.sqrt(1000*10/1000),
            markeredgecolor=cmap_type(lognorm(1000*10)),
        ),
        Line2D(
            [0], [0],
            marker='o', color='w', 
            markerfacecolor=cmap_type(lognorm(1000*100)), 
            markersize=np.sqrt(1000*100/1000),
            markeredgecolor=cmap_type(lognorm(1000*100)),
        ))
        for cmap_type in [cmap_browns,cmap_greens]
    ]    
    
    if vln_type == 'grdi':
        legend_labels= [
            '20% highest deprivation',
            '20% lowest deprivation'
        ]
    elif vln_type == 'gdp':
        legend_labels= [
            '20% lowest GDP',
            '20% highest GDP'
        ]        
            
    x0 = 0.
    y0 = -0.4
    xlen = 0.6
    ylen = 0.8

    leg = ax.legend(
        handles=handles, 
        labels=legend_labels,
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'lower left',
        fontsize=legend_font, 
        labelcolor=fontcolor,
        handletextpad=0.8,
        handlelength=7,
        # ncol=3,
        # columnspacing=0.5,
        markerscale=1,
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        handler_map={tuple: HandlerTuple(ndivide=None)}
    )    
    
    # label pop sizes in legend
    for i,l in enumerate(population_labels):
        if i == 0:
            ax.annotate(
                l,
                (0.03,-0.15),
                xycoords=ax.transAxes,
                fontsize=legend_font,
                rotation='horizontal',
                color=fontcolor,
            )       
        if i == 1:
            ax.annotate(
                l,
                (0.09,-0.15),
                xycoords=ax.transAxes,
                fontsize=legend_font,
                rotation='horizontal',
                color=fontcolor,
            )                     
        if i == 2:
            ax.annotate(
                l,
                (0.165,-0.15),
                xycoords=ax.transAxes,
                fontsize=legend_font,
                rotation='horizontal',
                color=fontcolor,
            )                     
    f.savefig(
        './figures/pyramid/inverted/vln_map_{}.png'.format(vln_type),
        dpi=1000,
        bbox_inches='tight',
    )
    plt.show()      
    
# %%            
# map testing for panel showing all the quantiles
# in grdi, poor is high integers (8 & 9), in gdp, rich is low integers (0 & 1)
def pyramid_map_subplot(
    vln_type,
    ds_grdi_qntls,
    ds_gdp_qntls,
    da_cohort_size_1960_2020,
    gdf_robinson_bounds,
    f,
    ax
):

    # putting all my pyramid plot shit together
    # vln_type='grdi'
    
    ax.set_title(
        'a',
        fontweight='bold',
        loc='left'
    )

    #--------------------------------
    # map

    # vln_type = 'gdp'
    fontcolor='gray'
    if vln_type == 'grdi':
        qp_i = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=8,birth_year=2020) #"qp" for "quantile poor", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qp_i = xr.where(qp_i.notnull(),1,0)
        qp_ii = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=9,birth_year=2020)
        qp_ii = xr.where(qp_ii.notnull(),1,0)
        qp = qp_i + qp_ii
        qp = qp.where(qp!=0) # poor == 1

        qr_i = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=0,birth_year=2020) #"qr" for "quantile rich", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qr_i = xr.where(qr_i.notnull(),1,0)
        qr_ii = ds_grdi_qntls['grdi_q_by_p'].sel(qntl=1,birth_year=2020)
        qr_ii = xr.where(qr_ii.notnull(),1,0).squeeze()
        qr = qr_i + qr_ii    
        qr = qr.where(qr!=0)*2 # rich == 2
    elif vln_type == 'gdp':
        qp_i = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=0,birth_year=2020) #"qp" for "quantile poor", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qp_i = xr.where(qp_i.notnull(),1,0)
        qp_ii = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=1,birth_year=2020)
        qp_ii = xr.where(qp_ii.notnull(),1,0)
        qp = qp_i + qp_ii
        qp = qp.where(qp!=0) # poor == 1

        qr_i = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=8,birth_year=2020) #"qr" for "quantile rich", "_i" for first 10 percentiles, "__i" for next 10 percentiles
        qr_i = xr.where(qr_i.notnull(),1,0)
        qr_ii = ds_gdp_qntls['gdp_q_by_p'].sel(qntl=9,birth_year=2020)
        qr_ii = xr.where(qr_ii.notnull(),1,0).squeeze()
        qr = qr_i + qr_ii    
        qr = qr.where(qr!=0)*2 # rich == 2    
        
    # get population sizes per pixel for each quantile group. Will add this to geodataframe
    da_cohort_size_1960_2020_q_r = da_cohort_size_1960_2020.loc[{'birth_year':2020}].where(qr.notnull())
    da_cohort_size_1960_2020_q_p = da_cohort_size_1960_2020.loc[{'birth_year':2020}].where(qp.notnull())
    all_data = da_cohort_size_1960_2020.loc[{'birth_year':2020}]
    vmin = 1
    vmax = all_data.max()
    lognorm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)    

    # should convert pixels to points via geodataframe
    # first do for "poor"
    df_p = qp.to_dataframe().reset_index()
    gdf_p = gpd.GeoDataFrame(
        df_p['{}_q_by_p'.format(vln_type)], geometry=gpd.points_from_xy(df_p.lon,df_p.lat)
    )
    gdf_p.set_crs(epsg = "4326",inplace=True)
    gdf_p.loc[:,'{}_q_by_p'.format(vln_type)] = da_cohort_size_1960_2020_q_p.to_dataframe().reset_index()['mask']

    # then do for "rich"
    df_r = qr.to_dataframe().reset_index()
    gdf_r = gpd.GeoDataFrame(
        df_r['{}_q_by_p'.format(vln_type)], geometry=gpd.points_from_xy(df_r.lon,df_r.lat)
    )
    gdf_r.set_crs(epsg = "4326",inplace=True)     
    gdf_r.loc[:,'{}_q_by_p'.format(vln_type)] = da_cohort_size_1960_2020_q_r.to_dataframe().reset_index()['mask']   

    # get bounds
    robinson = ccrs.Robinson().proj4_init
    gdf_robinson_bounds_v1 = gdf_p.to_crs(robinson).total_bounds # (minx,miny,maxx,maxy) will use this for xlim
    gdf_p = gdf_p.dropna()
    gdf_r = gdf_r.dropna()

    # plot
    ax.add_feature(feature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='powderblue', facecolor='powderblue'))
    p_poor=gdf_p.to_crs(robinson).plot(
        ax=ax,
        column='{}_q_by_p'.format(vln_type),
        cmap='YlOrBr',
        zorder=5,
        norm=lognorm,
        markersize=gdf_p['{}_q_by_p'.format(vln_type)]/1000,
    )    
    p_rich=gdf_r.to_crs(robinson).plot(
        ax=ax,
        column='{}_q_by_p'.format(vln_type),
        cmap='Greens',
        zorder=4,
        norm=lognorm,
        markersize=gdf_r['{}_q_by_p'.format(vln_type)]/1000,
    )       
    ax.set_xlim(gdf_robinson_bounds_v1[0],gdf_robinson_bounds_v1[2])
    ax.set_ylim(gdf_robinson_bounds[1],gdf_robinson_bounds[3])      

    cmap_browns = plt.cm.get_cmap('YlOrBr')
    cmap_greens = plt.cm.get_cmap('Greens') 

    # space between entries
    legend_entrypad = 0.5

    # length per entry
    legend_entrylen = 0.75
    legend_font = 10
    legend_lw=3.5   

    # new handles
    legend_sizes = [1000, 10000, 100000]
    population_labels = ['1k', '10k', '100k population']

    handles = [
        (Line2D(
            [0], [0],
            marker='o', color='w', 
            markerfacecolor=cmap_type(lognorm(1000)), 
            markersize=np.sqrt(1000/1000),
            markeredgecolor=cmap_type(lognorm(1000)),
        ),
        Line2D(
            [0], [0],
            marker='o', color='w', 
            markerfacecolor=cmap_type(lognorm(1000*10)), 
            markersize=np.sqrt(1000*10/1000),
            markeredgecolor=cmap_type(lognorm(1000*10)),
        ),
        Line2D(
            [0], [0],
            marker='o', color='w', 
            markerfacecolor=cmap_type(lognorm(1000*100)), 
            markersize=np.sqrt(1000*100/1000),
            markeredgecolor=cmap_type(lognorm(1000*100)),
        ))
        for cmap_type in [cmap_browns,cmap_greens]
    ]    

    if vln_type == 'grdi':
        legend_labels= [
            '20% highest deprivation',
            '20% lowest deprivation'
        ]
    elif vln_type == 'gdp':
        legend_labels= [
            '20% lowest GDP',
            '20% highest GDP'
        ]        
            
    x0 = 0.
    y0 = -0.5
    xlen = 0.6
    ylen = 0.8

    leg = ax.legend(
        handles=handles, 
        labels=legend_labels,
        bbox_to_anchor=(x0, y0, xlen, ylen), 
        loc = 'lower left',
        fontsize=legend_font, 
        labelcolor=fontcolor,
        handletextpad=0.8,
        handlelength=7,
        # ncol=3,
        # columnspacing=0.5,
        markerscale=1,
        mode="expand", 
        borderaxespad=0.,\
        frameon=False, 
        handler_map={tuple: HandlerTuple(ndivide=None)}
    )    

    # label pop sizes in legend
    for i,l in enumerate(population_labels):
        if i == 0:
            ax.annotate(
                l,
                (0.03,-0.15),
                xycoords=ax.transAxes,
                fontsize=legend_font,
                rotation='horizontal',
                color=fontcolor,
            )       
        if i == 1:
            ax.annotate(
                l,
                (0.09,-0.15),
                xycoords=ax.transAxes,
                fontsize=legend_font,
                rotation='horizontal',
                color=fontcolor,
            )                     
        if i == 2:
            ax.annotate(
                l,
                (0.165,-0.15),
                xycoords=ax.transAxes,
                fontsize=legend_font,
                rotation='horizontal',
                color=fontcolor,
            )           

# %%
# testing pyramids for left/right = poor(1.5 pathway)/poor(3.5 pathway) & same for rich
def pyramid_poor_lowhigh(
    flags,
    df_GMT_strj,
    vln_type,
):

    per_x=5 # every how many years do we plot (i.e. 1960,1970,1980,...2020 on y axis would be "10")
    height=4 # thickness of bars
    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_cp=12 # "cp" for "current pathway"; 12 or 17
    GMT_low=0
    GMT_high=20
    # plot type (will get removed and looped outside function)
    # vln_type='gdp'
    fontcolor='gray'
    # bbox for legend
    x0 = 0.1
    y0 = 0.8
    xlen = 0.2
    ylen = 0.3
    cmap = ['steelblue','darkred']  
    # space between legend entries
    legend_entrypad = 0.5
    # length per legend entry
    legend_entrylen = 0.75
    # more legend stuff
    legend_font = 10
    legend_lw=3.5      

    # pick GMT
    GMT = GMT_cp 
    unit='pple'
    qntl_range = '20'

    # start with GDP ====================================================================================
    with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
        d_pyramid_plot = pk.load(f)    

    for e in extremes:    
        
        poor_unprec_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_quantiles_{}poorest'.format(qntl_range)])
        poor_std_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_std_{}poorest'.format(qntl_range)])
        poor_pop = np.asarray(d_pyramid_plot[e][GMT_low]['population_quantiles_{}poorest'.format(qntl_range)])
        
        poor_unprec_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_quantiles_{}poorest'.format(qntl_range)])
        poor_std_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_std_{}poorest'.format(qntl_range)])       
        
        pvalues_poor = np.asarray(d_pyramid_plot[e]['0_20']['ttest_{}pc_pvals_poor'.format(qntl_range)]) # pvalues for whether high GMT shows higher unrpecedented numbers that low GMT for poor
        # pvalues_rich = np.asarray(d_pyramid_plot[e]['0_20']['ttest_{}pc_pvals_rich'.format(qntl_range)]) # diddo for rich^      
        
        # d_pyramid_plot_grdi[e]['0_20']['ttest_20pc_pvals_poor'] = ttest_20pc_pvals_poor_0_vs_20
        # d_pyramid_plot_grdi[e]['0_20']['ttest_20pc_pvals_rich'] = ttest_20pc_pvals_rich_0_vs_20        
        
        # labels
        if qntl_range == '10':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-4,-8,-12]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["4","8","12"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [4,8,12]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]                               
            
        elif qntl_range == '20':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-5,-10,-15,-20,-25]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["5","10","15","20","25"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [5,10,15,20,25]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]  

        f,(ax1,ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(6,6),
            sharey=True,
        )
        f.subplots_adjust(wspace=0)
        # full population poor quantile (gdp)
        if unit == 'pple':
            ax1.barh(
                y=birth_years[::-1*per_x], # added to flip y axis
                width=[i * -1 for i in poor_pop[::-1*per_x]],
                height=height,
                color='steelblue',
                zorder=1,
                alpha=0.3
            )
            
        # unprec population poor quantile (gdp)
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_unprec_lowgmt[::-1*per_x]],
            height=height,
            color='steelblue',
            zorder=1,
            xerr=[i * -1 for i in poor_std_lowgmt[::-1*per_x]],
        )    
        # # ax1.set_xlabel('Millions of people')
        if unit == 'pple':
            variable = 'Millions of people'
        else:
            variable = 'Percentage of cohort'
        ax1.text(
            x=1,y=-0.1,
            s=variable,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax1.transAxes,
            fontsize=10,
            color='gray',
        )
        ax1.set_xticks(
            ticks=ax_xts['ax1_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
        )
        ax1.set_ylabel(
            "Birth year",
            fontsize=10,
            labelpad=5,
            color=fontcolor,
        )
        # ax1.text(
        #     x=0.5,y=1.1,
        #     s=ax1_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax1.transAxes,
        #     fontsize=10
        # )    
        
        # full population rich quantile (grdi)
        if unit == 'pple':
            ax2.barh(
                y=birth_years[::-1*per_x],
                width=poor_pop[::-1*per_x],
                height=height,
                color='darkred',
                zorder=1,
                alpha=0.3
            ) 
        # unprec population rich quantile (grdi)
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=poor_unprec_higmt[::-1*per_x],
            height=height,
            color='darkred',
            zorder=1,
            xerr=[i * -1 for i in poor_std_higmt[::-1*per_x]],
            
        )     
        ax2.tick_params(left=False)
        ax2.set_xticks(
            ticks=ax_xts['ax2_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
            color=fontcolor
        )     
        
        # legend stuff
        ax1_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_low],1))
        ax2_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_high],1))

        # ax2.text(
        #     x=0.5,y=1.1,
        #     s=ax2_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax2.transAxes,
        #     fontsize=10
        # )        
        
        # legend stuff
        legendcols = cmap
        handles = [
            Rectangle((0,0),1,1,color=legendcols[0]),
            Rectangle((0,0),1,1,color=legendcols[1]),
        ]

        labels= [
            ax1_title,
            ax2_title
        ]

        ax1.legend(
            handles, 
            labels, 
            bbox_to_anchor=(x0, y0, xlen, ylen), 
            loc='upper right',
            ncol=1,
            fontsize=legend_font, 
            labelcolor=fontcolor,
            mode="expand", 
            borderaxespad=0.,\
            frameon=False, 
            columnspacing=0.05, 
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad
        )                
        
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        for ax in (ax1,ax2):
            ax.tick_params(colors='gray')
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
        
        # plot asterisk or star 1/4 through the bar (0.25 * unprec number as position) if significant differenc and if given side is bigger
        # also plot percentage numbers midway through the bars
        poor_unprec_lowgmt_pct = poor_unprec_lowgmt / poor_pop * 100
        poor_unprec_higmt_pct = poor_unprec_higmt / poor_pop * 100
        
        for i,by in enumerate(birth_years[::-1*per_x]):
            
            # asterisks for significance
            if pvalues_poor[::-1*per_x][i] < sl:
                ax2.plot(
                    poor_unprec_higmt[::-1*per_x][i]/4,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,  
                    color='k',                  
                )          
                
            # percentages
            # left side / low gmt
            if poor_unprec_lowgmt_pct[::-1*per_x][i] > 50:
                x_poor_lowgmt = poor_unprec_lowgmt[::-1*per_x][i] * -2/3
            elif poor_unprec_lowgmt_pct[::-1*per_x][i] < 50:
                x_poor_lowgmt = poor_pop[::-1*per_x][i] * -0.6
            ax1.text(
                x=x_poor_lowgmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_lowgmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transData,
                fontsize=9,
                color='k'
            ) 
            
            # right side / high gmt
            if poor_unprec_higmt_pct[::-1*per_x][i] > 50:
                x_poor_higmt = poor_unprec_higmt[::-1*per_x][i] * 2/3
            elif poor_unprec_higmt_pct[::-1*per_x][i] < 50:
                x_poor_higmt = poor_pop[::-1*per_x][i] * 0.6
            ax2.text(
                x=x_poor_higmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_higmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transData,
                fontsize=9,
                color='k'
            )                    
        ax1.invert_yaxis() # only have to do this once because because y axis are shared
        f.savefig(
            './figures/pyramid/inverted/vln_pyramid_poorlohi_{}_{}_{}_{}.png'.format(vln_type,e,qntl_range,unit),
            dpi=1000,
            bbox_inches='tight',
        )
        plt.show()

# %%
# testing pyramids for left/right = poor(1.5 pathway)/poor(3.5 pathway) & same for rich
def pyramid_poor_lowhigh_subplot(
    flags,
    df_GMT_strj,
    vln_type,
    f,
    ax1,
    ax2
):

    ax1.set_title(
        'c',
        fontweight='bold',
        loc='left'
    )    
    ax1.set_title(
        'High deprivation',
        fontweight='bold',
        loc='right',
        color='gray',
        x=1.4
    )          

    per_x=5 # every how many years do we plot (i.e. 1960,1970,1980,...2020 on y axis would be "10")
    height=4 # thickness of bars
    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_cp=12 # "cp" for "current pathway"; 12 or 17
    GMT_low=0
    GMT_high=20
    # plot type (will get removed and looped outside function)
    # vln_type='gdp'
    fontcolor='gray'
    # bbox for legend
    x0 = 0.1
    y0 = 0.99
    xlen = 0.2
    ylen = 0.3
    cmap = ['steelblue','darkred']  
    # space between legend entries
    legend_entrypad = 0.5
    # length per legend entry
    legend_entrylen = 0.75
    # more legend stuff
    legend_font = 10
    legend_lw=3.5      

    # pick GMT
    GMT = GMT_cp 
    unit='pple'
    qntl_range = '20'

    # start with GDP ====================================================================================
    with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
        d_pyramid_plot = pk.load(f)    

    for e in extremes:    
        
        poor_unprec_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_quantiles_{}poorest'.format(qntl_range)])
        poor_std_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_std_{}poorest'.format(qntl_range)])
        poor_pop = np.asarray(d_pyramid_plot[e][GMT_low]['population_quantiles_{}poorest'.format(qntl_range)])
        
        poor_unprec_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_quantiles_{}poorest'.format(qntl_range)])
        poor_std_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_std_{}poorest'.format(qntl_range)])       
        
        pvalues_poor = np.asarray(d_pyramid_plot[e]['0_20']['ttest_{}pc_pvals_poor'.format(qntl_range)]) # pvalues for whether high GMT shows higher unrpecedented numbers that low GMT for poor
        # pvalues_rich = np.asarray(d_pyramid_plot[e]['0_20']['ttest_{}pc_pvals_rich'.format(qntl_range)]) # diddo for rich^      
        
        # d_pyramid_plot_grdi[e]['0_20']['ttest_20pc_pvals_poor'] = ttest_20pc_pvals_poor_0_vs_20
        # d_pyramid_plot_grdi[e]['0_20']['ttest_20pc_pvals_rich'] = ttest_20pc_pvals_rich_0_vs_20        
        
        # labels
        if qntl_range == '10':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-4,-8,-12]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["4","8","12"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [4,8,12]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]                               
            
        elif qntl_range == '20':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-5,-10,-15,-20,-25]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["5","10","15","20",None]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [5,10,15,20,25]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]  

        # f,(ax1,ax2) = plt.subplots(
        #     nrows=1,
        #     ncols=2,
        #     figsize=(6,6),
        #     sharey=True,
        # )
        # f.subplots_adjust(wspace=0)
        # full population poor quantile (gdp)
        if unit == 'pple':
            ax1.barh(
                y=birth_years[::-1*per_x], # added to flip y axis
                width=[i * -1 for i in poor_pop[::-1*per_x]],
                height=height,
                color='steelblue',
                zorder=1,
                alpha=0.3
            )
            
        # unprec population poor quantile (gdp)
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_unprec_lowgmt[::-1*per_x]],
            height=height,
            color='steelblue',
            zorder=1,
            xerr=[i * -1 for i in poor_std_lowgmt[::-1*per_x]],
        )    
        # # ax1.set_xlabel('Millions of people')
        if unit == 'pple':
            variable = 'Millions of people'
        else:
            variable = 'Percentage of cohort'
        ax1.text(
            # x=1,y=-0.1,
            x=2,y=-0.15,
            s=variable,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax1.transAxes,
            fontsize=10,
            color='gray',
        )
        ax1.tick_params(left=False,right=False)
        ax1.set_xticks(
            ticks=ax_xts['ax1_xticks_{}'.format(unit)],
            # labels=ax_xts['xtick_labels_{}'.format(unit)],
            labels=["5","10","15","20","25"]
        )
        # ax1.set_ylabel(
        #     "Birth year",
        #     fontsize=10,
        #     labelpad=5,
        #     color=fontcolor,
        # )
        ax1.set_ylabel(
            None,
        )        
        # ax1.text(
        #     x=0.5,y=1.1,
        #     s=ax1_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax1.transAxes,
        #     fontsize=10
        # )    
        
        # full population rich quantile (grdi)
        if unit == 'pple':
            ax2.barh(
                y=birth_years[::-1*per_x],
                width=poor_pop[::-1*per_x],
                height=height,
                color='darkred',
                zorder=1,
                alpha=0.3
            ) 
        # unprec population rich quantile (grdi)
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=poor_unprec_higmt[::-1*per_x],
            height=height,
            color='darkred',
            zorder=1,
            xerr=[i * -1 for i in poor_std_higmt[::-1*per_x]],
            
        )     
        # ax2.tick_params(left=False)
        # ax2.set_xticks(
        #     ticks=ax_xts['ax2_xticks_{}'.format(unit)],
        #     labels=ax_xts['xtick_labels_{}'.format(unit)],
        #     color=fontcolor
        # )     
        
        ax2.tick_params(left=False,right=False)
        ax2.set_xticks(
            ticks=ax_xts['ax2_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
            color=fontcolor
        )    
        ax2.set_yticks(
            birth_years[::-1*per_x*2]
        )
        ax1.set_yticks([])        
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')             
        
        # legend stuff
        ax1_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_low],1))
        ax2_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_high],1))

        # ax2.text(
        #     x=0.5,y=1.1,
        #     s=ax2_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax2.transAxes,
        #     fontsize=10
        # )        
        
        # legend stuff
        legendcols = cmap
        handles = [
            Rectangle((0,0),1,1,color=legendcols[0]),
            Rectangle((0,0),1,1,color=legendcols[1]),
        ]

        labels= [
            ax1_title,
            ax2_title
        ]

        ax1.legend(
            handles, 
            labels, 
            bbox_to_anchor=(x0, y0, xlen, ylen), 
            loc='upper right',
            ncol=1,
            fontsize=legend_font, 
            labelcolor=fontcolor,
            mode="expand", 
            borderaxespad=0.,\
            frameon=False, 
            columnspacing=0.05, 
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad
        )                
        
        # ax1.spines['top'].set_visible(False)
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # for ax in (ax1,ax2):
        #     ax.tick_params(colors='gray')
        #     ax.spines['left'].set_color('gray')
        #     ax.spines['bottom'].set_color('gray')
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        # ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # for ax in (ax1,ax2):
        ax1.tick_params(colors='gray')
        ax2.tick_params(colors='gray')
        ax1.spines['bottom'].set_color('gray')
        ax2.spines['bottom'].set_color('gray')
        ax2.spines['right'].set_color('gray')
        ax2.spines['left'].set_color('gray')        
        
        # plot asterisk or star 1/4 through the bar (0.25 * unprec number as position) if significant differenc and if given side is bigger
        # also plot percentage numbers midway through the bars
        poor_unprec_lowgmt_pct = poor_unprec_lowgmt / poor_pop * 100
        poor_unprec_higmt_pct = poor_unprec_higmt / poor_pop * 100
        
        for i,by in enumerate(birth_years[::-1*per_x]):
            
            # asterisks for significance
            if pvalues_poor[::-1*per_x][i] < sl:
                ax2.plot(
                    poor_unprec_higmt[::-1*per_x][i]/4,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,  
                    color='k',                  
                )          
                
            # percentages
            # left side / low gmt
            if poor_unprec_lowgmt_pct[::-1*per_x][i] > 50:
                x_poor_lowgmt = poor_unprec_lowgmt[::-1*per_x][i] * -2/3
            elif poor_unprec_lowgmt_pct[::-1*per_x][i] < 50:
                x_poor_lowgmt = poor_pop[::-1*per_x][i] * -0.6
            ax1.text(
                x=x_poor_lowgmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_lowgmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transData,
                fontsize=9,
                color='k'
            ) 
            
            # right side / high gmt
            if poor_unprec_higmt_pct[::-1*per_x][i] > 50:
                x_poor_higmt = poor_unprec_higmt[::-1*per_x][i] * 2/3
            elif poor_unprec_higmt_pct[::-1*per_x][i] < 50:
                x_poor_higmt = poor_pop[::-1*per_x][i] * 0.6
            ax2.text(
                x=x_poor_higmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_higmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transData,
                fontsize=9,
                color='k'
            )                    
        ax1.invert_yaxis() # only have to do this once because because y axis are shared
        # ax2.invert_yaxis()
        # f.savefig(
        #     './figures/pyramid/inverted/vln_pyramid_poorlohi_{}_{}_{}_{}.png'.format(vln_type,e,qntl_range,unit),
        #     dpi=1000,
        #     bbox_inches='tight',
        # )
        # plt.show()
        
        
# %%        
def pyramid_rich_lowhigh(
    flags,
    df_GMT_strj,
    vln_type,
):

    per_x=5 # every how many years do we plot (i.e. 1960,1970,1980,...2020 on y axis would be "10")
    height=4 # thickness of bars
    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_cp=12 # "cp" for "current pathway"; 12 or 17
    GMT_low=0
    GMT_high=20
    # plot type (will get removed and looped outside function)
    # vln_type='gdp'
    fontcolor='gray'
    # bbox for legend
    x0 = 0.1
    y0 = 0.8
    xlen = 0.2
    ylen = 0.3
    cmap = ['steelblue','darkred']  
    # space between legend entries
    legend_entrypad = 0.5
    # length per legend entry
    legend_entrylen = 0.75
    # more legend stuff
    legend_font = 10
    legend_lw=3.5      

    # pick GMT
    GMT = GMT_cp 
    unit='pple'
    qntl_range = '20'

    # start with GDP ====================================================================================
    with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
        d_pyramid_plot = pk.load(f)    

    for e in extremes:    
        
        poor_unprec_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
        poor_std_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_std_{}richest'.format(qntl_range)])
        poor_pop = np.asarray(d_pyramid_plot[e][GMT_low]['population_quantiles_{}richest'.format(qntl_range)])
        
        poor_unprec_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
        poor_std_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_std_{}richest'.format(qntl_range)])       
        
        pvalues_poor = np.asarray(d_pyramid_plot[e]['0_20']['ttest_{}pc_pvals_rich'.format(qntl_range)]) # pvalues for whether high GMT shows higher unrpecedented numbers that low GMT for poor             
        
        # labels
        if qntl_range == '10':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-4,-8,-12]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["4","8","12"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [4,8,12]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]                               
            
        elif qntl_range == '20':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-5,-10,-15,-20,-25]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["5","10","15","20","25"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [5,10,15,20,25]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]  

        f,(ax1,ax2) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(6,6),
            sharey=True,
        )
        f.subplots_adjust(wspace=0)
        # full population poor quantile (gdp)
        if unit == 'pple':
            ax1.barh(
                y=birth_years[::-1*per_x], # added to flip y axis
                width=[i * -1 for i in poor_pop[::-1*per_x]],
                height=height,
                color='steelblue',
                zorder=1,
                alpha=0.3
            )
            
        # unprec population poor quantile (gdp)
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_unprec_lowgmt[::-1*per_x]],
            height=height,
            color='steelblue',
            zorder=1,
            xerr=[i * -1 for i in poor_std_lowgmt[::-1*per_x]],
        )    
        # # ax1.set_xlabel('Millions of people')
        if unit == 'pple':
            variable = 'Millions of people'
        else:
            variable = 'Percentage of cohort'
        ax1.text(
            x=1,y=-0.1,
            s=variable,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax1.transAxes,
            fontsize=10,
            color='gray',
        )
        ax1.set_xticks(
            ticks=ax_xts['ax1_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
        )
        ax1.set_ylabel(
            "Birth year",
            fontsize=10,
            labelpad=5,
            color=fontcolor,
        )
        # ax1.text(
        #     x=0.5,y=1.1,
        #     s=ax1_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax1.transAxes,
        #     fontsize=10
        # )    
        
        # full population rich quantile (grdi)
        if unit == 'pple':
            ax2.barh(
                y=birth_years[::-1*per_x],
                width=poor_pop[::-1*per_x],
                height=height,
                color='darkred',
                zorder=1,
                alpha=0.3
            ) 
        # unprec population rich quantile (grdi)
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=poor_unprec_higmt[::-1*per_x],
            height=height,
            color='darkred',
            zorder=1,
            xerr=[i * -1 for i in poor_std_higmt[::-1*per_x]],
            
        )     
        ax2.tick_params(left=False)
        ax2.set_xticks(
            ticks=ax_xts['ax2_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
            color=fontcolor
        )     
        
        # legend stuff
        ax1_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_low],1))
        ax2_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_high],1))

        # ax2.text(
        #     x=0.5,y=1.1,
        #     s=ax2_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax2.transAxes,
        #     fontsize=10
        # )        
        
        # legend stuff
        legendcols = cmap
        handles = [
            Rectangle((0,0),1,1,color=legendcols[0]),
            Rectangle((0,0),1,1,color=legendcols[1]),
        ]

        labels= [
            ax1_title,
            ax2_title
        ]

        ax1.legend(
            handles, 
            labels, 
            bbox_to_anchor=(x0, y0, xlen, ylen), 
            loc='upper right',
            ncol=1,
            fontsize=legend_font, 
            labelcolor=fontcolor,
            mode="expand", 
            borderaxespad=0.,\
            frameon=False, 
            columnspacing=0.05, 
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad
        )                
        
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        for ax in (ax1,ax2):
            ax.tick_params(colors='gray')
            ax.spines['left'].set_color('gray')
            ax.spines['bottom'].set_color('gray')
        
        # plot asterisk or star 1/4 through the bar (0.25 * unprec number as position) if significant differenc and if given side is bigger
        # also plot percentage numbers midway through the bars
        poor_unprec_lowgmt_pct = poor_unprec_lowgmt / poor_pop * 100
        poor_unprec_higmt_pct = poor_unprec_higmt / poor_pop * 100
        
        for i,by in enumerate(birth_years[::-1*per_x]):
            
            # asterisks for significance
            if pvalues_poor[::-1*per_x][i] < sl:
                ax2.plot(
                    poor_unprec_higmt[::-1*per_x][i]/4,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,  
                    color='k',                  
                )          
                
            # percentages
            # left side / low gmt
            if poor_unprec_lowgmt_pct[::-1*per_x][i] > 50:
                x_poor_lowgmt = poor_unprec_lowgmt[::-1*per_x][i] * -2/3
            elif poor_unprec_lowgmt_pct[::-1*per_x][i] < 50:
                x_poor_lowgmt = poor_pop[::-1*per_x][i] * -0.6
            ax1.text(
                x=x_poor_lowgmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_lowgmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transData,
                fontsize=9,
                color='k'
            ) 
            
            # right side / high gmt
            if poor_unprec_higmt_pct[::-1*per_x][i] > 50:
                x_poor_higmt = poor_unprec_higmt[::-1*per_x][i] * 2/3
            elif poor_unprec_higmt_pct[::-1*per_x][i] < 50:
                x_poor_higmt = poor_pop[::-1*per_x][i] * 0.6
            ax2.text(
                x=x_poor_higmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_higmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transData,
                fontsize=9,
                color='k'
            )                    
        ax1.invert_yaxis() # only have to do this once because because y axis are shared
        f.savefig(
            './figures/pyramid/inverted/vln_pyramid_richlohi_{}_{}_{}_{}.png'.format(vln_type,e,qntl_range,unit),
            dpi=1000,
            bbox_inches='tight',
        )
        plt.show()
        
# %%        
def pyramid_rich_lowhigh_subplot(
    flags,
    df_GMT_strj,
    vln_type,
    f,
    ax1,
    ax2
):

    ax1.set_title(
        'd',
        fontweight='bold',
        loc='left'
    )  
    ax1.set_title(
        'Low deprivation',
        fontweight='bold',
        loc='right',
        color='gray',
        x=1.4
    )          

    per_x=5 # every how many years do we plot (i.e. 1960,1970,1980,...2020 on y axis would be "10")
    height=4 # thickness of bars
    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_cp=12 # "cp" for "current pathway"; 12 or 17
    GMT_low=0
    GMT_high=20
    # plot type (will get removed and looped outside function)
    # vln_type='gdp'
    fontcolor='gray'
    # bbox for legend
    x0 = 0.1
    y0 = 0.8
    xlen = 0.2
    ylen = 0.3
    cmap = ['steelblue','darkred']  
    # space between legend entries
    legend_entrypad = 0.5
    # length per legend entry
    legend_entrylen = 0.75
    # more legend stuff
    legend_font = 10
    legend_lw=3.5      

    # pick GMT
    GMT = GMT_cp 
    unit='pple'
    qntl_range = '20'

    # start with GDP ====================================================================================
    with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
        d_pyramid_plot = pk.load(f)    

    for e in extremes:    
        
        poor_unprec_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
        poor_std_lowgmt = np.asarray(d_pyramid_plot[e][GMT_low]['unprec_pop_std_{}richest'.format(qntl_range)])
        poor_pop = np.asarray(d_pyramid_plot[e][GMT_low]['population_quantiles_{}richest'.format(qntl_range)])
        
        poor_unprec_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
        poor_std_higmt = np.asarray(d_pyramid_plot[e][GMT_high]['unprec_pop_std_{}richest'.format(qntl_range)])       
        
        pvalues_poor = np.asarray(d_pyramid_plot[e]['0_20']['ttest_{}pc_pvals_rich'.format(qntl_range)]) # pvalues for whether high GMT shows higher unrpecedented numbers that low GMT for poor             
        
        # labels
        if qntl_range == '10':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-4,-8,-12]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["4","8","12"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [4,8,12]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]                               
            
        elif qntl_range == '20':
            ax_xts = {}
            ax_xts['ax1_xticks_pple'] = [-5,-10,-15,-20,-25]
            ax_xts['ax1_xticks_pct'] = [-25,-50,-75,-100]
            ax_xts['xtick_labels_pple'] = ["5","10","15","20","25"]
            ax_xts['xtick_labels_pct'] = ["25","50","75","100"]
            ax_xts['ax2_xticks_pple'] = [5,10,15,20,25]
            ax_xts['ax2_xticks_pct'] = [25,50,75,100]  

        # f,(ax1,ax2) = plt.subplots(
        #     nrows=1,
        #     ncols=2,
        #     figsize=(6,6),
        #     sharey=True,
        # )
        # f.subplots_adjust(wspace=0)
        # full population poor quantile (gdp)
        if unit == 'pple':
            ax1.barh(
                y=birth_years[::-1*per_x], # added to flip y axis
                width=[i * -1 for i in poor_pop[::-1*per_x]],
                height=height,
                color='steelblue',
                zorder=1,
                alpha=0.3
            )
            
        # unprec population poor quantile (gdp)
        ax1.barh(
            y=birth_years[::-1*per_x], # added to flip y axis
            width=[i * -1 for i in poor_unprec_lowgmt[::-1*per_x]],
            height=height,
            color='steelblue',
            zorder=1,
            xerr=[i * -1 for i in poor_std_lowgmt[::-1*per_x]],
        )    
        # # ax1.set_xlabel('Millions of people')
        if unit == 'pple':
            variable = 'Millions of people'
        else:
            variable = 'Percentage of cohort'
        # ax1.text(
        #     x=1,y=-0.1,
        #     s=variable,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax1.transAxes,
        #     fontsize=10,
        #     color='gray',
        # )
        ax1.tick_params(left=True,right=False)
        ax1.set_xticks(
            ticks=ax_xts['ax1_xticks_{}'.format(unit)],
            # labels=ax_xts['xtick_labels_{}'.format(unit)],
            labels=["5","10","15","20",None]
        )
        # ax1.set_ylabel(
        #     "Birth year",
        #     fontsize=10,
        #     labelpad=5,
        #     color=fontcolor,
        # )
        ax1.set_ylabel(
            None
        )        
        ax2.set_ylabel(
            "Birth year",
            fontsize=10,
            labelpad=5,
            color=fontcolor,
        )        
        # ax1.text(
        #     x=0.5,y=1.1,
        #     s=ax1_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax1.transAxes,
        #     fontsize=10
        # )    
        
        # full population rich quantile (grdi)
        if unit == 'pple':
            ax2.barh(
                y=birth_years[::-1*per_x],
                width=poor_pop[::-1*per_x],
                height=height,
                color='darkred',
                zorder=1,
                alpha=0.3
            ) 
        # unprec population rich quantile (grdi)
        ax2.barh(
            y=birth_years[::-1*per_x],
            width=poor_unprec_higmt[::-1*per_x],
            height=height,
            color='darkred',
            zorder=1,
            xerr=[i * -1 for i in poor_std_higmt[::-1*per_x]],
            
        )     
        ax2.tick_params(left=False,right=True)
        ax2.set_xticks(
            ticks=ax_xts['ax2_xticks_{}'.format(unit)],
            labels=ax_xts['xtick_labels_{}'.format(unit)],
            color=fontcolor
        )     
        
        # ax2.tick_params(left=False,right=True)
        # ax2.set_xticks(
        #     ticks=ax_xts['ax2_xticks_{}'.format(unit)],
        #     labels=ax_xts['xtick_labels_{}'.format(unit)],
        #     color=fontcolor
        # )    
        ax2.set_yticks(
            birth_years[::-1*per_x*2]
        )
        ax1.set_yticks(
            birth_years[::-1*per_x*2]
        )        
        ax1.set_yticklabels([])
        ax2.set_yticklabels(birth_years[::-1*per_x*2])
        ax1.yaxis.tick_left()        
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')              
        
        # legend stuff
        ax1_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_low],1))
        ax2_title = '{} °C GMT warming by 2100'.format(np.round(df_GMT_strj.loc[2100,GMT_high],1))

        # ax2.text(
        #     x=0.5,y=1.1,
        #     s=ax2_title,
        #     horizontalalignment='center',
        #     verticalalignment='center',
        #     transform=ax2.transAxes,
        #     fontsize=10
        # )        
        
        # legend stuff
        # legendcols = cmap
        # handles = [
        #     Rectangle((0,0),1,1,color=legendcols[0]),
        #     Rectangle((0,0),1,1,color=legendcols[1]),
        # ]

        # labels= [
        #     ax1_title,
        #     ax2_title
        # ]

        # ax1.legend(
        #     handles, 
        #     labels, 
        #     bbox_to_anchor=(x0, y0, xlen, ylen), 
        #     loc='upper right',
        #     ncol=1,
        #     fontsize=legend_font, 
        #     labelcolor=fontcolor,
        #     mode="expand", 
        #     borderaxespad=0.,\
        #     frameon=False, 
        #     columnspacing=0.05, 
        #     handlelength=legend_entrylen, 
        #     handletextpad=legend_entrypad
        # )                
        
        # ax1.spines['top'].set_visible(False)
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # for ax in (ax1,ax2):
        #     ax.tick_params(colors='gray')
        #     ax.spines['left'].set_color('gray')
        #     ax.spines['bottom'].set_color('gray')
        
        ax1.spines['top'].set_visible(False)
        # ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)
        # for ax in (ax1,ax2):
        ax1.tick_params(colors='gray')
        ax2.tick_params(colors='gray')
        ax1.spines['bottom'].set_color('gray')
        ax2.spines['bottom'].set_color('gray')
        ax1.spines['left'].set_color('gray')
        ax2.spines['right'].set_color('gray')
        ax2.spines['left'].set_color('gray')        
        
        # plot asterisk or star 1/4 through the bar (0.25 * unprec number as position) if significant differenc and if given side is bigger
        # also plot percentage numbers midway through the bars
        poor_unprec_lowgmt_pct = poor_unprec_lowgmt / poor_pop * 100
        poor_unprec_higmt_pct = poor_unprec_higmt / poor_pop * 100
        
        for i,by in enumerate(birth_years[::-1*per_x]):
            
            # asterisks for significance
            if pvalues_poor[::-1*per_x][i] < sl:
                ax2.plot(
                    poor_unprec_higmt[::-1*per_x][i]/4,
                    by,
                    marker=(6,2,0),
                    zorder=5,
                    markersize=8,  
                    color='k',                  
                )          
                
            # percentages
            # left side / low gmt
            if poor_unprec_lowgmt_pct[::-1*per_x][i] > 50:
                x_poor_lowgmt = poor_unprec_lowgmt[::-1*per_x][i] * -2/3
            elif poor_unprec_lowgmt_pct[::-1*per_x][i] < 50:
                x_poor_lowgmt = poor_pop[::-1*per_x][i] * -0.6
            ax1.text(
                x=x_poor_lowgmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_lowgmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transData,
                fontsize=9,
                color='k'
            ) 
            
            # right side / high gmt
            if poor_unprec_higmt_pct[::-1*per_x][i] > 50:
                x_poor_higmt = poor_unprec_higmt[::-1*per_x][i] * 2/3
            elif poor_unprec_higmt_pct[::-1*per_x][i] < 50:
                x_poor_higmt = poor_pop[::-1*per_x][i] * 0.6
            ax2.text(
                x=x_poor_higmt,
                y=by,
                s='{}%'.format(str(int(np.round(poor_unprec_higmt_pct[::-1*per_x][i])))),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transData,
                fontsize=9,
                color='k'
            )                    
        ax1.invert_yaxis() # only have to do this once because because y axis are shared
        ax2.invert_yaxis()
        # f.savefig(
        #     './figures/pyramid/inverted/vln_pyramid_richlohi_{}_{}_{}_{}.png'.format(vln_type,e,qntl_range,unit),
        #     dpi=1000,
        #     bbox_inches='tight',
        # )
        # plt.show()        
        
# %% ===============================================================================
# composing entire pyramid plot together
def pyramid_combined(
    ds_grdi_qntls,
    ds_gdp_qntls,
    da_cohort_size_1960_2020,
    gdf_robinson_bounds,
    df_GMT_strj,
    flags,
):

    # putting all my pyramid plot shit together
    vln_type='grdi'
    x=10
    y=10
    f = plt.figure(figsize=(x,y))    
    gs0 = gridspec.GridSpec(11,4, wspace=0.0)

    # # map top left
    # ax00 = f.add_subplot(gs0[0:5,0:2],projection=ccrs.Robinson())

    # main pyramid of rich vs poor
    # ax01 for top left -> extra "1" for left, extra "2" for right (i.e., to match ax1 and ax2 from original singular pyramid plot)
    ax011 = f.add_subplot(gs0[1:5,2:3]) 
    ax012 = f.add_subplot(gs0[1:5,3:4])

    # map top left (calling this after top right subplots helps move the map border on top of the other subplot's space)
    ax00 = f.add_subplot(gs0[0:5,0:2],projection=ccrs.Robinson())

    # secondary pyramid of poor low vs high gmt trajectory
    ax101 = f.add_subplot(gs0[6:10,0:1])
    ax102 = f.add_subplot(gs0[6:10,1:2],sharey=ax101)

    # secondary pyramid of rich low vs high gmt trajectory
    ax111 = f.add_subplot(gs0[6:10,2:3])
    ax112 = f.add_subplot(gs0[6:10,3:4])
    # ax112 = f.add_subplot(gs0[6:10,3:4],sharey=ax111)

    #--------------------------------
    # map
    pyramid_map_subplot(
        vln_type,
        ds_grdi_qntls,
        ds_gdp_qntls,
        da_cohort_size_1960_2020,
        gdf_robinson_bounds,
        f,
        ax00
    )        
            
    #--------------------------------
    # pyramid - top right
    pyramid_subplot(
        flags,
        df_GMT_strj,
        vln_type,
        f,
        ax011,
        ax012
    )

    #--------------------------------
    # pyramid - bottom left
    pyramid_poor_lowhigh_subplot(
        flags,
        df_GMT_strj,
        vln_type,
        f,
        ax101,
        ax102
    )
        
    #--------------------------------
    # pyramid - bottom right
    pyramid_rich_lowhigh_subplot(
        flags,
        df_GMT_strj,
        vln_type,
        f,
        ax111,
        ax112
    )

    f.savefig('./ms_figures/f4_pyramid_combined_{}_{}.png'.format(vln_type,flags['version']),dpi=1000,bbox_inches='tight')
    f.savefig('./ms_figures/f4_pyramid_combined_{}_{}.pdf'.format(vln_type,flags['version']),dpi=1000,bbox_inches='tight')
    # %%

    # # first set up quantiles for plotting
    # pyramid_setup(
    #     flags,
    #     ds_gdp,
    #     ds_grdi,
    #     da_cohort_size_1960_2020,
    #     ds_vulnerability,
    # )
    # # then run plots
    # for vln_type in ('gdp','grdi'):
    #     print(vln_type)
    #     pyramid_plot( # this is plot of rich vs poor (brown & green) for current trajectories 
    #         flags,
    #         df_GMT_strj,
    #         vln_type,
    #     )
    #     pyramid_map( # map showing locations of top 20 vs bottom 20 quantiles
    #         vln_type,
    #         ds_grdi_qntls,
    #         ds_gdp_qntls,
    #         da_cohort_size_1960_2020,
    #         gdf_robinson_bounds,
    #     )
    #     pyramid_poor_lowhigh( # poor pop but high vs low GMT trajectories
    #         flags,
    #         df_GMT_strj,
    #         vln_type,
    #     )
    #     pyramid_rich_lowhigh( # rich pop but high vs low GMT trajectories
    #         flags,
    #         df_GMT_strj,
    #         vln_type
    #     )