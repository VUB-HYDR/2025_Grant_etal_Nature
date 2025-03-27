# ---------------------------------------------------------------
# Functions to compute emergence of exposure from noise
# ----------------------------------------------------------------

#               
#%%  ----------------------------------------------------------------
# IMPORT AND PATH 
# ----------------------------------------------------------------

import os
import glob
import requests
from zipfile import ZipFile
import io
import xarray as xr
import pickle as pk
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import matplotlib as mpl
import mapclassify as mc
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import regionmask as rm
import numpy as np
import pandas as pd
import geopandas as gpd
# import rioxarray as rxr
from scipy import interpolate
from scipy.stats import ttest_rel
import cartopy.crs as ccrs
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

#%% ----------------------------------------------------------------   
# Function to open isimip data array and read years from filename
# (the isimip calendar "days since 1661-1-1 00:00:00" cannot be read by xarray datetime )
# this implies that years in file need to correspond to years in filename
def open_dataarray_isimip(file_name): 
    
    begin_year = int(file_name.split('_')[-2])
    end_year = int(file_name.split('_')[-1].split('.')[0])
    
    # some files contain extra var 'time_bnds', first try reading for single var
    try:
        
        da = xr.open_dataarray(file_name, decode_times=False)
        
    except:
        
        da = xr.open_dataset(file_name, decode_times=False).exposure
    
    da['time'] = np.arange(begin_year,end_year+1)
    
    return da

#%% ----------------------------------------------------------------
# bootstrapping function 
# ------------------------------------------------------------------

def resample(
    da, 
    resample_dim,
    life_extent,
):
    """Resample with replacement in dimension ``resample_dim``. https://climpred.readthedocs.io/en/stable/_modules/climpred/bootstrap.html

    Args:
        initialized (xr.Dataset): input xr.Dataset to be resampled.
        resample_dim (str): dimension to resample along.
        life_extent (int): number of years per lifetime
        
    Returns:
        xr.Dataset: resampled along ``resample_dim``.

    """
    to_be_resampled = da[resample_dim].values
    smp = np.random.choice(to_be_resampled, life_extent)
    smp_da = da.sel({resample_dim: smp})
    smp_da[resample_dim] = np.arange(1960,1960+life_extent)
    return smp_da


#%% ----------------------------------------------------------------
# function for returning countries for grid scale analysis
# ------------------------------------------------------------------

def get_gridscale_regions(
    grid_area,
    flags,
    gdf_country_borders,
):

    lat = grid_area.lat.values
    lon = grid_area.lon.values  
    
    if flags['gridscale_country_subset']:
    
        # take all countries for heatwavedarea
        if flags['extr'] == 'heatwavedarea':
            
            gdf_country = gdf_country_borders.loc[:,'geometry']
            list_countries = gdf_country.index.values

        # countries in Mediterranean for drought
        elif flags['extr'] == 'driedarea': 
            
            med = 'Mediterranean'
            gdf_ar6 = gpd.read_file('./data/shapefiles/IPCC-WGI-reference-regions-v4.shp')
            gdf_ar6 = gdf_ar6.loc[gdf_ar6['Name']==med]
            gdf_ar6 = gdf_ar6.loc[:,['Name','geometry']]
            gdf_ar6 = gdf_ar6.rename(columns={'Name':'name'})
            gdf_country = gdf_country_borders.loc[:,'geometry']
            gdf_country = gdf_country.reset_index()
            gdf_med_countries = gdf_country.loc[gdf_country.intersects(gdf_ar6['geometry'].iloc[0])]
            countries_med_3D = rm.mask_3D_geopandas(gdf_med_countries,lon,lat)
            ar6_regs_3D = rm.defined_regions.ar6.land.mask_3D(lon,lat)
            med_3D = ar6_regs_3D.isel(region=(ar6_regs_3D.names == 'Mediterranean')).squeeze()
            
            c_valid = []
            for c in gdf_med_countries.index:
                # next line gives nans outside country, 
                # 1 in parts of country in AR6 Medit
                # and 0 in parts outside Medit.
                c_in_med = med_3D.where(countries_med_3D.sel(region=c)==1) 
                c_area_in_med = c_in_med.weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area = countries_med_3D.sel(region=c).weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area_frac = c_area_in_med.item() / c_area.item()
                if c_area_frac > 0.5:
                    c_valid.append(c)
            countries_med_3D = countries_med_3D.loc[{'region':c_valid}]
            gdf_med_countries = gdf_med_countries.loc[c_valid]
            list_countries = gdf_med_countries['name'].values
            
        # countries in the Nile for floods
        elif flags['extr'] == 'floodedarea':
            
            basin = 'Nile'
            gdf_basins = gpd.read_file('./data/shapefiles/Major_Basins_of_the_World.shp')
            gdf_basins = gdf_basins.loc[:,['NAME','geometry']]
            gdf_basin = gdf_basins.loc[gdf_basins['NAME']==basin]
            gdf_basin = gdf_basin.rename(columns={'Name':'name'})
            if len(gdf_basin.index) > 1:
                gdf_basin = gdf_basin.dissolve()
            gdf_country = gdf_country_borders.loc[:,'geometry']
            gdf_country = gdf_country.reset_index()
            gdf_basin_countries = gdf_country.loc[gdf_country.intersects(gdf_basin['geometry'].iloc[0])]
            countries_basin_3D = rm.mask_3D_geopandas(gdf_basin_countries,lon,lat)
            basin_3D = rm.mask_3D_geopandas(gdf_basin,lon,lat)
            
            c_valid = []
            for c in gdf_basin_countries.index:
                # next line gives nans outside country, 
                # 1 in parts of country in basin
                # and 0 in parts outside basin.
                c_in_basin = basin_3D.where(countries_basin_3D.sel(region=c)==1) 
                c_area_in_basin = c_in_basin.weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area = countries_basin_3D.sel(region=c).weighted(grid_area/10**6).sum(dim=('lat','lon'))
                c_area_frac = c_area_in_basin.item() / c_area.item()
                if c_area_frac > 0.3:
                    c_valid.append(c)
            countries_basin_3D = countries_basin_3D.loc[{'region':c_valid}]
            gdf_basin_countries = gdf_basin_countries.loc[c_valid]
            list_countries = gdf_basin_countries['name'].values
            list_countries = np.sort(np.insert(list_countries,-1,'Egypt'))
            
    else:
        
        gdf_country = gdf_country_borders.loc[:,'geometry']
        list_countries = gdf_country.index.values        
            
    return list_countries

#%% ----------------------------------------------------------------
# grid scale emergence function
# ------------------------------------------------------------------

def gridscale_emergence(
    d_isimip_meta,
    d_pic_meta,
    flags,
    list_countries,
    da_cohort_size,
    countries_regions,
    countries_mask,
    df_life_expectancy_5,
    da_population,
):

    # lifetime exposure dataset (pop weighted mean of pixel scale lifetime exposure per country, run, GMT and birthyear)
    # ds_le = xr.Dataset(
    #     data_vars={
    #         'lifetime_exposure_popweight': (
    #             ['country','run','GMT','birth_year'],
    #             np.full(
    #                 (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
    #                 fill_value=np.nan,
    #             ),
    #         ),
    #         'lifetime_exposure_latweight': (
    #             ['country','run','GMT','birth_year'],
    #             np.full(
    #                 (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
    #                 fill_value=np.nan,
    #             ),
    #         )            
    #     },
    #     coords={
    #         'country': ('country', list_countries),
    #         'birth_year': ('birth_year', birth_years),
    #         'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
    #         'GMT': ('GMT', GMT_labels)
    #     }
    # )

    # pop fraction dataset (sum of unprecedented exposure pixels' population per per country, run, GMT and birthyear)
    ds_pf = xr.Dataset(
        data_vars={
            'unprec': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),      
        },
        coords={
            'country': ('country', list_countries),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_labels)
        }
    )
    for pthresh in pic_qntl_labels:
        ds_pf['unprec_{}'.format(pthresh)] = ds_pf['unprec'].copy()
        ds_pf['unprec_fraction_{}'.format(pthresh)] = ds_pf['unprec'].copy()
    ds_pf = ds_pf.drop(labels=['unprec'])

    for cntry in list_countries:
        
        print('running gridscale emergence for {}'.format(cntry))
        
        if not os.path.exists('./data/{}/{}/{}'.format(flags['version'],flags['extr'],cntry)):
            os.makedirs('./data/{}/{}/{}'.format(flags['version'],flags['extr'],cntry)) # testing makedirs

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
        
        # check for gridscale demography pickle file and make it if it doesn't exist
        if not os.path.isfile('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry)):
            
            da_smple_pop = da_population.where(da_cntry==1) * da_smple_cht_prp # use pop and relative cohort sizes to get people per cohort

            # demography dataset
            ds_dmg = xr.Dataset(
                data_vars={
                    'life_expectancy': (
                        ['birth_year'],
                        df_life_expectancy_5[cntry].values
                    ),
                    'death_year': (
                        ['birth_year'],
                        np.floor(df_life_expectancy_5[cntry].values + df_life_expectancy_5[cntry].index).astype('int')
                    ),
                    'population': (
                        ['time','lat','lon','age'],
                        da_smple_pop.data
                    ),
                    'country_extent': (
                        ['lat','lon'],
                        da_cntry.data
                    ),
                },
                coords={
                    'birth_year': ('birth_year', birth_years),
                    'time': ('time', da_population.time.data),
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                    'age': ('age', np.arange(100,-1,-1)),
                }
            )

            # get birthyear aligned population for unprecedented calculation (by_population), also use for weighted mean of lifetime exposure and age emergence
            bys = []
            for by in birth_years:
                    
                time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
                ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                data = ds_dmg['population'].sel(time=time,age=ages) # paired selections
                data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':by}).drop_vars('age')
                bys.append(data)

            # adding by_population_y0 to rep cohort sizes as number of people at birth year (y0)
            ds_dmg['by_population_y0'] = xr.concat(bys,dim='birth_year').where(ds_dmg['country_extent']==1)
            da_times=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')
            da_birth_years=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')        
            ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].sel(time=da_times,birth_year=da_birth_years)
            # set order of dims
            ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].transpose('birth_year','lat','lon')
            
            # pickle gridscale demography for country
            with open('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry), 'wb') as f:
                pk.dump(ds_dmg,f)       
                
        else:
            
            # load demography pickle
            with open('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry), 'rb') as f:
                ds_dmg = pk.load(f)                      
        
        # check for PIC lifetime exposure pickle file (for ds_pic); process and dump pickle if not already existing
        if not os.path.isfile('./data/{}/{}/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry)):
            
            # pic dataset for bootstrapped lifetimes
            ds_pic = xr.Dataset(
                data_vars={
                    'lifetime_exposure': (
                        ['lifetimes','lat','lon'],
                        np.full(
                            (len(list(d_pic_meta.keys())*nboots),len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    )
                },
                coords={
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                    'lifetimes': ('lifetimes', np.arange(len(list(d_pic_meta.keys())*nboots))),
                }
            )                
            
            # loop over PIC simulations
            c = 0
            for i in list(d_pic_meta.keys()):
                
                print('simulation {} of {}'.format(i,len(d_pic_meta)))
                
                # load AFA data of that run
                with open('./data/{}/{}/isimip_AFA_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
                    da_AFA_pic = pk.load(f)
                    
                da_AFA_pic = da_AFA_pic.where(ds_dmg['country_extent']==1,drop=True)
                
                # regular boot strapping for all reasonably sized countries (currently only exception is for Russia, might expand this for hazards with more sims)
                if not cntry in ['Russian Federation','Canada','United States','China']:

                    # resample 10000 lifetimes and then sum 
                    # pic life extent is just max 1960 across all countries, such that this covers enough time space for selection in the lifetime summing below
                    da_exposure_pic = xr.concat(
                        [resample(da_AFA_pic,resample_dim,pic_life_extent) for i in range(nboots)],
                        dim='lifetimes'    
                    ).assign_coords({'lifetimes':np.arange(c*nboots,c*nboots+nboots)})
                    
                    # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                    da_pic_le = da_exposure_pic.loc[
                        {'time':np.arange(pic_by,ds_dmg['death_year'].sel(birth_year=pic_by).item()+1)}
                    ].sum(dim='time') +\
                        da_exposure_pic.loc[{'time':ds_dmg['death_year'].sel(birth_year=pic_by).item()}].drop('time') *\
                            (ds_dmg['life_expectancy'].sel(birth_year=pic_by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=pic_by).item()))
                            
                    ds_pic['lifetime_exposure'].loc[
                        {
                            'lat': da_pic_le.lat.data,
                            'lon': da_pic_le.lon.data,
                            'lifetimes': np.arange(c*nboots,c*nboots+nboots),
                        }
                    ] = da_pic_le
                    c += 1
                   
                # for exceptionally sized countries, do piecewise boot strapping (nboots in 10 steps)
                else:
                    
                    for bstep in range(10):
                        
                        # resample 1000 lifetimes and then sum 
                        da_exposure_pic = xr.concat(
                            [resample(da_AFA_pic,resample_dim,pic_life_extent) for i in range(int(nboots/10))],
                            dim='lifetimes'    
                        ).assign_coords({'lifetimes':np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10)})       
                        
                        # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                        da_pic_le = da_exposure_pic.loc[
                            {'time':np.arange(pic_by,ds_dmg['death_year'].sel(birth_year=pic_by).item()+1)}
                        ].sum(dim='time') +\
                            da_exposure_pic.loc[{'time':ds_dmg['death_year'].sel(birth_year=pic_by).item()}].drop('time') *\
                                (ds_dmg['life_expectancy'].sel(birth_year=pic_by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=pic_by).item()))
                                
                        ds_pic['lifetime_exposure'].loc[
                            {
                                'lat': da_pic_le.lat.data,
                                'lon': da_pic_le.lon.data,
                                'lifetimes': np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10),
                            }
                        ] = da_pic_le
                        
                    c += 1
                    
            # pickle PIC lifetime exposure for var, country
            with open('./data/{}/{}/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_pic,f)                
                
        else:
            
            # load PIC pickle
            with open('./data/{}/{}/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'rb') as f:
                ds_pic = pk.load(f)     
                
        # check for PIC quantiles calc'd from bootstrapped lifetime exposures (for ds_pic_qntl); process and dump pickle if not already existing
        if not os.path.isfile('./data/{}/{}/{}/gridscale_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry)):
                         
            # pic dataset for quantiles
            ds_pic_qntl = xr.Dataset(
                data_vars={
                    '99.99': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    # '99.9': (
                    #     ['lat','lon'],
                    #     np.full(
                    #         (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                    #         fill_value=np.nan,
                    #     ),
                    # ),
                    # '99.0': (
                    #     ['lat','lon'],
                    #     np.full(
                    #         (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                    #         fill_value=np.nan,
                    #     ),
                    # ),            
                    # '97.5': (
                    #     ['lat','lon'],
                    #     np.full(
                    #         (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                    #         fill_value=np.nan,
                    #     ),
                    # ),                     
                    # '95.0': (
                    #     ['lat','lon'],
                    #     np.full(
                    #         (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                    #         fill_value=np.nan,
                    #     ),
                    # ),                      
                    # '90.0': (
                    #     ['lat','lon'],
                    #     np.full(
                    #         (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                    #         fill_value=np.nan,
                    #     ),
                    # ),                                                                                          
                },
                coords={
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                }
            )                              
                                
            # pic extreme lifetime exposure definition (added more quantiles for v2)
            ds_pic_qntl['99.99'] = ds_pic['lifetime_exposure'].quantile(
                    q=pic_qntl,
                    dim='lifetimes',
                    method='closest_observation',
                )
            # ds_pic_qntl['99.9'] = ds_pic['lifetime_exposure'].quantile(
            #         q=0.999,
            #         dim='lifetimes',
            #         method='closest_observation',
            #     )            
            # ds_pic_qntl['99.0'] = ds_pic['lifetime_exposure'].quantile(
            #         q=0.99,
            #         dim='lifetimes',
            #         method='closest_observation',
            #     )           
            # ds_pic_qntl['97.5'] = ds_pic['lifetime_exposure'].quantile(
            #         q=0.975,
            #         dim='lifetimes',
            #         method='closest_observation',
            #     )                
            # ds_pic_qntl['95.0'] = ds_pic['lifetime_exposure'].quantile(
            #         q=0.95,
            #         dim='lifetimes',
            #         method='closest_observation',
            #     )     
            # ds_pic_qntl['90.0'] = ds_pic['lifetime_exposure'].quantile(
            #         q=0.9,
            #         dim='lifetimes',
            #         method='closest_observation',
            #     )     
                                             
            # pickle PIC lifetime exposure for var, country
            with open('./data/{}/{}/{}/gridscale_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_pic_qntl,f)                
                
        else:
            
            # load PIC pickle
            with open('./data/{}/{}/{}/gridscale_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'rb') as f:
                ds_pic_qntl = pk.load(f)          

        # loop over simulations
        for i in list(d_isimip_meta.keys()): 

            print('simulation {} of {}'.format(i,len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
                da_AFA = pk.load(f)
                
            # mask to sample country and reduce spatial extent
            da_AFA = da_AFA.where(ds_dmg['country_extent']==1,drop=True)
            
            # loop over GMT trajectories
            for step in GMT_labels:
                
                # run GMT-mapping of years if valid
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    
                    # GMT-mapping
                    da_AFA_step = da_AFA.reindex(
                        {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                    ).assign_coords({'time':year_range})                     
                    
                    # # check for pickle of gridscale lifetime exposure (da_le); process if not existing; os.mkdir('./data/{}/{}/{}'.format(flags['version'],flags['extr'],cntry))
                    # if not os.path.isfile('./data/{}/{}/{}/gridscale_le_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry,i,step)):
                            
                    #     # simple lifetime exposure sum
                    #     da_le = xr.concat(
                    #         [(da_AFA_step.loc[{'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1)}].sum(dim='time') +\
                    #         da_AFA_step.sel(time=ds_dmg['death_year'].sel(birth_year=by).item()).drop('time') *\
                    #         (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item()))\
                    #         for by in birth_years],
                    #         dim='birth_year',
                    #     ).assign_coords({'birth_year':birth_years})
                        
                    #     # dump spatial lifetime exposure for this country/run/GMT
                    #     with open('./data/{}/{}/{}/gridscale_le_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry,i,step), 'wb') as f:
                    #         pk.dump(da_le,f)
                    
                    # # load existing pickle
                    # else:
                        
                    #     with open('./data/{}/{}/{}/gridscale_le_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry,i,step), 'rb') as f:
                    #         da_le = pk.load(f)
                    
                    # # assign pop weighted mean exposure to dataset
                    # ds_le['lifetime_exposure_popweight'].loc[
                    #     {
                    #         'country':cntry,
                    #         'run':i,
                    #         'GMT':step,
                    #         'birth_year':birth_years,
                    #     }
                    # ] = da_le.weighted(ds_dmg['by_population_y0'].fillna(0)).mean(('lat','lon')) # why is there a fillna here?
                    
                    # # assign lat weighted mean exposure to dataset
                    # ds_le['lifetime_exposure_latweight'].loc[
                    #     {
                    #         'country':cntry,
                    #         'run':i,
                    #         'GMT':step,
                    #         'birth_year':birth_years,
                    #     }
                    # ] = da_le.weighted(lat_weights).mean(('lat','lon'))       
                    
                    # commented out this check because it can't incporporate multiple pthresh values (moved indent to left for code below)
                    # # check for pickles of gridscale exposure emergence mask and age emergence; os.mkdir('./data/{}/{}/{}'.format(flags['version'],flags['extr'],cntry))
                    # if not os.path.isfile('./data/{}/{}/{}/gridscale_emergence_mask_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry,i,step,pthresh)):                                 
                        
                    # area affected ('AFA') per year per age (in 'population')
                    da_exp_py_pa = da_AFA_step * xr.full_like(ds_dmg['population'],1)
                    bys = []
            
                    # per birth year, make (year,age) paired selections
                    for by in birth_years:
                            
                        time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
                        ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                        data = da_exp_py_pa.sel(time=time,age=ages) # paired selections
                        data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
                        data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                        # when I reindex time, the len=1 lat coord for cyprus, a small country, disappears
                        # therefore need to reintroduce len=1 coord at correct position
                        for scoord in ['lat','lon']:
                            if data[scoord].size == 1:
                                if scoord == 'lat':
                                    a_pos = 1
                                elif scoord == 'lon':
                                    a_pos = 2
                                data = data.expand_dims(dim={scoord:1},axis=a_pos).copy()
                        data = data.assign_coords({'birth_year':by}).drop_vars('age')
                        data.loc[
                            {'time':ds_dmg['death_year'].sel(birth_year=by).item()}
                        ] = da_AFA_step.loc[{'time':ds_dmg['death_year'].sel(birth_year=by).item()}] *\
                            (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item())
                        bys.append(data)
            
                    da_exp_py_pa = xr.concat(bys,dim='birth_year')
                            
                    # cumulative sum per birthyear (in emergence.py, this cumsum then has .where(==0), should I add this here too?)
                    da_exp_py_pa_cumsum = da_exp_py_pa.cumsum(dim='time')
                    
                    # loop through pic thresholds for emergence (separate file per threshold or dataset? separate file)
                    for pthresh in pic_qntl_labels:
                        
                        # generate exposure mask for timesteps after reaching pic extreme to find emergence
                        da_emergence_mask = xr.where(
                            da_exp_py_pa_cumsum > ds_pic_qntl[pthresh],
                            1,
                            0,
                        )
                            
                        # find birth years/pixels crossing threshold
                        da_birthyear_emergence_mask = xr.where(da_emergence_mask.sum(dim='time')>0,1,0) 
                        
                        with open('./data/{}/{}/{}/gridscale_emergence_mask_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry,i,step,pthresh), 'wb') as f:
                            pk.dump(da_birthyear_emergence_mask,f)
                            
                    # grid cells of population emerging for each PIC threshold
                    for pthresh in pic_qntl_labels:
                        
                        with open('./data/{}/{}/{}/gridscale_emergence_mask_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry,i,step,pthresh), 'rb') as f:
                            da_birthyear_emergence_mask = pk.load(f)                        
                        
                        da_unprec_pop = ds_dmg['by_population_y0'].where(da_birthyear_emergence_mask==1)          
                        
                        ds_pf['unprec_{}'.format(pthresh)].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }] = da_unprec_pop.sum(('lat','lon'))

                        ds_pf['unprec_fraction_{}'.format(pthresh)].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }] = da_unprec_pop.sum(('lat','lon')) / ds_dmg['by_population_y0'].sum(('lat','lon'))
                        
    # # pickle aggregated lifetime exposure, age emergence and pop frac datasets
    # with open('./data/{}/{}/gridscale_aggregated_lifetime_exposure_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
    #     pk.dump(ds_le,f)    
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_pf,f)
        
    return ds_pf

#%% ----------------------------------------------------------------
# grid scale emergence function for testing analysis on constant life expectancy
# ------------------------------------------------------------------

def gridscale_emergence_life_expectancy_constant(
    d_isimip_meta,
    d_pic_meta,
    flags,
    list_countries,
    da_cohort_size,
    countries_regions,
    countries_mask,
    df_life_expectancy_5,
    da_population,
):

    # this is basically a copy of the function above, but runs the analysis on a single life expectancy 
    # but uses same pic lifetime exposure and quantiles as above
    life_expectancy_global = np.round(df_life_expectancy_5.mean().mean()) # one value for all birth years, all countries
    life_expectancy_country = np.round(df_life_expectancy_5.mean()) # one value per country

    # pop fraction dataset (sum of unprecedented exposure pixels' population per per country, run, GMT and birthyear)
    ds_pf = xr.Dataset(
        data_vars={
            'unprec': (
                ['country','run','GMT','birth_year'],
                np.full(
                    (len(list_countries),len(list(d_isimip_meta.keys())),len(GMT_labels),len(birth_years)),
                    fill_value=np.nan,
                ),
            ),      
        },
        coords={
            'country': ('country', list_countries),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_labels)
        }
    )
    for pthresh in pic_qntl_labels:
        # labelling of unprec vars will be same as before for the country-constant life expectancy
        ds_pf['unprec_{}_country_le'.format(pthresh)] = ds_pf['unprec'].copy()
        ds_pf['unprec_fraction_{}_country_le'.format(pthresh)] = ds_pf['unprec'].copy()
        # new labelling for global constant life expectancy
        ds_pf['unprec_{}_global_le'.format(pthresh)] = ds_pf['unprec'].copy()
        ds_pf['unprec_fraction_{}_global_le'.format(pthresh)] = ds_pf['unprec'].copy()        
    ds_pf = ds_pf.drop(labels=['unprec'])

    for cntry in list_countries:
        
        if not os.path.exists('./data/{}/{}/{}'.format(flags['version'],flags['extr']+'_le_test',cntry)):
            os.makedirs('./data/{}/{}/{}'.format(flags['version'],flags['extr']+'_le_test',cntry)) # testing makedirs

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
        
        # check for gridscale demography pickle file and make it if it doesn't exist
        if not os.path.isfile('./data/{}/gridscale_dmg_{}_le_test.pkl'.format(flags['version'],cntry)):
            
            da_smple_pop = da_population.where(da_cntry==1) * da_smple_cht_prp # use pop and relative cohort sizes to get people per cohort

            # demography dataset
            # 'life_expectancy' and 'death_year' use average life expectancy for given country (i.e. repeated over all birth years)
            # 'global_life_expectancy' and 'global_death_year' use average of all birth years/all countries; single constant
            # a bit messy that global_life_expectancy is an attribute rather than variable, but oh well
            ds_dmg = xr.Dataset(
                data_vars={
                    'life_expectancy': (
                        ['birth_year'],
                        np.repeat(life_expectancy_country[cntry],len(birth_years))
                    ),
                    'death_year': (
                        ['birth_year'],
                        np.floor(np.repeat(life_expectancy_country[cntry],len(birth_years)) + df_life_expectancy_5[cntry].index).astype('int')
                    ),
                    'global_death_year': (
                        ['birth_year'],
                        np.floor(np.repeat(life_expectancy_global,len(birth_years)) + df_life_expectancy_5[cntry].index).astype('int')
                    ),                    
                    'population': (
                        ['time','lat','lon','age'],
                        da_smple_pop.data
                    ),
                    'country_extent': (
                        ['lat','lon'],
                        da_cntry.data
                    ),
                },
                coords={
                    'birth_year': ('birth_year', birth_years),
                    'time': ('time', da_population.time.data),
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                    'age': ('age', np.arange(100,-1,-1)),
                },
                attrs={
                    'global_life_expectancy':life_expectancy_global
                }
            )

            # get birthyear aligned population for unprecedented calculation (by_population), also use for weighted mean of lifetime exposure and age emergence
            bys = []
            for by in birth_years:
                    
                time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
                ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                data = ds_dmg['population'].sel(time=time,age=ages) # paired selections
                data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
                data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':by}).drop_vars('age')
                bys.append(data)

            # adding by_population_y0 to rep cohort sizes as number of people at birth year (y0)
            ds_dmg['by_population_y0'] = xr.concat(bys,dim='birth_year').where(ds_dmg['country_extent']==1)
            da_times=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')
            da_birth_years=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')        
            ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].sel(time=da_times,birth_year=da_birth_years)
            # set order of dims
            ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].transpose('birth_year','lat','lon')
            
            # pickle gridscale demography for country
            with open('./data/{}/gridscale_dmg_{}_le_test.pkl'.format(flags['version'],cntry), 'wb') as f:
                pk.dump(ds_dmg,f)       
                
        else:
            
            # load demography pickle
            with open('./data/{}/gridscale_dmg_{}_le_test.pkl'.format(flags['version'],cntry), 'rb') as f:
                ds_dmg = pk.load(f)                      
        
        # check for PIC lifetime exposure pickle file (for ds_pic); process and dump pickle if not already existing
        # 1) version of pic lifetime exposure and quantiles that is using countries' birth_year average life expectancy (i.e. mean life expectancy in belgium across 1960-2020)
        # after the pic quantiles are calculated and pickled ("ds_pic_qntl_le_country"), will delete these variables from memory but read them in when necessary below
            # this is because it will take too much memory to have both types of pic bootstrapped samples available (country le and global le)
        if not os.path.isfile('./data/{}/{}/{}/gridscale_le_country_constant_pic_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry)):
            
            # pic dataset for bootstrapped lifetimes
            ds_pic_le_country = xr.Dataset(
                data_vars={
                    'lifetime_exposure': (
                        ['lifetimes','lat','lon'],
                        np.full(
                            (len(list(d_pic_meta.keys())*nboots),len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    )
                },
                coords={
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                    'lifetimes': ('lifetimes', np.arange(len(list(d_pic_meta.keys())*nboots))),
                }
            )                
            
            # loop over PIC simulations
            c = 0
            for i in list(d_pic_meta.keys()):
                
                print('simulation {} of {}'.format(i,len(d_pic_meta)))
                
                # load AFA data of that run
                with open('./data/{}/{}/isimip_AFA_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
                    da_AFA_pic = pk.load(f)
                    
                da_AFA_pic = da_AFA_pic.where(ds_dmg['country_extent']==1,drop=True)
                
                # regular boot strapping for all reasonably sized countries (currently only exception is for Russia, might expand this for hazards with more sims)
                if not cntry in ['Russian Federation','Canada','United States','China']:

                    # resample 10000 lifetimes and then sum 
                    da_exposure_pic = xr.concat(
                        [resample(da_AFA_pic,resample_dim,int(life_expectancy_country[cntry])) for i in range(nboots)],
                        dim='lifetimes'    
                    ).assign_coords({'lifetimes':np.arange(c*nboots,c*nboots+nboots)})
                    
                    # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                    # super confused about whether this line does anything productive; I think I was just copying a single lifetime expsoure across all the different bootstraps in the ds_pic['lifetime_exposure'] assignment?
                    # must test
                    da_pic_le = da_exposure_pic.loc[
                        {'time':np.arange(pic_by,pic_by+life_expectancy_country[cntry])}
                    ].sum(dim='time')
                            
                    ds_pic_le_country['lifetime_exposure'].loc[
                        {
                            'lat': da_pic_le.lat.data,
                            'lon': da_pic_le.lon.data,
                            'lifetimes': np.arange(c*nboots,c*nboots+nboots),
                        }
                    ] = da_pic_le
                    c += 1
                   
                # for exceptionally sized countries, do piecewise boot strapping (nboots in 10 steps)
                else:
                    
                    for bstep in range(10):
                        
                        # resample 1000 lifetimes and then sum 
                        da_exposure_pic = xr.concat(
                            [resample(da_AFA_pic,resample_dim,int(life_expectancy_country[cntry])) for i in range(int(nboots/10))],
                            dim='lifetimes'    
                        ).assign_coords({'lifetimes':np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10)})       
                        
                        # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                        da_pic_le = da_exposure_pic.loc[
                            {'time':np.arange(pic_by,pic_by+life_expectancy_country[cntry])}
                        ].sum(dim='time')
                                
                        ds_pic_le_country['lifetime_exposure'].loc[
                            {
                                'lat': da_pic_le.lat.data,
                                'lon': da_pic_le.lon.data,
                                'lifetimes': np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10),
                            }
                        ] = da_pic_le
                        
                    c += 1
                    
            # pickle PIC lifetime exposure for var, country
            with open('./data/{}/{}/{}/gridscale_le_country_constant_pic_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_pic_le_country,f)                
                
        else:
            
            # load PIC pickle
            with open('./data/{}/{}/{}/gridscale_le_country_constant_pic_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'rb') as f:
                ds_pic_le_country = pk.load(f)     
                
        # check for PIC quantiles calc'd from bootstrapped lifetime exposures (for ds_pic_qntl); process and dump pickle if not already existing
        if not os.path.isfile('./data/{}/{}/{}/gridscale_le_country_constant_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry)):                
                         
            # pic dataset for quantiles
            ds_pic_qntl_le_country = xr.Dataset(
                data_vars={
                    '99.99': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.9': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.0': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),            
                    '97.5': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                     
                    '95.0': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                      
                    '90.0': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                                                                                          
                },
                coords={
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                }
            )                              
                                
            # pic extreme lifetime exposure definition (added more quantiles for v2)
            ds_pic_qntl_le_country['99.99'] = ds_pic_le_country['lifetime_exposure'].quantile(
                    q=pic_qntl,
                    dim='lifetimes',
                    method='closest_observation',
                )
            ds_pic_qntl_le_country['99.9'] = ds_pic_le_country['lifetime_exposure'].quantile(
                    q=0.999,
                    dim='lifetimes',
                    method='closest_observation',
                )            
            ds_pic_qntl_le_country['99.0'] = ds_pic_le_country['lifetime_exposure'].quantile(
                    q=0.99,
                    dim='lifetimes',
                    method='closest_observation',
                )           
            ds_pic_qntl_le_country['97.5'] = ds_pic_le_country['lifetime_exposure'].quantile(
                    q=0.975,
                    dim='lifetimes',
                    method='closest_observation',
                )                
            ds_pic_qntl_le_country['95.0'] = ds_pic_le_country['lifetime_exposure'].quantile(
                    q=0.95,
                    dim='lifetimes',
                    method='closest_observation',
                )     
            ds_pic_qntl_le_country['90.0'] = ds_pic_le_country['lifetime_exposure'].quantile(
                    q=0.9,
                    dim='lifetimes',
                    method='closest_observation',
                )     
                                             
            # pickle PIC lifetime exposure for var, country
            with open('./data/{}/{}/{}/gridscale_le_country_constant_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_pic_qntl_le_country,f)   
                
                
        else:
            
            # load PIC pickle
            with open('./data/{}/{}/{}/gridscale_le_country_constant_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'rb') as f:
                ds_pic_qntl_le_country = pk.load(f)          
           
        del ds_pic_le_country
                
        # 2) version of pic lifetime exposure and quantiles that is using global birth_year average life expectancy (i.e. mean life expectancy across all countries and 1960-2020)
        # after the pic quantiles are calculated and pickled ("ds_pic_qntl_le_country"), will delete these variables from memory but read them in when necessary below
            # this is because it will take too much memory to have both types of pic bootstrapped samples available (country le and global le)
        if not os.path.isfile('./data/{}/{}/{}/gridscale_le_global_constant_pic_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry)):
            
            # pic dataset for bootstrapped lifetimes
            ds_pic_le_global = xr.Dataset(
                data_vars={
                    'lifetime_exposure': (
                        ['lifetimes','lat','lon'],
                        np.full(
                            (len(list(d_pic_meta.keys())*nboots),len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    )
                },
                coords={
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                    'lifetimes': ('lifetimes', np.arange(len(list(d_pic_meta.keys())*nboots))),
                }
            )                
            
            # loop over PIC simulations
            c = 0
            for i in list(d_pic_meta.keys()):
                
                print('simulation {} of {}'.format(i,len(d_pic_meta)))
                
                # load AFA data of that run
                with open('./data/{}/{}/isimip_AFA_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
                    da_AFA_pic = pk.load(f)
                    
                da_AFA_pic = da_AFA_pic.where(ds_dmg['country_extent']==1,drop=True)
                
                # regular boot strapping for all reasonably sized countries (currently only exception is for Russia, might expand this for hazards with more sims)
                if not cntry in ['Russian Federation','Canada','United States','China']:

                    # resample 10000 lifetimes and then sum 
                    da_exposure_pic = xr.concat(
                        [resample(da_AFA_pic,resample_dim,int(life_expectancy_global)) for i in range(nboots)],
                        dim='lifetimes'    
                    ).assign_coords({'lifetimes':np.arange(c*nboots,c*nboots+nboots)})
                    
                    # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                    # super confused about whether this line does anything productive; I think I was just copying a single lifetime expsoure across all the different bootstraps in the ds_pic['lifetime_exposure'] assignment?
                    # must test
                    da_pic_le = da_exposure_pic.loc[
                        {'time':np.arange(pic_by,pic_by+life_expectancy_global)}
                    ].sum(dim='time')
                            
                    ds_pic_le_global['lifetime_exposure'].loc[
                        {
                            'lat': da_pic_le.lat.data,
                            'lon': da_pic_le.lon.data,
                            'lifetimes': np.arange(c*nboots,c*nboots+nboots),
                        }
                    ] = da_pic_le
                    c += 1
                   
                # for exceptionally sized countries, do piecewise boot strapping (nboots in 10 steps)
                else:
                    
                    for bstep in range(10):
                        
                        # resample 1000 lifetimes and then sum 
                        da_exposure_pic = xr.concat(
                            [resample(da_AFA_pic,resample_dim,int(life_expectancy_global)) for i in range(int(nboots/10))],
                            dim='lifetimes'    
                        ).assign_coords({'lifetimes':np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10)})       
                        
                        # like regular exposure, sum lifespan from birth to death year and add fracitonal exposure of death year
                        da_pic_le = da_exposure_pic.loc[
                            {'time':np.arange(pic_by,pic_by+life_expectancy_global)}
                        ].sum(dim='time')
                                
                        ds_pic_le_global['lifetime_exposure'].loc[
                            {
                                'lat': da_pic_le.lat.data,
                                'lon': da_pic_le.lon.data,
                                'lifetimes': np.arange(c*nboots + bstep*nboots/10,c*nboots + bstep*nboots/10 + nboots/10),
                            }
                        ] = da_pic_le
                        
                    c += 1
                    
            # pickle PIC lifetime exposure for var, country
            with open('./data/{}/{}/{}/gridscale_le_global_constant_pic_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_pic_le_global,f)                
                
        else:
            
            # load PIC pickle
            with open('./data/{}/{}/{}/gridscale_le_global_constant_pic_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'rb') as f:
                ds_pic_le_global = pk.load(f)     
                
        # check for PIC quantiles calc'd from bootstrapped lifetime exposures (for ds_pic_qntl); process and dump pickle if not already existing
        if not os.path.isfile('./data/{}/{}/{}/gridscale_le_global_constant_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry)):                
                         
            # pic dataset for quantiles
            ds_pic_qntl_le_global = xr.Dataset(
                data_vars={
                    '99.99': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.9': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.0': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),            
                    '97.5': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                     
                    '95.0': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                      
                    '90.0': (
                        ['lat','lon'],
                        np.full(
                            (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                                                                                          
                },
                coords={
                    'lat': ('lat', da_cntry.lat.data),
                    'lon': ('lon', da_cntry.lon.data),
                }
            )                              
                                
            # pic extreme lifetime exposure definition (added more quantiles for v2)
            ds_pic_qntl_le_global['99.99'] = ds_pic_le_global['lifetime_exposure'].quantile(
                    q=pic_qntl,
                    dim='lifetimes',
                    method='closest_observation',
                )
            ds_pic_qntl_le_global['99.9'] = ds_pic_le_global['lifetime_exposure'].quantile(
                    q=0.999,
                    dim='lifetimes',
                    method='closest_observation',
                )            
            ds_pic_qntl_le_global['99.0'] = ds_pic_le_global['lifetime_exposure'].quantile(
                    q=0.99,
                    dim='lifetimes',
                    method='closest_observation',
                )           
            ds_pic_qntl_le_global['97.5'] = ds_pic_le_global['lifetime_exposure'].quantile(
                    q=0.975,
                    dim='lifetimes',
                    method='closest_observation',
                )                
            ds_pic_qntl_le_global['95.0'] = ds_pic_le_global['lifetime_exposure'].quantile(
                    q=0.95,
                    dim='lifetimes',
                    method='closest_observation',
                )     
            ds_pic_qntl_le_global['90.0'] = ds_pic_le_global['lifetime_exposure'].quantile(
                    q=0.9,
                    dim='lifetimes',
                    method='closest_observation',
                )     
                                             
            # pickle PIC lifetime exposure for var, country
            with open('./data/{}/{}/{}/gridscale_le_global_constant_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'wb') as f:
                pk.dump(ds_pic_qntl_le_global,f)                
                
        else:
            
            # load PIC pickle
            with open('./data/{}/{}/{}/gridscale_le_global_constant_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry), 'rb') as f:
                ds_pic_qntl_le_global = pk.load(f)     
                
        del ds_pic_le_global

        # loop over simulations
        for i in list(d_isimip_meta.keys()): 

            print('simulation {} of {}'.format(i,len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
                da_AFA = pk.load(f)
                
            # mask to sample country and reduce spatial extent
            da_AFA = da_AFA.where(ds_dmg['country_extent']==1,drop=True)
            
            # loop over GMT trajectories
            for step in GMT_labels:
                
                # run GMT-mapping of years if valid
                # first use country-constant life expectancy
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    
                    # GMT-mapping
                    da_AFA_step = da_AFA.reindex(
                        {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                    ).assign_coords({'time':year_range})         
                    
                    # os.makedirs('./data/{}/{}/{}/{}'.format(flags['version'],flags['extr']+'_le_test',cntry)) # just for ref of directory addition for this life expectancy testing            
                    
                    # commented out this check because it can't incporporate multiple pthresh values (moved indent to left for code below)
                    # # check for pickles of gridscale exposure emergence mask and age emergence; os.mkdir('./data/{}/{}/{}'.format(flags['version'],flags['extr'],cntry))
                    # if not os.path.isfile('./data/{}/{}/{}/gridscale_emergence_mask_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry,i,step,pthresh)):                                 
                        
                    # area affected ('AFA') per year per age (in 'population')
                    da_exp_py_pa_cntry = da_AFA_step * xr.full_like(ds_dmg['population'],1)
                    bys_cc = [] # "cc" for country-constant
                    bys_gc = [] # "gc" for global-constant
            
                    # 1) country-constant life expectancy (same as gridscale_emergence(), but death_year and life expectancy already adapted above in ds_dmg)
                    # per birth year, make (year,age) paired selections
                    for by in birth_years:
                            
                        time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
                        ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                        data = da_exp_py_pa_cntry.sel(time=time,age=ages) # paired selections
                        data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
                        data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                        # when I reindex time, the len=1 lat coord for cyprus, a small country, disappears
                        # therefore need to reintroduce len=1 coord at correct position
                        for scoord in ['lat','lon']:
                            if data[scoord].size == 1:
                                if scoord == 'lat':
                                    a_pos = 1
                                elif scoord == 'lon':
                                    a_pos = 2
                                data = data.expand_dims(dim={scoord:1},axis=a_pos).copy()
                        data = data.assign_coords({'birth_year':by}).drop_vars('age')
                        # data.loc[
                        #     {'time':ds_dmg['death_year'].sel(birth_year=by).item()}
                        # ] = da_AFA_step.loc[{'time':ds_dmg['death_year'].sel(birth_year=by).item()}] *\
                        #     (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item())
                        bys_cc.append(data)
            
                    da_exp_py_pa_cntry = xr.concat(bys_cc,dim='birth_year')
                            
                    # cumulative sum per birthyear (in emergence.py, this cumsum then has .where(==0), should I add this here too?)
                    da_exp_py_pa_cntry_cumsum = da_exp_py_pa_cntry.cumsum(dim='time')
                    
                    # loop through pic thresholds for emergence (separate file per threshold or dataset? separate file)
                    for pthresh in pic_qntl_labels:
                        
                        # generate exposure mask for timesteps after reaching pic extreme to find emergence
                        da_emergence_mask = xr.where(
                            da_exp_py_pa_cntry_cumsum > ds_pic_qntl_le_country[pthresh],
                            1,
                            0,
                        )
                            
                        # find birth years/pixels crossing threshold
                        da_birthyear_emergence_mask = xr.where(da_emergence_mask.sum(dim='time')>0,1,0) 
                        
                        with open('./data/{}/{}/{}/gridscale_emergence_mask_le_country_constant_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry,i,step,pthresh), 'wb') as f:
                            pk.dump(da_birthyear_emergence_mask,f)
                            
                    # grid cells of population emerging for each PIC threshold
                    for pthresh in pic_qntl_labels:
                        
                        with open('./data/{}/{}/{}/gridscale_emergence_mask_le_country_constant_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry,i,step,pthresh), 'rb') as f:
                            da_birthyear_emergence_mask = pk.load(f)                        
                        
                        da_unprec_pop = ds_dmg['by_population_y0'].where(da_birthyear_emergence_mask==1)          
                        
                        ds_pf['unprec_{}_country_le'.format(pthresh)].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }] = da_unprec_pop.sum(('lat','lon'))

                        ds_pf['unprec_fraction_{}_country_le'.format(pthresh)].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }] = da_unprec_pop.sum(('lat','lon')) / ds_dmg['by_population_y0'].sum(('lat','lon'))
                        
                    # 2) global-constant life expectancy; use 'global_death_year' evolving with birth year and global_life_expectancy attribute from ds_dmg
                    # per birth year, make (year,age) paired selections
                    # redefine area affected ('AFA') per year per age (in 'population')
                    da_exp_py_pa_glbl = da_AFA_step * xr.full_like(ds_dmg['population'],1)                    
                    for by in birth_years:
                            
                        time = xr.DataArray(np.arange(by,ds_dmg['global_death_year'].sel(birth_year=by).item()+1),dims='cohort')
                        ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                        data = da_exp_py_pa_glbl.sel(time=time,age=ages) # paired selections
                        data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['global_death_year'].sel(birth_year=by).item()+1,dtype='int')})
                        data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                        # when I reindex time, the len=1 lat coord for cyprus, a small country, disappears
                        # therefore need to reintroduce len=1 coord at correct position
                        for scoord in ['lat','lon']:
                            if data[scoord].size == 1:
                                if scoord == 'lat':
                                    a_pos = 1
                                elif scoord == 'lon':
                                    a_pos = 2
                                data = data.expand_dims(dim={scoord:1},axis=a_pos).copy()
                        data = data.assign_coords({'birth_year':by}).drop_vars('age')
                        # data.loc[ # don't need to include fractional death year because I do some rounding above in the ds_dmg dataset (same with the country-constant version)
                        #     {'time':ds_dmg['death_year'].sel(birth_year=by).item()}
                        # ] = da_AFA_step.loc[{'time':ds_dmg['death_year'].sel(birth_year=by).item()}] *\
                        #     (ds_dmg['life_expectancy'].sel(birth_year=by).item() - np.floor(ds_dmg['life_expectancy'].sel(birth_year=by)).item())
                        bys_gc.append(data)
            
                    da_exp_py_pa_glbl = xr.concat(bys_gc,dim='birth_year')
                            
                    # cumulative sum per birthyear (in emergence.py, this cumsum then has .where(==0), should I add this here too?)
                    da_exp_py_pa_glbl_cumsum = da_exp_py_pa_glbl.cumsum(dim='time')
                    
                    # loop through pic thresholds for emergence (separate file per threshold or dataset? separate file)
                    for pthresh in pic_qntl_labels:
                        
                        # generate exposure mask for timesteps after reaching pic extreme to find emergence
                        da_emergence_mask = xr.where(
                            da_exp_py_pa_glbl_cumsum > ds_pic_qntl_le_global[pthresh],
                            1,
                            0,
                        )
                            
                        # find birth years/pixels crossing threshold
                        da_birthyear_emergence_mask = xr.where(da_emergence_mask.sum(dim='time')>0,1,0) 
                        
                        with open('./data/{}/{}/{}/gridscale_emergence_mask_le_global_constant_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry,i,step,pthresh), 'wb') as f:
                            pk.dump(da_birthyear_emergence_mask,f)
                            
                    # grid cells of population emerging for each PIC threshold
                    for pthresh in pic_qntl_labels:
                        
                        with open('./data/{}/{}/{}/gridscale_emergence_mask_le_global_constant_{}_{}_{}_{}_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',cntry,flags['extr'],cntry,i,step,pthresh), 'rb') as f:
                            da_birthyear_emergence_mask = pk.load(f)                        
                        
                        da_unprec_pop = ds_dmg['by_population_y0'].where(da_birthyear_emergence_mask==1)          
                        
                        ds_pf['unprec_{}_global_le'.format(pthresh)].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }] = da_unprec_pop.sum(('lat','lon'))

                        ds_pf['unprec_fraction_{}_global_le'.format(pthresh)].loc[{
                            'country':cntry,
                            'run':i,
                            'GMT':step,
                            'birth_year':birth_years,
                        }] = da_unprec_pop.sum(('lat','lon')) / ds_dmg['by_population_y0'].sum(('lat','lon'))                        
           
        # new labels for ref while coding
        # ds_pf['unprec_{}_country_le'.format(pthresh)] = ds_pf['unprec'].copy()
        # ds_pf['unprec_fraction_{}_country_le'.format(pthresh)] = ds_pf['unprec'].copy()
        # # new labelling for global constant life expectancy
        # ds_pf['unprec_{}_global_le'.format(pthresh)] = ds_pf['unprec'].copy()
        # ds_pf['unprec_fraction_{}_global_le'.format(pthresh)] = ds_pf['unprec'].copy()                                
                        
    # pickle aggregated lifetime exposure, age emergence and pop frac datasets
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_le_test_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',flags['extr']), 'wb') as f:
        pk.dump(ds_pf,f)
        
    return ds_pf



#%% ----------------------------------------------------------------
# grid scale population per birthyear
# ------------------------------------------------------------------

def get_gridscale_popdenom(
    list_countries,
    da_cohort_size,
    countries_mask,
    countries_regions,
    da_population,
    df_life_expectancy_5,
):
    # need denominator for pf @ grid scale, which is not just wcde data in ds_cohorts, but rather isimip pop data x fractions in ds_cohorts
    cntry_pops = []
    for cntry in list_countries:
    
        print(cntry)
        da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
        da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages') # cohort relative sizes in sample country
        da_cntry = xr.DataArray(
            np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
            dims=countries_mask.dims,
            coords=countries_mask.coords,
        )
        da_cntry = da_cntry.where(da_cntry,drop=True)
        da_smple_pop = da_population.where(da_cntry==1) * da_smple_cht_prp # use pop and relative cohort sizes to get people per cohort

        # demography dataset
        ds_dmg = xr.Dataset(
            data_vars={
                'life_expectancy': (
                    ['birth_year'],
                    df_life_expectancy_5[cntry].values
                ),
                'death_year': (
                    ['birth_year'],
                    np.floor(df_life_expectancy_5[cntry].values + df_life_expectancy_5[cntry].index).astype('int')
                ),
                'population': (
                    ['time','lat','lon','age'],
                    da_smple_pop.data
                ),
                'country_extent': (
                    ['lat','lon'],
                    da_cntry.data
                ),                
            },
            coords={
                'birth_year': ('birth_year', birth_years),
                'time': ('time', da_population.time.data),
                'lat': ('lat', da_cntry.lat.data),
                'lon': ('lon', da_cntry.lon.data),
                'age': ('age', np.arange(100,-1,-1)),
            }
        )

        # get birthyear aligned population for unprecedented calculation (by_population), also use for weighted mean of lifetime exposure and age emergence
        bys = []
        for by in birth_years:
                
            time = xr.DataArray(np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1),dims='cohort')
            ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
            data = ds_dmg['population'].sel(time=time,age=ages) # paired selections
            data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,ds_dmg['death_year'].sel(birth_year=by).item()+1,dtype='int')})
            data = data.reindex({'time':np.arange(year_start,year_end+1,dtype='int')}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
            data = data.assign_coords({'birth_year':by}).drop_vars('age')
            bys.append(data)

        ds_dmg['by_population_y0'] = xr.concat(bys,dim='birth_year').where(ds_dmg['country_extent']==1)
        da_times=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')
        da_birth_years=xr.DataArray(ds_dmg.birth_year.data,dims='birth_year')        
        ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].sel(time=da_times,birth_year=da_birth_years)
        ds_dmg['by_population_y0'] = ds_dmg['by_population_y0'].transpose('birth_year','lat','lon')     
        cntry_pops.append(ds_dmg['by_population_y0'].sum(dim=('lat','lon')))   
    
    da_cntry_pops = xr.concat(cntry_pops,dim='country').assign_coords({'country':list_countries})     
    return da_cntry_pops

#%% ----------------------------------------------------------------
# grid scale emergence union
# ------------------------------------------------------------------

def get_gridscale_union(
    da_population,
    flags,
    gridscale_countries,
    countries_mask,
    countries_regions,
):
    
    if not os.path.isfile('./data/{}/emergence_hazards_subset_new.pkl'.format(flags['version'])) or not os.path.isfile('./data/{}/emergence_union_subset_new.pkl'.format(flags['version'])):
        
        extremes = [
            'burntarea', 
            'cropfailedarea', 
            'driedarea', 
            'floodedarea', 
            'heatwavedarea', 
            'tropicalcyclonedarea',
        ]

        ds_emergence_union = xr.Dataset(
            data_vars={
                'emergence_mean': (
                    ['hazard','GMT','birth_year','lat','lon'],
                    np.full(
                        (len(extremes),len(GMT_labels),len(birth_years),len(da_population.lat.data),len(da_population.lon.data)),
                        fill_value=np.nan,
                    ),
                ),                           
                'emergence_union': (
                    ['GMT','birth_year','lat','lon'],
                    np.full(
                        (len(GMT_labels),len(birth_years),len(da_population.lat.data),len(da_population.lon.data)),
                        fill_value=np.nan,
                    ),
                ),        
            },
            coords={
                'lat': ('lat', da_population.lat.data),
                'lon': ('lon', da_population.lon.data),
                'birth_year': ('birth_year', birth_years),
                'hazard': ('hazard', extremes),
                'GMT': ('GMT', GMT_labels),
            }
        )

        # loop through extremes
        for extr in extremes:
            
            # get metadata for extreme
            with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
                d_isimip_meta = pk.load(f)
                
            sims_per_step = {}
            for step in GMT_labels:
                sims_per_step[step] = []
                print('step {}'.format(step))
                for i in list(d_isimip_meta.keys()):
                    if d_isimip_meta[i]['GMT_strj_valid'][step]:
                        sims_per_step[step].append(i)                
                
            # loop through countries
            for cntry in gridscale_countries:
                
                da_cntry = xr.DataArray(
                    np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                    dims=countries_mask.dims,
                    coords=countries_mask.coords,
                )
                da_cntry = da_cntry.where(da_cntry,drop=True)                  
                    
                # loop through GMT trajectories
                for step in GMT_labels:
                    
                    # dataset for extreme - country - GMT
                    ds_cntry_emergence = xr.Dataset(
                        data_vars={
                            'emergence': (
                                ['run','birth_year','lat','lon'],
                                np.full(
                                    (len(sims_per_step[step]),len(birth_years),len(da_cntry.lat.data),len(da_cntry.lon.data)),
                                    fill_value=np.nan,
                                ),
                            ),                          
                        },
                        coords={
                            'lat': ('lat', da_cntry.lat.data),
                            'lon': ('lon', da_cntry.lon.data),
                            'birth_year': ('birth_year', birth_years),
                            'run': ('run', sims_per_step[step]),
                        }
                    )                      
                
                    # loop through sims and pick emergence masks for sims that are valid
                    for i in sims_per_step[step]: 
                        
                        if d_isimip_meta[i]['GMT_strj_valid'][step]:
                        
                            with open('./data/{}/{}/gridscale_emergence_mask_{}_{}_{}_{}.pkl'.format(flags['version'],extr,extr,cntry,i,step), 'rb') as f:
                                da_birthyear_emergence_mask = pk.load(f)
                                
                            ds_cntry_emergence['emergence'].loc[{
                                'run':i,
                                'birth_year':birth_years,
                                'lat':da_cntry.lat.data,
                                'lon':da_cntry.lon.data,
                            }] = da_birthyear_emergence_mask
                            
                    # compute mean for extreme - country - GMT, assign into greater dataset for eventual union
                    da_loc_mean = ds_cntry_emergence['emergence'].loc[{
                        'run':sims_per_step[step],
                        'lat':da_cntry.lat.data,
                        'lon':da_cntry.lon.data,
                    }].where(da_cntry==1).mean(dim='run')
                    
                    ds_emergence_union['emergence_mean'].loc[{
                        'hazard':extr,
                        'GMT':step,
                        'birth_year':birth_years,
                        'lat':da_cntry.lat.data,
                        'lon':da_cntry.lon.data,
                    }] = xr.where(
                        da_loc_mean.notnull(),
                        da_loc_mean,
                        ds_emergence_union['emergence_mean'].loc[{'hazard':extr,'GMT':step,'birth_year':birth_years,'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                    )                    
                    
        # if mean greater than 0.5, consider emerged on average, sum across hazards for union        
        ds_emergence_union['emergence_union'].loc[{
            'GMT':GMT_labels,
            'birth_year':birth_years,
            'lat':da_population.lat.data,
            'lon':da_population.lon.data,    
        }] = xr.where(ds_emergence_union['emergence_mean']>0,1,0).sum(dim='hazard')
                        
        # pickle mean emergence
        with open('./data/{}/emergence_hazards.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_emergence_union['emergence_mean'],f)  
            
        # pickle emergence union
        with open('./data/{}/emergence_union.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_emergence_union['emergence_union'],f)     
            
        da_emergence_mean = ds_emergence_union['emergence_mean']
        da_emergence_union = ds_emergence_union['emergence_union']
        
        # get subsets of these large objects/pickles for plotting locally
        da_emergence_mean_subset = da_emergence_mean.loc[{
            'GMT':[0,10,12,17,20],
            'birth_year':[1960,1980,2000,2020],
        }]  
        da_emergence_union_subset = da_emergence_union.loc[{
            'GMT':[0,10,12,17,20],
            'birth_year':[1960,1980,2000,2020],
        }]  
        
        # pickle mean emergence
        with open('./data/{}/emergence_hazards_mean_subset.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(da_emergence_mean_subset,f)         
            
        # pickle emergence union
        with open('./data/{}/emergence_union_subset.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(da_emergence_union_subset,f)                   
            
    else:
        
        # # load mean emergence
        # with open('./data/{}/emergence_hazards.pkl'.format(flags['version']), 'rb') as f:
        #     da_emergence_mean = pk.load(f) 
            
        # # load emergence union
        # with open('./data/{}/emergence_union.pkl'.format(flags['version']), 'rb') as f:
        #     da_emergence_union = pk.load(f)            
            
        with open('./data/{}/emergence_hazards_subset_new.pkl'.format(flags['version']), 'rb') as f:
            da_emergence_mean_subset = pk.load(f)  
            
        # load emergence union
        with open('./data/{}/emergence_union_subset_new.pkl'.format(flags['version']), 'rb') as f:
            da_emergence_union_subset = pk.load(f)             
                     
        
        
    return da_emergence_mean_subset,da_emergence_union_subset

#%% ----------------------------------------------------------------
# proc emergences into global array for ar6 hexagons and otherwise
# ------------------------------------------------------------------

def collect_global_emergence(
    grid_area,
    flags,
    countries_mask,
    countries_regions,
    gridscale_countries,
    df_GMT_strj,
):
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    land_mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    ar6_land_3D = rm.defined_regions.ar6.land.mask_3D(lon,lat)
    grid_area_land = grid_area.where(land_mask.notnull())
    da_grid_area_total_ar6 = grid_area_land.where(ar6_land_3D).sum(dim=('lat','lon'))
    GMT_integers = [0,10,12,17,20]

    extremes = [
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]

    d_emergence_masks = {}

    for extr in extremes:
        
        d_emergence_masks[extr] = {}
                
        # this loop section is purely for emergence masks, expanding them to all birth years (1960-2020)                
        if not os.path.isfile('./data/{}/{}/emergence_masks_collected_{}.pkl'.format(flags['version'],extr,extr)):
            
            with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
                d_isimip_meta = pk.load(f)
                
            sims_per_step = {}
            for step in GMT_labels:
                sims_per_step[step] = []
                print('step {}'.format(step))
                for i in list(d_isimip_meta.keys()):
                    if d_isimip_meta[i]['GMT_strj_valid'][step]:
                        sims_per_step[step].append(i)

            # per current policies GMT trajectory, collect emergence masks
            for step in GMT_integers:
            
                ds_global_emergence = xr.Dataset(
                    data_vars={
                        'emergence_per_run_{}'.format(extr): (
                            ['qntl','run','birth_year','lat','lon'],
                            np.full(
                                (len(['99.99', '99.9']),len(sims_per_step[step]),len(birth_years),len(lat),len(lon)),
                                fill_value=np.nan,
                            ),
                        ),                                
                    },
                    coords={
                        'qntl': ('qntl', ['99.99', '99.9']),
                        'run': ('run', sims_per_step[step]),
                        'lat': ('lat', lat),
                        'lon': ('lon', lon),
                        'birth_year': ('birth_year', birth_years),
                    }
                )

                start_time = time.time()
                    
                    
                # loop through countries
                for i,cntry in enumerate(gridscale_countries):
                    
                    print('Country {}, {}'.format(i+1,cntry))
                    
                    da_cntry = xr.DataArray(
                        np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                        dims=countries_mask.dims,
                        coords=countries_mask.coords,
                    )
                    da_cntry = da_cntry.where(da_cntry,drop=True)     
                    
                    # cape verde
                    if cntry == 'Cape Verde':
                        pass
                    else:
                        # loop through sims and pick emergence masks for sims that are valid
                        for i in sims_per_step[step]: 
                            
                            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                                
                                # grid cells of population emerging for each PIC threshold
                                for pthresh in ['99.99', '99.9']:
                                    
                                    with open('./data/{}/{}/{}/gridscale_emergence_mask_{}_{}_{}_{}_{}.pkl'.format(flags['version'],extr,cntry,extr,cntry,i,step,pthresh), 'rb') as f:
                                        da_birthyear_emergence_mask = pk.load(f)
                                
                                    # make assignment of emergence mask to global emergence 
                                    ds_global_emergence['emergence_per_run_{}'.format(extr)].loc[{
                                        'qntl':pthresh,
                                        'run':i,
                                        'birth_year':birth_years,
                                        'lat':da_cntry.lat.data,
                                        'lon':da_cntry.lon.data,                
                                    }] = xr.where(
                                            da_cntry.notnull(),
                                            da_birthyear_emergence_mask.loc[{'birth_year':birth_years,'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                                            ds_global_emergence['emergence_per_run_{}'.format(extr)].loc[{'qntl':pthresh,'run':i,'birth_year':birth_years,'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                                        ).transpose('birth_year','lat','lon')
                                
                print("--- {} minutes for {} ---".format(
                    np.floor((time.time() - start_time) / 60),
                    extr
                    )
                        )   
                d_emergence_masks[extr][str(df_GMT_strj.loc[2100,step])] = ds_global_emergence.copy()
                
            with open('./data/{}/{}/emergence_masks_collected_{}.pkl'.format(flags['version'],extr,extr), 'wb') as f:
                pk.dump(d_emergence_masks[extr],f) 
            
        else:
                
            with open('./data/{}/{}/emergence_masks_collected_{}.pkl'.format(flags['version'],extr,extr), 'rb') as f:
                d_emergence_masks[extr] = pk.load(f)               
                
    return d_emergence_masks # note, this used to return d_global_emergence. adjusted this temporarily for the vulnerability analysis

#%% ----------------------------------------------------------------
# proc pic quantiles into global array for sensitivity analysis
# ------------------------------------------------------------------

def collect_pic_qntls(
    grid_area,
    flags,
    gridscale_countries,
    countries_mask,
    countries_regions,
):
    lat = grid_area.lat.values
    lon = grid_area.lon.values

    extremes = [
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]

    # this loop section is purely for emergence masks, expanding them to all birth years (1960-2020)                
    if not os.path.isfile('./data/{}/pic_qntls_collected.pkl'.format(flags['version'])):

        d_pic_qntls = {}

        for extr in extremes:
            
            d_pic_qntls[extr] = xr.Dataset(
                data_vars={
                    '99.99': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.9': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.0': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),            
                    '97.5': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                     
                    '95.0': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                      
                    '90.0': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                                 
                },
                coords={
                    'lat': ('lat', lat),
                    'lon': ('lon', lon),
                }
            )
            
            # loop through countries
            for i,cntry in enumerate(gridscale_countries):
                
                # cape verde
                if cntry == 'Cape Verde':
                    pass
                else:                
                
                    da_cntry = xr.DataArray(
                        np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                        dims=countries_mask.dims,
                        coords=countries_mask.coords,
                    )
                    da_cntry = da_cntry.where(da_cntry,drop=True)     
                    
                    # loop through countries and get pic qntls     
                    with open('./data/{}/{}/{}/gridscale_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'rb') as f:
                        ds_pic_qntl = pk.load(f) 
                    
                    # make assignment of emergence mask to global emergence 
                    d_pic_qntls[extr].loc[{
                        'lat':da_cntry.lat.data,
                        'lon':da_cntry.lon.data,                
                    }] = xr.where(
                            da_cntry.notnull(),
                            ds_pic_qntl.loc[{'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                            d_pic_qntls[extr].loc[{'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                        ).transpose('lat','lon')

            
        with open('./data/{}/pic_qntls_collected.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(d_pic_qntls,f) 
        
    else:
            
        with open('./data/{}/pic_qntls_collected.pkl'.format(flags['version']), 'rb') as f:
            d_pic_qntls = pk.load(f)        
                
    return d_pic_qntls

#%% ----------------------------------------------------------------
# proc pic quantiles into global array for sensitivity analysis; extra extreme pic quantiles
# ------------------------------------------------------------------

def collect_pic_qntls_extra(
    grid_area,
    flags,
    gridscale_countries,
    countries_mask,
    countries_regions,
):    
    
    lat = grid_area.lat.values
    lon = grid_area.lon.values

    extremes = [
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]

    d_pic_qntls = {}    
    
    # this loop section is purely for emergence masks, expanding them to all birth years (1960-2020)                
    if not os.path.isfile('./data/{}/pic_qntls_collected_extra.pkl'.format(flags['version'])):    
    
        for cntry in gridscale_countries:
        
            # load PIC pickle
            with open('./data/{}/{}/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'rb') as f:
                ds_pic = pk.load(f)     
                
            # load demography pickle
            with open('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry), 'rb') as f:
                ds_dmg = pk.load(f)               
                    
            # check for PIC quantiles calc'd from bootstrapped lifetime exposures (for ds_pic_qntl); process and dump pickle if not already existing
            if not os.path.isfile('./data/{}/{}/{}/gridscale_pic_qntls_extra_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry)):
                                
                # pic dataset for quantiles
                ds_pic_qntl = xr.Dataset(
                    data_vars={
                        '99.999': (
                            ['lat','lon'],
                            np.full(
                                (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                                fill_value=np.nan,
                            ),
                        ),
                        '99.9999': (
                            ['lat','lon'],
                            np.full(
                                (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                                fill_value=np.nan,
                            ),
                        ),
                        '99.99999': (
                            ['lat','lon'],
                            np.full(
                                (len(ds_dmg.lat.data),len(ds_dmg.lon.data)),
                                fill_value=np.nan,
                            ),
                        ),            
                                                                                        
                    },
                    coords={
                        'lat': ('lat', ds_dmg.lat.data),
                        'lon': ('lon', ds_dmg.lon.data),
                    }
                )                              
                                    
                # pic extreme lifetime exposure definition (added more quantiles for v2)
                ds_pic_qntl['99.999'] = ds_pic['lifetime_exposure'].quantile(
                        q=0.99999,
                        dim='lifetimes',
                        method='closest_observation',
                    )
                ds_pic_qntl['99.9999'] = ds_pic['lifetime_exposure'].quantile(
                        q=0.999999,
                        dim='lifetimes',
                        method='closest_observation',
                    )            
                ds_pic_qntl['99.99999'] = ds_pic['lifetime_exposure'].quantile(
                        q=0.9999999,
                        dim='lifetimes',
                        method='closest_observation',
                    )           
                                                    
                # pickle PIC lifetime exposure for var, country
                with open('./data/{}/{}/{}/gridscale_pic_qntls_extra_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'wb') as f:
                    pk.dump(ds_pic_qntl,f)      

        for extr in extremes:
            
            d_pic_qntls[extr] = xr.Dataset(
                data_vars={
                    '99.999': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.9999': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),
                    '99.99999': (
                        ['lat','lon'],
                        np.full(
                            (len(lat.data),len(lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                                     
                },
                coords={
                    'lat': ('lat', lat),
                    'lon': ('lon', lon),
                }
            )
            
            # loop through countries
            for i,cntry in enumerate(gridscale_countries):
                
                # cape verde
                if cntry == 'Cape Verde':
                    pass
                else:                
                
                    da_cntry = xr.DataArray(
                        np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                        dims=countries_mask.dims,
                        coords=countries_mask.coords,
                    )
                    da_cntry = da_cntry.where(da_cntry,drop=True)     
                    
                    # loop through countries and get pic qntls     
                    with open('./data/{}/{}/{}/gridscale_pic_qntls_extra_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'rb') as f:
                        ds_pic_qntl = pk.load(f) 
                    
                    # make assignment of emergence mask to global emergence 
                    d_pic_qntls[extr].loc[{
                        'lat':da_cntry.lat.data,
                        'lon':da_cntry.lon.data,                
                    }] = xr.where(
                            da_cntry.notnull(),
                            ds_pic_qntl.loc[{'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                            d_pic_qntls[extr].loc[{'lat':da_cntry.lat.data,'lon':da_cntry.lon.data}],
                        ).transpose('lat','lon')

            
        with open('./data/{}/pic_qntls_collected_extra.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(d_pic_qntls,f) 
        
    else:
            
        with open('./data/{}/pic_qntls_collected_extra.pkl'.format(flags['version']), 'rb') as f:
            d_pic_qntls = pk.load(f)        
                
    return d_pic_qntls

#%% ----------------------------------------------------------------
# proc emergences into averages across simulations for emergence fractions in ensemble
# ------------------------------------------------------------------

def get_mean_emergence(
    df_GMT_strj,
    flags,
    da_population,
    d_global_emergence,
):
    # additionally retrieve emergence fracs for 2.7 and 3.2 degrees (for figure SF5; this function substitutes earlier get_gridscale_union())
    if not os.path.isfile('./data/{}/emergence_means.pkl'.format(flags['version'])):
        
        GMTs_cp = [12,17] # "12" is for 2.7 degree pathway, "17" is for 3.2 degree pathway
        by_subset = [1960,2020]
        pthresholds=['99.99', '99.9'] # can only use pthresh's already available in d_global_emergence proc, which are 99.99 (og) and 99.9 (alternative)
        extremes = [
            'burntarea', 
            'cropfailedarea', 
            'driedarea', 
            'floodedarea', 
            'heatwavedarea', 
            'tropicalcyclonedarea',
        ]

        ds_emergence_mean = xr.Dataset(
            data_vars={
                'emergence_mean': (
                    ['hazard','qntl','GMT','birth_year','lat','lon'],
                    np.full(
                        (len(extremes),len(pthresholds),len(GMTs_cp),len(by_subset),len(da_population.lat.data),len(da_population.lon.data)),
                        fill_value=np.nan,
                    ),
                ),                           
            },
            coords={
                'lat': ('lat', da_population.lat.data),
                'lon': ('lon', da_population.lon.data),
                'birth_year': ('birth_year', by_subset),
                'hazard': ('hazard', extremes),
                'qntl': ('qntl', pthresholds),
                'GMT': ('GMT', GMTs_cp),
            }
        )

        # loop through extremes "d_emergence_masks[extr][str(df_GMT_strj.loc[2100,step])] = ds_global_emergence.copy()
        for extr in extremes:
            for step in GMTs_cp:
                for pthresh in pthresholds:
                    for by in by_subset:
                        ds_emergence_mean['emergence_mean'].loc[{
                            'hazard':extr,
                            'GMT':step,
                            'qntl':pthresh,
                            'birth_year':by,
                            'lat':da_population.lat.data,
                            'lon':da_population.lon.data,
                        }] = d_global_emergence[extr][str(df_GMT_strj.loc[2100,step])]['emergence_per_run_{}'.format(extr)].loc[{
                                'qntl':pthresh,
                                'birth_year':by,
                                'lat':da_population.lat.data,
                                'lon':da_population.lon.data,
                            }].mean(dim='run')
                        
        with open('./data/{}/emergence_means.pkl'.format(flags['version']), 'wb') as f:
                pk.dump(ds_emergence_mean,f) 
                
    else:
        
        with open('./data/{}/emergence_means.pkl'.format(flags['version']), 'rb') as f:
                ds_emergence_mean = pk.load(f)     

#%% ----------------------------------------------------------------
# proc gdp
# ------------------------------------------------------------------

def load_gdp_deprivation(
    flags,
    grid_area,
    da_population,
    countries_mask,
    countries_regions,
    gridscale_countries,
    df_life_expectancy_5,
):
  
    # ------------------------------------------------------------------
    # start with gdp data
    
    # define ssps and decades for wang and sun data
    ssps = ('ssp1','ssp2','ssp3','ssp4','ssp5')
    decades = ['2030','2040','2050','2060','2070','2080','2090','2100']
  
    # grid data
    lat = grid_area.lat.values
    lon = grid_area.lon.values  
  
    # if pickle isn't there, go through all GDP processing
    if not os.path.isfile('./data/{}/gdp_deprivation/gdp_means_dataset.pkl'.format(flags['version'])):
    
        # define xarray dataset for our GDP information
        ds_gdp = xr.Dataset(
            data_vars={
                'gdp_isimip_rcp26': ( # isimip histsoc + rcp26soc
                    ['year','lat','lon'],
                    np.full(
                        (len(year_range),len(lat),len(lon)),
                        fill_value=np.nan,
                    ),
                ),                       
            },
            coords={
                'year': ('year', year_range),
                'lat': ('lat', lat),
                'lon': ('lon', lon)
            }
        )    
        
        # add data arrays for wang and sun GDP over ssps
        for ssp in ssps:
            ds_gdp['gdp_ws_{}'.format(ssp)] = ( # will fill this data array with isimip grid gdp information across available years (1960-2113 for rcp26, 2005-2113 for wang & sun)
                ['year','lat','lon'],    
                np.full(
                    (len(year_range),len(lat),len(lon)),
                    fill_value=np.nan,
                )
            )     

        # read in ISIMIP grid data (histsoc + rcp26soc)
        f_gdp_historical_1861_2005 = './data/isimip/gdp/gdp_histsoc_0p5deg_annual_1861_2005.nc4'
        f_gdp_rcp26_2006_2099 = './data/isimip/gdp/gdp_rcp26soc_0p5deg_annual_2006_2099.nc4'
        da_gdp_historical_1861_2005 = open_dataarray_isimip(f_gdp_historical_1861_2005)
        da_gdp_rcp26_2006_2099 = open_dataarray_isimip(f_gdp_rcp26_2006_2099)    
        da_gdp_hist_rcp26 = xr.concat(
            [da_gdp_historical_1861_2005,da_gdp_rcp26_2006_2099],
            dim='time'
        ).sel(time=slice(year_start,2099)).where(countries_mask.notnull())    
        da_population_subset = da_population.sel(time=slice(year_start,2099))
        da_gdp_hist_rcp26_pc = da_gdp_hist_rcp26 / da_population_subset.where(da_population_subset>0) # get per capita GDP
        ds_gdp['gdp_isimip_rcp26'].loc[{
            'year': da_gdp_hist_rcp26_pc.time.data,
            'lat': lat,
            'lon': lon,
        }] = da_gdp_hist_rcp26_pc
        for y in np.arange(2100,2114): # repeat final year until end year of analysis
            ds_gdp['gdp_isimip_rcp26'].loc[{
                'year': y,
                'lat': lat,
                'lon': lon,
            }] = ds_gdp['gdp_isimip_rcp26'].loc[{'year':2099,'lat':lat,'lon':lon}]
        
        # Read in Wang and Sun data (2005, 2030-2100 in 10 yr steps)
        time_coords = decades.copy()
        time_coords.insert(0,'2005')
        time_coords = list(map(int,time_coords))
        dims_dict = { # rename dimensions
            'x':'lon',
            'y':'lat',
        }
        if len(glob.glob('./data/gdp_wang_sun/GDP_*_isimipgrid.nc4')) == 0: # run the Geotiff conversion and py-cdo stuff if proc'd file not there at ISIMIP resolution
            rds = {}
            for i,s in enumerate(ssps):
                rds[s] = {}
                rds[s]['2005'] = rxr.open_rasterio('./data/gdp_wang_sun/GDP2005.tif')
                for d in decades:
                    rds[s][d] = rxr.open_rasterio('./data/gdp_wang_sun/GDP{}_{}.tif'.format(d,s))
                rds[s]['full_series'] = xr.concat(
                    [rds[s][str(t)] for t in time_coords],
                    dim='time'
                ).squeeze(dim='band').rename(dims_dict)
                rds[s]['full_series'].coords['time']=time_coords
                ds = rds[s]['full_series'].to_dataset(name=s)
                if i == 0:
                    ds_fresh = xr.Dataset( # dataset for netcdf generation
                        data_vars={
                            s: (
                                ['time','lat','lon'],
                                np.full(
                                    (len(ds.time.data),len(ds.lat.data),len(ds.lon.data)),
                                    fill_value=np.nan,
                                ),
                            ),        
                        },
                        coords={
                            'time': ('time', time_coords),
                            'lat': ('lat', ds.lat.data, {
                                'standard_name': 'latitude',
                                'long_name': 'latitude',
                                'units': 'degrees_north',
                                'axis': 'Y'
                            }),
                            'lon': ('lon', ds.lon.data, {
                                'standard_name': 'longitude',
                                'long_name': 'longitude',
                                'units': 'degrees_east',
                                'axis': 'X'
                            })
                        }
                    )
                else:
                    ds_fresh[s] = xr.full_like(ds_fresh[ssps[0]],fill_value=np.nan)
                ds_fresh[s].loc[{
                    'time':ds_fresh.time.data,
                    'lat':ds_fresh.lat.data,
                    'lon':ds_fresh.lon.data,
                }] = ds[s]
                ds_fresh[s].to_netcdf( # save to netcdf
                    './data/gdp_wang_sun/GDP_{}.nc4'.format(s),
                    format='NETCDF4',
                    encoding={
                        'lat': {
                            'dtype': 'float64',
                        },
                        'lon': {
                            'dtype': 'float64',
                        }            
                    }
                )   
                # cdo.remapcon2( # remap netcdf
                #     './data/isimip/clm45_area.nc4',  
                #     input = './data/gdp_wang_sun/GDP_{}.nc4',
                #     output = './data/gdp_wang_sun/GDP_{}_isimipgrid.nc4'.format(s),
                #     options = '-f nc4'
                # )
                
            
        else:

            # read in remap'd netcdfs
            for i,s in enumerate(ssps):
                ds_gdp_regrid = xr.open_dataset('./data/gdp_wang_sun/GDP_{}_isimipgrid.nc4'.format(s))
                ds_gdp_regrid.coords['time'] = time_coords
                ds_gdp['gdp_ws_{}'.format(s)].loc[{
                    'year': time_coords,
                    'lat': lat,
                    'lon': lon,
                }] = ds_gdp_regrid[s] # interpolation doesn't work well on ds_gdp_regrid for some reason, so I do it below (before division by population)
            
            # for computing lifetime means, we will need to interpolate/extrapolate the gdp_ws_ssps between 2005 and 2113
            for v in list(ds_gdp.data_vars):
                if '_ws_' in v:
                    ds_gdp[v] = ds_gdp[v].interpolate_na(dim='year') # interpolation
                    for y in np.arange(2101,2114): # "extrapolate" or repeat 2100 until 2113
                        ds_gdp[v].loc[{
                            'year': y,
                            'lat': lat,
                            'lon': lon,
                        }] = ds_gdp[v].loc[{'year':2100,'lat':lat,'lon':lon}]
                    ds_gdp[v] = ds_gdp[v] / da_population.rename({'time':'year'}).loc[{'year':np.arange(2005,year_end+1)}] # get per capita gdp by diving by population

        # now compute means for each GDP series
        for v in list(ds_gdp.data_vars):
            if v != 'gdp_isimip_rcp26': # patching this to have 1960-2020 for isimiprcp26; for ssps, only do 2020 lifetime mean
                ds_gdp['{}_mean'.format(v)] = ( # will fill this data array with 2020 birth cohort mean
                    ['lat','lon'],    
                    np.full(
                        (len(lat),len(lon)),
                        fill_value=np.nan,
                    )   
                )
                # fill mean gdp based on country life expectancy
                for cntry in gridscale_countries:
                    da_cntry = countries_mask.where(countries_mask==countries_regions.map_keys(cntry)) # country mask
                    cntry_life_expectancy = df_life_expectancy_5.loc[2020,cntry] # country 2020 life expectancy
                    ds_gdp['{}_mean'.format(v)].loc[{
                        'lat':lat,
                        'lon':lon,
                    }] = xr.where( # will this replacement scheme work?
                        da_cntry.notnull(),
                        ds_gdp[v].where(da_cntry.notnull()).sel(year=np.arange(2020,2020+cntry_life_expectancy+1),method='nearest').mean(dim='year'),
                        ds_gdp['{}_mean'.format(v)]
                    )
            else: # use all birth years for isimip rcp26 dataset
                ds_gdp['{}_mean'.format(v)] = ( # will fill this data array with 1960-2020 birth cohort life expectancy means
                    ['birth_year','lat','lon'],    
                    np.full(
                        (len(birth_years),len(lat),len(lon)),
                        fill_value=np.nan,
                    )   
                )   
                ds_gdp['{}_mean'.format(v)] = ds_gdp['{}_mean'.format(v)].assign_coords({'birth_year':birth_years})
                for by in birth_years:
                    # fill mean gdp based on country life expectancy
                    for cntry in gridscale_countries:
                        da_cntry = countries_mask.where(countries_mask==countries_regions.map_keys(cntry)) # country mask
                        cntry_life_expectancy = df_life_expectancy_5.loc[by,cntry] # country 2020 life expectancy
                        ds_gdp['{}_mean'.format(v)].loc[{
                            'birth_year':by,
                            'lat':lat,
                            'lon':lon,
                        }] = xr.where( # will this replacement scheme work?
                            da_cntry.notnull(),
                            ds_gdp[v].where(da_cntry.notnull()).sel(year=np.arange(by,by+cntry_life_expectancy+1),method='nearest').mean(dim='year'),
                            ds_gdp['{}_mean'.format(v)].loc[{'birth_year':by}]
                        )                             
            
                
        # dump pickles
        with open('./data/{}/gdp_deprivation/gdp_dataset.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_gdp,f) # first one with all data (pretty large);
            
        with open('./data/{}/gdp_deprivation/gdp_means_dataset.pkl'.format(flags['version']), 'wb') as f:
            ds_gdp_means = ds_gdp.drop_vars([v for v in list(ds_gdp.data_vars) if not 'mean' in v])
            pk.dump(ds_gdp_means,f) # first one with just means (will read this in)        
            
    else:
        
        # load pickled aggregated lifetime exposure, age emergence and pop frac datasets
        with open('./data/{}/gdp_deprivation/gdp_means_dataset.pkl'.format(flags['version']), 'rb') as f:
            ds_gdp = pk.load(f)        

    # ------------------------------------------------------------------
    # load/proc deprivation data           
    
    # check for pickle
    if not os.path.isfile('./data/deprivation/grdi_con_nanreplace_isimipgrid.nc4'): # run the Geotiff conversion and py-cdo stuff if proc'd file not there
        
        rda = rxr.open_rasterio('./data/deprivation/povmap-grdi-v1.tif')
        ds_export = xr.Dataset( # dataset for netcdf generation
            data_vars={
                'grdi': (
                    ['lat','lon'],
                    np.full(
                        (len(rda.y.data),len(rda.x.data)),
                        fill_value=np.nan,
                    ),
                ),        
            },
            coords={
                'lat': ('lat', rda.y.data, {
                    'standard_name': 'latitude',
                    'long_name': 'latitude',
                    'units': 'degrees_north',
                    'axis': 'Y'
                }),
                'lon': ('lon', rda.x.data, {
                    'standard_name': 'longitude',
                    'long_name': 'longitude',
                    'units': 'degrees_east',
                    'axis': 'X'
                })
            }
        )
        ds_export['grdi'].loc[{
            'lat':ds_export.lat.data,
            'lon':ds_export.lon.data,
        }] = rda.squeeze()
        ds_export['grdi'] = ds_export['grdi'].where(ds_export['grdi']!=-9999.0)
        ds_export.to_netcdf( # save to netcdf
            './data/deprivation/grdi_30arcsec.nc4',
            format='NETCDF4',
            encoding={
                'lat': {
                    'dtype': 'float64',
                },
                'lon': {
                    'dtype': 'float64',
                }            
            }
        )     
        # cdo.remapcon( # remap netcdf
        #     './data/isimip/clm45_area.nc4',  
        #     input = './data/deprivation/grdi_30arcsec.nc4',
        #     output = './data/deprivation/grdi_con_isimipgrid.nc4'.format(s),
        #     options = '-f nc4'
        # ) 
    else:
    
        ds_grdi = xr.open_dataset('./data/deprivation/grdi_con_nanreplace_isimipgrid.nc4')
        
    return ds_gdp,ds_grdi
    
         
# %%
def timeseries_vulnerability(
    
):
    # use rioxarray to read geotiffs
    import rioxarray as rxr
    ssps = ('ssp1','ssp2','ssp3','ssp4','ssp5')
    decades = ['2030','2040','2050','2060','2070','2080','2090','2100']
    time_coords = decades.copy()
    time_coords.insert(0,'2005')
    time_coords = list(map(int,time_coords))
    dims_dict = { # rename dimensions
        'x':'lon',
        'y':'lat',
    }
    ssp_colors={
        'ssp1':'forestgreen',
        'ssp2':'slategrey',
        'ssp3':'purple',
        'ssp4':'darkorange',
        'ssp5':'firebrick',
    }

    # read in tiffs as data array; concat all years, clean up in new array, pump out to netcdf
    if len(glob.glob('./data/gdp_wang_sun/GDP_*_isimipgrid.nc4')) == 0:
        
        rds = {}
        for i,s in enumerate(ssps):
            rds[s] = {}
            rds[s]['2005'] = rxr.open_rasterio('./data/gdp_wang_sun/GDP2005.tif')
            for d in decades:
                rds[s][d] = rxr.open_rasterio('./data/gdp_wang_sun/GDP{}_{}.tif'.format(d,s))
            rds[s]['full_series'] = xr.concat(
                [rds[s][str(t)] for t in time_coords],
                dim='time'
            ).squeeze(dim='band').rename(dims_dict)
            rds[s]['full_series'].coords['time']=time_coords
            ds = rds[s]['full_series'].to_dataset(name=s)
            if i == 0:
                ds_fresh = xr.Dataset(
                    data_vars={
                        s: (
                            ['time','lat','lon'],
                            np.full(
                                (len(ds.time.data),len(ds.lat.data),len(ds.lon.data)),
                                fill_value=np.nan,
                            ),
                        ),        
                    },
                    coords={
                        'time': ('time', time_coords),
                        'lat': ('lat', ds.lat.data, {
                            'standard_name': 'latitude',
                            'long_name': 'latitude',
                            'units': 'degrees_north',
                            'axis': 'Y'
                        }),
                        'lon': ('lon', ds.lon.data, {
                            'standard_name': 'longitude',
                            'long_name': 'longitude',
                            'units': 'degrees_east',
                            'axis': 'X'
                        })
                    }
                )
            else:
                ds_fresh[s] = xr.full_like(ds_fresh[ssps[0]],fill_value=np.nan)
            ds_fresh[s].loc[{
                'time':ds_fresh.time.data,
                'lat':ds_fresh.lat.data,
                'lon':ds_fresh.lon.data,
            }] = ds[s]
            ds_fresh[s].to_netcdf(
                './data/gdp_wang_sun/GDP_{}.nc4'.format(s),
                format='NETCDF4',
                encoding={
                    'lat': {
                        'dtype': 'float64',
                    },
                    'lon': {
                        'dtype': 'float64',
                    }            
                }
            )
            
    # enter pycdo stuff for remapping to isimip grid        
            
    else:

        # start again, read in netcdf that were regridded on command line
        for i,s in enumerate(ssps):
            if i == 0:
                ds_gdp_regrid = xr.open_dataset('./data/gdp_wang_sun/GDP_{}_isimipgrid.nc4'.format(s))
                ds_gdp_regrid.coords['time'] = time_coords
            else:
                other_ssp = xr.open_dataset('./data/gdp_wang_sun/GDP_{}_isimipgrid.nc4'.format(s))[s]
                other_ssp.coords['time'] = time_coords
                ds_gdp_regrid[s] = other_ssp
            
    # associated years in da population
    da_population_subset = da_population.loc[{'time':ds_gdp_regrid.time.values}]


    # checking sample country time series for these ssps

    legend_lw=3.5 # legend line width
    x0 = 0.15 # bbox for legend
    y0 = 0.7
    xlen = 0.2
    ylen = 0.2    
    legend_entrypad = 0.5 # space between entries
    legend_entrylen = 0.75 # length per entry
    handles = [
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp1']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp2']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp3']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp4']),
        Line2D([0],[0],linestyle='-',lw=legend_lw,color=ssp_colors['ssp5']),
    ]  
    
    for cntry in ['Canada','United States','China', 'France', 'Germany']:
        
        print('')
        print('Entire series for {}'.format(cntry))  
        print('plots for {}'.format(cntry))
        f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

        for s in ssps:
            
            # color
            clr = ssp_colors[s]
            
            ds_gdp_regrid[s].where(ds_gdp_regrid[s]>0).where(countries_mask==countries_regions.map_keys(cntry)).sum(dim=('lat','lon')).plot(
                ax=ax1,
                color=clr,
                add_legend='False'
            )
            
            da_gdp_regrid_s = ds_gdp_regrid[s]
            da_gdp_regrid_pc = da_gdp_regrid_s.where(countries_mask==countries_regions.map_keys(cntry)) / da_population_subset.where(da_population_subset>0).where(countries_mask==countries_regions.map_keys(cntry))
            da_gdp_regrid_pc_mean = da_gdp_regrid_pc.mean(dim=('lat','lon'))    
            da_gdp_regrid_pc_mean.plot(
                ax=ax2,
                color=clr,
                add_legend=False,
            )
        ax1.set_title('GDP: {}'.format(cntry))
        ax2.set_title('GDP per capita: {}'.format(cntry))
        ax2.legend(
            handles, 
            list(ssp_colors.keys()), 
            bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
            loc='lower left',
            ncol=1,
            fontsize=10, 
            mode="expand", 
            borderaxespad=0.,
            frameon=False, 
            columnspacing=0.05, 
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad,
        )     
        
        plt.show() 
        
    # alternative testing for the Wang and Sun data while using ISIMIP historical GDP estimates
    f_gdp_historical_1861_2005 = './data/isimip/gdp/gdp_histsoc_0p5deg_annual_1861_2005.nc4'
    da_gdp_historical_1861_2005 = open_dataarray_isimip(f_gdp_historical_1861_2005)
    da_gdp_historical_1861_2004 = da_gdp_historical_1861_2005.loc[{'time':np.arange(1960,2005)}] 
        
    for cntry in ['Canada','United States','China', 'France', 'Germany']:
        
        print('')
        print('Entire series for {}'.format(cntry))  
        print('plots for {}'.format(cntry))
        f,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

        for s in ssps:
            
            future = ds_gdp_regrid[s]
            historical = da_gdp_historical_1861_2004
            da_gdp_full = xr.concat([historical,future],dim='time')
            da_population_subset = da_population.loc[{'time':da_gdp_full.time.values}]    
            
            # color
            clr = ssp_colors[s]
            
            da_gdp_full.where(da_gdp_full>0).where(countries_mask==countries_regions.map_keys(cntry)).sum(dim=('lat','lon')).plot(
                ax=ax1,
                color=clr,
                add_legend='False'
            )
            
            da_gdp_full_pc = da_gdp_full.where(countries_mask==countries_regions.map_keys(cntry)) / da_population_subset.where(da_population_subset>0).where(countries_mask==countries_regions.map_keys(cntry))
            da_gdp_full_pc = da_gdp_full_pc.mean(dim=('lat','lon'))    
            da_gdp_full_pc.plot(
                ax=ax2,
                color=clr,
                add_legend=False,
            )
        ax1.set_title('GDP: {}'.format(cntry))
        ax2.set_title('GDP per capita: {}'.format(cntry))
        ax2.legend(
            handles, 
            list(ssp_colors.keys()), 
            bbox_to_anchor=(x0, y0, xlen, ylen), # bbox: (x, y, width, height)
            loc='lower left',
            ncol=1,
            fontsize=10, 
            mode="expand", 
            borderaxespad=0.,
            frameon=False, 
            columnspacing=0.05, 
            handlelength=legend_entrylen, 
            handletextpad=legend_entrypad,
        )     
        
        plt.show()         
        
# %%        
def plot_maps_gdp_grdi():
    pass
    
        # # do some test plots of geo distribution of percentiles for gdp
    # for v in list(ds_gdp.data_vars):
    #     print('')
    #     print(v)
    #     f,axes = plt.subplots(
    #         nrows=2,
    #         ncols=3,
    #         figsize=(14,6),
    #         subplot_kw=dict(projection=ccrs.PlateCarree()),
    #         transform=ccrs.PlateCarree()
    #     )
    #     for ax,qntl in zip(axes.flatten(),[0.1,0.25,0.49,0.51,0.75,0.9]):
    #         qntl_gdp = ds_gdp[v].quantile(
    #             qntl,
    #             dim=('lat','lon'),
    #             method='closest_observation',
    #         )
    #         if qntl > 0.5:
    #             p = xr.where(ds_gdp[v]>=qntl_gdp.item(),1,np.nan).plot(
    #                 ax=ax,
    #                 add_colorbar=False,
    #             )
    #             ax.set_title('>{}th quantile'.format(qntl*100))
    #         elif qntl < 0.5:
    #             p = xr.where(ds_gdp[v]<=qntl_gdp.item(),1,np.nan).plot(
    #                 ax=ax,
    #                 add_colorbar=False,
    #             )
    #             ax.set_title('<{}th quantile'.format(qntl*100))
    #         ax.set_global()
    #         ax.coastlines()
    #     plt.show()

    # # then check percentiles of deprivation        
    # v = 'grdi'
    # print('')
    # print(v)
    # f,axes = plt.subplots(
    #     nrows=2,
    #     ncols=3,
    #     figsize=(14,6),
    #     subplot_kw=dict(projection=ccrs.PlateCarree()),
    #     transform=ccrs.PlateCarree()
    # )
    # for ax,qntl in zip(axes.flatten(),[0.1,0.25,0.49,0.51,0.75,0.9]):
    #     qntl_grdi = ds_grdi[v].quantile(
    #         qntl,
    #         dim=('lat','lon'),
    #         method='closest_observation',
    #     )
    #     if qntl > 0.5:
    #         p = xr.where(ds_grdi[v]>=qntl_grdi.item(),1,np.nan).plot(
    #             ax=ax,
    #             add_colorbar=False,
    #         )
    #         ax.set_title('>{}th quantile'.format(qntl*100))
    #     elif qntl < 0.5:
    #         p = xr.where(ds_grdi[v]<=qntl_grdi.item(),1,np.nan).plot(
    #             ax=ax,
    #             add_colorbar=False,
    #         )
    #         ax.set_title('<{}th quantile'.format(qntl*100))
    #     ax.set_global()
    #     ax.coastlines()
    # plt.show()                
    
#%% ----------------------------------------------------------------
# get quantiles of vulnerability (gdp and grdi) based on population
# ------------------------------------------------------------------

def get_vulnerability_quantiles(
    flags,
    grid_area,
    da_cohort_size_1960_2020,
    ds_gdp,
    ds_grdi,
):
       
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    qntls_vulnerability=np.arange(10)  
    
    if not os.path.exists('./data/{}/gdp_deprivation/'.format(flags['version'])):
        os.mkdir('./data/{}/gdp_deprivation/'.format(flags['version']))    
            
    # start with gdp
    if not os.path.isfile('./data/{}/gdp_deprivation/gdp_quantiles.pkl'.format(flags['version'])):
    
        # datasets in the return call
        ds_gdp_qntls = xr.Dataset(
            data_vars={
                'gdp_q_by_p': ( 
                    ['qntl','birth_year','lat','lon'],
                    np.full(
                        (len(qntls_vulnerability),len(birth_years),len(lat),len(lon)),
                        fill_value=np.nan,
                    ),
                ),                
                'gdp_ranks_q_by_p': ( 
                    ['qntl','birth_year','lat','lon'],
                    np.full(
                        (len(qntls_vulnerability),len(birth_years),len(lat),len(lon)),
                        fill_value=np.nan,
                    ),
                ),                               
            },
            coords={
                'qntl': ('qntl', qntls_vulnerability),
                'birth_year': ('birth_year', birth_years),
                'lat': ('lat', lat),
                'lon': ('lon', lon),
            },
        )     
    
        # get quantiles per birth year 
        for by in birth_years:
            
            gdp = ds_gdp['gdp_isimip_rcp26_mean'].sel(birth_year=by)
            pop = da_cohort_size_1960_2020.sel(birth_year=by)

            gdp = gdp.where(pop.notnull())
            pop = pop.where(gdp.notnull())

            # check that this worked by seeing len of non-nans
            if len(xr.DataArray(gdp.values.flatten())) == len(xr.DataArray(pop.values.flatten())):
                print('should only be using overlapping grid cells')

            #=======================
            # another attempt where I rename and assign coords to "dim_0" so that coord labels are preserved after dropping nans
            # will play around with order of sorting/dropping nans/binning here until I get population to bin with roughly equal group sizes, like above, 
            # NOTE this is the correct/working approach
            vulnerability = xr.DataArray(gdp.values.flatten())
            vulnerability = vulnerability.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(vulnerability))}) # have to do this so the coords are traceable back to the 2-D layout
            vulnerability_ranks = vulnerability.rank(dim='gridcell_number').round()
            vulnerability_indices = vulnerability_ranks.gridcell_number
            sorted_vr = vulnerability_ranks.sortby(vulnerability_ranks) # sort ranks of vulnerability
            sorted_vr_nonans = sorted_vr[sorted_vr.notnull()] # drop nans from sorted v-ranks
            sorted_v = vulnerability.sortby(vulnerability_ranks) # sort vulnerability
            sorted_v_nonans = sorted_v[sorted_v.notnull()] # drop nans from sorted vulnerability array

            pop_flat = xr.DataArray(pop.values.flatten())
            pop_flat = pop_flat.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(pop_flat))}) # have to do this so the coords are traceable back to the 2-D layout
            sorted_pop = pop_flat.sortby(vulnerability_ranks) # failed because gdp and pop need common mask
            sorted_pop_nonans = sorted_pop[sorted_pop.notnull()] # have to remove nans because they will all add 0s in cumulative sum, which messes with binning (forget how, but it did)
            sorted_pop_nonans_cumsum = sorted_pop_nonans.cumsum()
            sorted_pop_nonans_cumsum_pct = sorted_pop_nonans_cumsum / sorted_pop_nonans.sum()

            # test the bins on population
            sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum() # groups all even population!!!

            # bin vulnerability and ranked vulnerability
            sorted_v_nonans_bins = sorted_v_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10)
            sorted_vr_nonans_bins = sorted_vr_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10)
            sorted_v_nonans_bin_keys = list(sorted_v_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).groups.keys()) # keys per group for looping
            sorted_vr_nonans_bin_keys = list(sorted_vr_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).groups.keys()) # keys per group for looping
            template_for_groups = xr.full_like(vulnerability,fill_value=np.nan) # will copy this template for each group
            
            # assign vulnerability quantile ranges, reshaped to ISIMIP lat/lon grid to ds_gdp; for both ranks and original vulnerability estimates
            for i,(vgroup,vrgroup) in enumerate(zip(sorted_v_nonans_bin_keys,sorted_vr_nonans_bin_keys)):
                # print(vgroup)
                template_v = template_for_groups.copy() # for absolute vulnerability numbers
                template_vr = template_for_groups.copy() # for ranks of vulnerability (would like to see maps of both)
                template_v.loc[{'gridcell_number':sorted_v_nonans_bins[vgroup].gridcell_number}] = sorted_v_nonans_bins[vgroup]
                template_vr.loc[{'gridcell_number':sorted_vr_nonans_bins[vrgroup].gridcell_number}] = sorted_vr_nonans_bins[vrgroup]
                v_assigned_reshape = template_v.coarsen(gridcell_number=720).construct(gridcell_number=('lat', 'lon')).assign_coords({'lat':lat,'lon':lon})
                vr_assigned_reshape = template_vr.coarsen(gridcell_number=720).construct(gridcell_number=('lat', 'lon')).assign_coords({'lat':lat,'lon':lon})
                ds_gdp_qntls['gdp_q_by_p'].loc[{
                    'birth_year':by,
                    'qntl':i,
                    'lat':lat,
                    'lon':lon,
                }] = v_assigned_reshape
                ds_gdp_qntls['gdp_ranks_q_by_p'].loc[{
                    'birth_year':by,
                    'qntl':i,
                    'lat':lat,
                    'lon':lon,
                }] = vr_assigned_reshape 
        
        with open('./data/{}/gdp_deprivation/gdp_quantiles.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_gdp_qntls,f)   
            
    else:
        
        with open('./data/{}/gdp_deprivation/gdp_quantiles.pkl'.format(flags['version']), 'rb') as f:
            ds_gdp_qntls = pk.load(f)          
    
    # then grdi
    if not os.path.isfile('./data/{}/gdp_deprivation/grdi_quantiles.pkl'.format(flags['version'])):
    
        # datasets in the return call
        # note that the grdi has quantiles across birth_years, only because we're projecting the 2020 grdi versus 1960-2020 birth years
        ds_grdi_qntls = xr.Dataset(
            data_vars={
                'grdi_q_by_p': ( 
                    ['qntl','birth_year','lat','lon'],
                    np.full(
                        (len(qntls_vulnerability),len(birth_years),len(lat),len(lon)),
                        fill_value=np.nan,
                    ),
                ),                
                'grdi_ranks_q_by_p': ( 
                    ['qntl','birth_year','lat','lon'],
                    np.full(
                        (len(qntls_vulnerability),len(birth_years),len(lat),len(lon)),
                        fill_value=np.nan,
                    ),
                ),                               
            },
            coords={
                'qntl': ('qntl', qntls_vulnerability),
                'birth_year': ('birth_year', birth_years),
                'lat': ('lat', lat),
                'lon': ('lon', lon),
            },
        )     
    
        # get quantiles per birth year 
        for by in birth_years:
            
            grdi = ds_grdi['grdi']
            pop = da_cohort_size_1960_2020.sel(birth_year=by)

            grdi = grdi.where(pop.notnull())
            pop = pop.where(grdi.notnull())

            #=======================
            # another attempt where I rename and assign coords to "dim_0" so that coord labels are preserved after dropping nans
            # will play around with order of sorting/dropping nans/binning here until I get population to bin with roughly equal group sizes, like above, 
            # NOTE this is the correct/working approach
            vulnerability = xr.DataArray(grdi.values.flatten())
            vulnerability = vulnerability.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(vulnerability))}) # have to do this so the coords are traceable back to the 2-D layout
            vulnerability_ranks = vulnerability.rank(dim='gridcell_number').round()
            sorted_vr = vulnerability_ranks.sortby(vulnerability_ranks) # sort ranks of vulnerability
            sorted_vr_nonans = sorted_vr[sorted_vr.notnull()] # drop nans from sorted v-ranks
            sorted_v = vulnerability.sortby(vulnerability_ranks) # sort vulnerability
            sorted_v_nonans = sorted_v[sorted_v.notnull()] # drop nans from sorted vulnerability array

            pop_flat = xr.DataArray(pop.values.flatten())
            pop_flat = pop_flat.rename({'dim_0':'gridcell_number'}).assign_coords({'gridcell_number':range(len(pop_flat))}) # have to do this so the coords are traceable back to the 2-D layout
            sorted_pop = pop_flat.sortby(vulnerability_ranks) # failed because grdi and pop need common mask
            sorted_pop_nonans = sorted_pop[sorted_pop.notnull()]
            sorted_pop_nonans_cumsum = sorted_pop_nonans.cumsum()
            sorted_pop_nonans_cumsum_pct = sorted_pop_nonans_cumsum / sorted_pop_nonans.sum()

            # test the bins on population
            sorted_pop_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).sum() # groups all even population!!!

            # bin vulnerability and ranked vulnerability
            sorted_v_nonans_bins = sorted_v_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10)
            sorted_vr_nonans_bins = sorted_vr_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10)
            sorted_v_nonans_bin_keys = list(sorted_v_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).groups.keys()) # keys per group for looping
            sorted_vr_nonans_bin_keys = list(sorted_vr_nonans.groupby_bins(sorted_pop_nonans_cumsum_pct,bins=10).groups.keys()) # keys per group for looping
            template_for_groups = xr.full_like(vulnerability,fill_value=np.nan) # will copy this template for each group
            
            # assign vulnerability quantile ranges, reshaped to ISIMIP lat/lon grid to ds_grdi; for both ranks and original vulnerability estimates
            for i,(vgroup,vrgroup) in enumerate(zip(sorted_v_nonans_bin_keys,sorted_vr_nonans_bin_keys)):
                # print(vgroup)
                template_v = template_for_groups.copy() # for absolute vulnerability numbers
                template_vr = template_for_groups.copy() # for ranks of vulnerability (would like to see maps of both)
                template_v.loc[{'gridcell_number':sorted_v_nonans_bins[vgroup].gridcell_number}] = sorted_v_nonans_bins[vgroup]
                template_vr.loc[{'gridcell_number':sorted_vr_nonans_bins[vrgroup].gridcell_number}] = sorted_vr_nonans_bins[vrgroup]
                v_assigned_reshape = template_v.coarsen(gridcell_number=720).construct(gridcell_number=('lat', 'lon')).assign_coords({'lat':lat,'lon':lon})
                vr_assigned_reshape = template_vr.coarsen(gridcell_number=720).construct(gridcell_number=('lat', 'lon')).assign_coords({'lat':lat,'lon':lon})
                ds_grdi_qntls['grdi_q_by_p'].loc[{
                    'birth_year':by,
                    'qntl':i,
                    'lat':lat,
                    'lon':lon,
                }] = v_assigned_reshape
                ds_grdi_qntls['grdi_ranks_q_by_p'].loc[{
                    'birth_year':by,
                    'qntl':i,
                    'lat':lat,
                    'lon':lon,
                }] = vr_assigned_reshape 
        
        with open('./data/{}/gdp_deprivation/grdi_quantiles.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_grdi_qntls,f)   
            
    else:
        
        with open('./data/{}/gdp_deprivation/grdi_quantiles.pkl'.format(flags['version']), 'rb') as f:
            ds_grdi_qntls = pk.load(f)                      
        
    return ds_gdp_qntls, ds_grdi_qntls

#%% ----------------------------------------------------------------
# get quantiles of vulnerability (gdp and grdi) based on population
# ------------------------------------------------------------------

# vulnerability subsets of emergence (so crossing quantiles above with emergence masks to see unprecedented populations based on your vulnerablity group)
def emergence_by_vulnerability(
    flags,
    df_GMT_strj,
    ds_gdp,
    ds_grdi,
    da_cohort_size_1960_2020,
    d_global_emergence,
):
    
    if not os.path.isfile('./data/{}/emergence_by_vulnerability.pkl'.format(flags['version'])):
    
        extremes = [
            'burntarea', 
            'cropfailedarea', 
            'driedarea', 
            'floodedarea', 
            'heatwavedarea', 
            'tropicalcyclonedarea',
        ]
        qntls_vulnerability = range(10)
        indices_vulnerability = ['gdp_q_by_p','grdi_q_by_p'] # instead just rcp26 and grdi
        all_runs=np.arange(1,87)
        GMT_integers = [0,10,12,17,20]
        ds_vulnerability = xr.Dataset(
            data_vars={
                'heatwavedarea': (
                    ['run','qntl','vulnerability_index', 'GMT', 'birth_year'],
                    np.full(
                        (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(GMT_integers),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),
                'cropfailedarea': (
                    ['run','qntl','vulnerability_index', 'GMT', 'birth_year'],
                    np.full(
                        (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(GMT_integers),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),        
                'floodedarea': (
                    ['run','qntl','vulnerability_index', 'GMT', 'birth_year'],
                    np.full(
                        (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(GMT_integers),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),
                'burntarea': (
                    ['run','qntl','vulnerability_index', 'GMT', 'birth_year'],
                    np.full(
                        (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(GMT_integers),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),
                'driedarea': (
                    ['run','qntl','vulnerability_index', 'GMT', 'birth_year'],
                    np.full(
                        (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(GMT_integers),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),          
                'tropicalcyclonedarea': (
                    ['run','qntl','vulnerability_index', 'GMT', 'birth_year'],
                    np.full(
                        (len(all_runs),len(qntls_vulnerability),len(indices_vulnerability),len(GMT_integers),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),                                     
            },
            coords={
                'run': ('run', all_runs),
                'qntl': ('qntl', qntls_vulnerability),
                'vulnerability_index': ('vulnerability_index', indices_vulnerability),
                'GMT': ('GMT', GMT_integers),
                'birth_year': ('birth_year', birth_years)
            }
        )

        # get number people living unprecedented exposure for each vulnerability group
        for e in extremes:
            
            # first with gdp dataset
            v = 'gdp_q_by_p'
            for by in birth_years:
                for q in qntls_vulnerability:
                    qntl_gdp = ds_gdp[v].loc[{'birth_year':by,'qntl':q}]
                    for step in GMT_integers:  
                        for r in d_global_emergence[e][str(df_GMT_strj.loc[2100,step])]['emergence_per_run_{}'.format(e)].run.data:
                            da_emerge = d_global_emergence[e][str(df_GMT_strj.loc[2100,step])]['emergence_per_run_{}'.format(e)].sel(qntl='99.99',birth_year=by,run=r).copy()
                            da_emerge_constrained = da_emerge.where(qntl_gdp.notnull())
                            unprec_pop = xr.where(
                                da_emerge_constrained == 1,
                                da_cohort_size_1960_2020.sel(birth_year=by),
                                0
                            ).sum(dim=('lat','lon'))
                            ds_vulnerability['{}'.format(e)].loc[{'run':r,'qntl':q,'vulnerability_index':v,'GMT':step,'birth_year':by}] = unprec_pop.copy()
                            
            # then the grdi dataset 
            v = 'grdi_q_by_p'
            for by in birth_years:
                for q in qntls_vulnerability:
                    qntl_grdi = ds_grdi[v].loc[{'birth_year':by,'qntl':q}] # broadcasting 2020 quantiles of grdi against all birth years
                    for step in GMT_integers: 
                        for r in d_global_emergence[e][str(df_GMT_strj.loc[2100,step])]['emergence_per_run_{}'.format(e)].run.data:
                            da_emerge = d_global_emergence[e][str(df_GMT_strj.loc[2100,step])]['emergence_per_run_{}'.format(e)].sel(qntl='99.99',birth_year=by,run=r).copy()
                            da_emerge_constrained = da_emerge.where(qntl_grdi.notnull())
                            unprec_pop = xr.where(
                                da_emerge_constrained == 1,
                                da_cohort_size_1960_2020.sel(birth_year=by),
                                0
                            ).sum(dim=('lat','lon'))                            
                            ds_vulnerability['{}'.format(e)].loc[{'run':r,'qntl':q,'vulnerability_index':v,'GMT':step,'birth_year':by}] = unprec_pop.copy()
                    
        with open('./data/{}/emergence_by_vulnerability.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(ds_vulnerability,f) 
                    
    else:
        
        with open('./data/{}/emergence_by_vulnerability.pkl'.format(flags['version']), 'rb') as f:
            ds_vulnerability = pk.load(f)   
            
    return ds_vulnerability

# %%=========================================================================

def get_spatially_explicit_cohorts_1960_2020(
    flags,
    gridscale_countries,
    countries_mask,
    countries_regions,
    da_cohort_size,
    da_population
):

    # first get all cohorts (spatially explicit) for vulnerability population binning and for crossing with emergence masks
    if not os.path.isfile('./data/{}/1960-2020_cohort_sizes.pkl'.format(flags['version'])):
        
        da_cohort_size_1960_2020 = xr.concat(
            [xr.full_like(countries_mask,fill_value=np.nan) for by in birth_years],
            dim='birth_year'
        ).assign_coords({'birth_year':birth_years})
        
        for by in birth_years:
            
            for cntry in gridscale_countries:
                
                da_cntry = countries_mask.where(countries_mask == countries_regions.map_keys(cntry))
                cntry_cohort_frac = da_cohort_size.sel(country=cntry,time=by,ages=0) / da_cohort_size.sel(country=cntry,time=by).sum(dim='ages')
                da_cohort_size_1960_2020.loc[{'birth_year':by}] = xr.where(
                    da_cntry.notnull(),
                    da_population.sel(time=by).where(da_cntry.notnull()) * cntry_cohort_frac,
                    da_cohort_size_1960_2020.loc[{'birth_year':by}],
                )
            
        with open('./data/{}/1960-2020_cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
            pk.dump(da_cohort_size_1960_2020,f)
        
    else:
        
        with open('./data/{}/1960-2020_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
            da_cohort_size_1960_2020 = pk.load(f)
            
    return da_cohort_size_1960_2020   
