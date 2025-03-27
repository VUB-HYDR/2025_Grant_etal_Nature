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
import mapclassify as mc
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import interpolate
import cartopy.crs as ccrs
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

#%% ----------------------------------------------------------------
# make age+time selections to align exposure, cohort exposure and cohort sizes along birth year and time
# this is new version that should be faster because I limit the alignment to 1960 to 2020 (birth_years instead of year range)
# Jan 18 2023 briefly added extra statements to handle those born up to 2100
def calc_birthyear_align(
    da,
    df_life_expectancy,
    by_emergence,
):
    
    country_list = []
    
    # loop through countries
    for country in da.country.values:
        
        birthyear_list = []
        
        # per birth year, make (year,age) selections
        for by in by_emergence:
            
            # use life expectancy information where available (until 2020)
            if by <= year_ref:            
                
                death_year = by + np.ceil(df_life_expectancy.loc[by,country]) # since we don't have AFA, best to round life expec up and then multiply last year of exposure by fraction of final year lived
                time = xr.DataArray(np.arange(by,death_year+1),dims='cohort')
                ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                data = da.sel(country=country,time=time,ages=ages) # paired selections
                data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,death_year+1,dtype='int')})
                data = data.reindex({'time':year_range}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':by}).drop_vars('ages')
                data.loc[{'time':death_year}] = data.loc[{'time':death_year}] * (df_life_expectancy.loc[by,country] - np.floor(df_life_expectancy.loc[by,country]))
                birthyear_list.append(data)
            
            # after 2020, assume constant life expectancy    
            elif by > year_ref and by < year_end:
                
                death_year = by + np.ceil(df_life_expectancy.loc[year_ref,country]) #for years after 2020, just take 2020 life expectancy
                
                # if lifespan not encompassed by 2113, set death to 2113
                if death_year > year_end:
                    
                    death_year = year_end
                
                time = xr.DataArray(np.arange(by,death_year+1),dims='cohort')
                ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                data = da.sel(country=country,time=time,ages=ages) # paired selections
                data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,death_year+1,dtype='int')})
                data = data.reindex({'time':year_range}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':by}).drop_vars('ages')
                data.loc[{'time':death_year}] = data.loc[{'time':death_year}] * (df_life_expectancy.loc[year_ref,country] - np.floor(df_life_expectancy.loc[year_ref,country]))
                birthyear_list.append(data)
            
            # for 2100, use single year of exposure    
            elif by == 2100:
                
                time = xr.DataArray([2100],dims='cohort')
                ages = xr.DataArray([0],dims='cohort')
                data = da.sel(country=country,time=time,ages=ages)
                data = data.rename({'cohort':'time'}).assign_coords({'time':[year_end]})
                data = data.reindex({'time':year_range}).squeeze()
                data = data.assign_coords({'birth_year':by}).drop_vars('ages')
                birthyear_list.append(data)            
        
        da_data = xr.concat(birthyear_list,dim='birth_year')
        country_list.append(da_data)
        
    da_all = xr.concat(country_list,dim='country')
    
    return da_all
#%% ----------------------------------------------------------------
# create dataset out of birthyear aligned cohort exposure and regular rexposure, add cumulative exposure, 
def ds_exposure_align(
    da,
):
        
    da_cumsum = da.cumsum(dim='time').where(da>0)
    ds_exposure_cohort = xr.Dataset(
        data_vars={
            'exposure': (da.dims, da.data),
            'exposure_cumulative': (da_cumsum.dims, da_cumsum.data)
        },
        coords={
            'country': ('country', da.country.data),
            'birth_year': ('birth_year', da.birth_year.data),
            'time': ('time', da.time.data),
        },
    )
     
    return ds_exposure_cohort

#%% ----------------------------------------------------------------
# create dataset out of birthyear aligned cohort sizes
def ds_cohort_align(
    da,
    da_aligned,
):
    da_t = da.sum(dim='ages') # population over time
    da_by = da_aligned.sum(dim='time') # population over birth years (with duplicate counting as we sum across birth cohorts lifespan)
    # population over birth years represented by first year of lifespan
    da_times=xr.DataArray(da_aligned.birth_year.data,dims='birth_year')
    da_birth_years=xr.DataArray(da_aligned.birth_year.data,dims='birth_year')
    da_by_y0 = da_aligned.sel(time=da_times,birth_year=da_birth_years)
    ds_cohort_sizes = xr.Dataset(
        data_vars={
            'by_population': (da_by.dims,da_by.data), # for each birth year, get total number of people for by_weights (population in ds; duplicate counting risk with time sum)
            'by_population_y0': (da_by_y0.dims,da_by_y0.data),
            'population': (da_aligned.dims,da_aligned.data), # population per birth year distributed over time for emergence mask
            't_population': (da_t.dims,da_t.data) # population per timestep across countries for t_weights
        },
        coords={
            'country': ('country',da_aligned.country.data),
            'birth_year': ('birth_year',da_aligned.birth_year.data),
            'time': ('time',da_aligned.time.data),
            'ages': ('ages',da.ages.data)
        },
    )
    
    ds_cohort_sizes['by_weights'] = (ds_cohort_sizes['by_population'] / ds_cohort_sizes['by_population'].sum(dim='country')) # add cohort weights to dataset
    ds_cohort_sizes['by_y0_weights'] = (ds_cohort_sizes['by_population_y0'] / ds_cohort_sizes['by_population_y0'].sum(dim='country')) # add cohort weights to dataset
    ds_cohort_sizes['t_weights'] = (ds_cohort_sizes['t_population'] / ds_cohort_sizes['t_population'].sum(dim='country')) # add cohort weights to dataset    
    
     
    return ds_cohort_sizes

#%% ----------------------------------------------------------------
# function to generate mask of unprecedented timesteps per birth year and age of emergence
def exposure_pic_masking(
    ds_emergence_mask,
    ds_exposure_pic,
):
    
    age_emergence_list = []
    emergence_mask_list = []
    ds_exposure_pic['ext'] = ds_exposure_pic['ext'].where(ds_exposure_pic['ext']>0)
    
    for c in ds_emergence_mask.country.data:
        
        # generate exposure mask for timesteps after reaching pic extreme to find age of emergence
        da_age_emergence_mask = xr.where(
            ds_emergence_mask['exposure_cumulative'].sel(country=c) > ds_exposure_pic['ext'].sel(country=c),
            1,
            0,
        )
        da_age_emergence = da_age_emergence_mask * (da_age_emergence_mask.time - da_age_emergence_mask.birth_year)
        da_age_emergence = da_age_emergence.where(da_age_emergence!=0).min(dim='time',skipna=True)
        age_emergence_list.append(da_age_emergence)
            
        # adjust exposure mask; for any birth cohorts that crossed extreme, keep 1s at all lived time steps and 0 for other birth cohorts
        da_birthyear_emergence_mask = xr.where(da_age_emergence_mask.sum(dim='time')>0,1,0) # find birth years crossing threshold
        da_birthyear_emergence_mask = xr.where(ds_emergence_mask['exposure'].sel(country=c).notnull(),1,0).where(da_birthyear_emergence_mask==1) # first get array of 1s for timesteps when lifetimes exist in aligned exposure array, then only keep birthyears crossing threshold in emergence mask
        da_emergence_mask = xr.where(da_birthyear_emergence_mask==1,1,0) # turn missing values to 0
        emergence_mask_list.append(da_emergence_mask)
        
    # concat across countries
    da_age_emergence = xr.concat(
        age_emergence_list,
        dim='country',
    )
    da_emergence_mask = xr.concat(
        emergence_mask_list,
        dim='country',
    )    
    
    # turn age emergence into dataset
    ds_age_emergence = xr.Dataset(
        data_vars={
            'age_emergence': (da_age_emergence.dims, da_age_emergence.data)
        },
        coords={
            'country': ('country', da_age_emergence.country.data),
            'birth_year': ('birth_year', da_age_emergence.birth_year.data),
        },        
    )        
    
    return da_emergence_mask,ds_age_emergence

#%% ----------------------------------------------------------------
# function to find number of people with unprecedented exposure 
def calc_unprec_exposure(
    ds_exposure_cohort,
    da_emergence_mask,
    ds_cohorts,
):

    # new empty dataset with variables for population experiencing unprecedented exposure
    ds_pop_frac = xr.Dataset(
        data_vars={
            'unprec_exposed_b': (['birth_year'], np.empty((len(ds_exposure_cohort.birth_year.data)))),
            'unprec_country_exposed_b': (['country','birth_year'], np.empty((len(ds_cohorts.country.data),len(ds_exposure_cohort.birth_year.data)))),
            'unprec_country_exposed_b_y0': (['country','birth_year'], np.empty((len(ds_cohorts.country.data),len(ds_exposure_cohort.birth_year.data)))),
            'unprec_all_b': (['birth_year'], np.empty((len(ds_exposure_cohort.birth_year.data)))),
            'unprec_all_b_y0': (['birth_year'], np.empty((len(ds_exposure_cohort.birth_year.data)))),
            'unprec_country_b_y0': (['country','birth_year'], np.empty((len(ds_cohorts.country.data),len(ds_exposure_cohort.birth_year.data)))),
            'unprec_all_t': (['time'], np.empty((len(year_range)))),
        },
        coords={
            'birth_year': ('birth_year', ds_exposure_cohort.birth_year.data),
            'time': ('time', year_range),
            'country': ('country', ds_cohorts.country.data)
        }
    )
    
    # -------------------------------------------
    # birth year based pop frac
    
    # keep people exposed during birth year's timesteps if cumulative exposure exceeds pic defined extreme
    unprec_exposed = ds_exposure_cohort['exposure'].where(da_emergence_mask==1)
    unprec_country_exposed_b = unprec_exposed.sum(dim='time')
    da_times=xr.DataArray(ds_pop_frac.birth_year.data,dims='birth_year')
    da_birth_years=xr.DataArray(ds_pop_frac.birth_year.data,dims='birth_year')   
    unprec_country_exposed_b_y0 = unprec_exposed.sel(time=da_times,birth_year=da_birth_years)
    unprec_exposed_b = unprec_exposed.sum(dim=['time','country'])
    # for stylized trajectories this "were" avoids including a bunch of 0 pop frac runs that got calc'd from nans from checking maxdiff criteria
    ds_pop_frac['unprec_exposed_b'].loc[{'birth_year':ds_exposure_cohort.birth_year.data}] = unprec_exposed_b.where(unprec_exposed_b!=0)    
    ds_pop_frac['unprec_country_exposed_b'].loc[{'birth_year':ds_exposure_cohort.birth_year.data}] = unprec_country_exposed_b.where(unprec_country_exposed_b!=0)    
    ds_pop_frac['unprec_country_exposed_b_y0'].loc[{'birth_year':ds_exposure_cohort.birth_year.data}] = unprec_country_exposed_b_y0.where(unprec_country_exposed_b_y0!=0)    
    
    
    # keep all people/members of birth year cohort if birth year's timesteps cum exposure exceeds pic extreme
    da_birthyear_emergence_mask = xr.where(da_emergence_mask.sum(dim='time')>0,1,0)
    # represent birth cohort by sum of their whole lifespan
    ds_pop_frac['unprec_all_b'].loc[{'birth_year':ds_exposure_cohort.birth_year.data}] = ds_cohorts['by_population'].where(da_birthyear_emergence_mask==1).sum(dim='country')
    # represent birth cohort by first year of their lifespan
    ds_pop_frac['unprec_all_b_y0'].loc[{'birth_year':ds_exposure_cohort.birth_year.data}] = ds_cohorts['by_population_y0'].where(da_birthyear_emergence_mask==1).sum(dim='country')
    # keep absolute numbers per country that emerge
    ds_pop_frac['unprec_country_b_y0'].loc[{'birth_year':ds_exposure_cohort.birth_year.data}] = ds_cohorts['by_population_y0'].where(da_birthyear_emergence_mask==1)
   
    
    # -------------------------------------------
    # time based pop frac
    
    # keep people exposed during birth year's timesteps if cumulative exposure exceeds pic defined extreme
    ds_pop_frac['unprec_all_t'].loc[{'time':year_range}] = ds_cohorts['population'].where(da_emergence_mask==1).sum(dim=('birth_year','country'))
     
    return ds_pop_frac

#%% ----------------------------------------------------------------
# function to run stats on pop frac across runs
def pop_frac_stats(
    ds_pop_frac,
    ds_cohorts,
):
    
    # stats on population emergence types
    ds_pop_frac['mean_unprec_exposed_b'] = ds_pop_frac['unprec_exposed_b'].mean(dim='run')
    ds_pop_frac['max_unprec_exposed_b'] = ds_pop_frac['unprec_exposed_b'].max(dim='run')
    ds_pop_frac['min_unprec_exposed_b'] = ds_pop_frac['unprec_exposed_b'].min(dim='run')
    ds_pop_frac['std_unprec_exposed_b'] = ds_pop_frac['unprec_exposed_b'].std(dim='run')
    ds_pop_frac['mean_unprec_country_exposed_b'] = ds_pop_frac['unprec_country_exposed_b'].mean(dim='run')
    ds_pop_frac['max_unprec_country_exposed_b'] = ds_pop_frac['unprec_country_exposed_b'].max(dim='run')
    ds_pop_frac['min_unprec_country_exposed_b'] = ds_pop_frac['unprec_country_exposed_b'].min(dim='run')
    ds_pop_frac['std_unprec_country_exposed_b'] = ds_pop_frac['unprec_country_exposed_b'].std(dim='run')    
    ds_pop_frac['mean_unprec_country_exposed_b_y0'] = ds_pop_frac['unprec_country_exposed_b_y0'].mean(dim='run')
    ds_pop_frac['max_unprec_country_exposed_b_y0'] = ds_pop_frac['unprec_country_exposed_b_y0'].max(dim='run')
    ds_pop_frac['min_unprec_country_exposed_b_y0'] = ds_pop_frac['unprec_country_exposed_b_y0'].min(dim='run')
    ds_pop_frac['std_unprec_country_exposed_b_y0'] = ds_pop_frac['unprec_country_exposed_b_y0'].std(dim='run')        
    ds_pop_frac['mean_unprec_all_b'] = ds_pop_frac['unprec_all_b'].mean(dim='run')
    ds_pop_frac['mean_unprec_all_b_y0'] = ds_pop_frac['unprec_all_b_y0'].mean(dim='run')
    ds_pop_frac['max_unprec_all_b'] = ds_pop_frac['unprec_all_b'].max(dim='run')
    ds_pop_frac['max_unprec_all_b_y0'] = ds_pop_frac['unprec_all_b_y0'].max(dim='run')
    ds_pop_frac['min_unprec_all_b'] = ds_pop_frac['unprec_all_b'].min(dim='run')
    ds_pop_frac['min_unprec_all_b_y0'] = ds_pop_frac['unprec_all_b_y0'].min(dim='run')
    ds_pop_frac['std_unprec_all_b'] = ds_pop_frac['unprec_all_b'].std(dim='run')
    ds_pop_frac['std_unprec_all_b_y0'] = ds_pop_frac['unprec_all_b_y0'].std(dim='run')
    
    ds_pop_frac['mean_unprec_all_t'] = ds_pop_frac['unprec_all_t'].mean(dim='run')
    ds_pop_frac['max_unprec_all_t'] = ds_pop_frac['unprec_all_t'].max(dim='run')
    ds_pop_frac['min_unprec_all_t'] = ds_pop_frac['unprec_all_t'].min(dim='run')
    ds_pop_frac['std_unprec_all_t'] = ds_pop_frac['unprec_all_t'].std(dim='run')
    
    # unprecedented exposure as fraction of total population estimate
    ds_pop_frac['frac_unprec_exposed_b'] = ds_pop_frac['unprec_exposed_b'] / ds_cohorts['by_population'].sum(dim=['country'])
    ds_pop_frac['frac_unprec_country_exposed_b'] = ds_pop_frac['unprec_country_exposed_b'] / ds_cohorts['by_population'].sum(dim=['country'])
    ds_pop_frac['frac_unprec_country_exposed_b_y0'] = ds_pop_frac['unprec_country_exposed_b_y0'] / ds_cohorts['by_population_y0'].sum(dim=['country'])
    ds_pop_frac['mean_frac_unprec_exposed_b'] = ds_pop_frac['mean_unprec_exposed_b'] / ds_cohorts['by_population'].sum(dim=['country'])
    ds_pop_frac['std_frac_unprec_exposed_b'] = ds_pop_frac['frac_unprec_exposed_b'].std(dim='run')
    ds_pop_frac['frac_unprec_all_b'] = ds_pop_frac['unprec_all_b'] / ds_cohorts['by_population'].sum(dim=['country'])
    ds_pop_frac['frac_unprec_all_b_y0'] = ds_pop_frac['unprec_all_b_y0'] / ds_cohorts['by_population_y0'].sum(dim=['country'])
    ds_pop_frac['mean_frac_unprec_all_b'] = ds_pop_frac['frac_unprec_all_b'].mean(dim='run')
    ds_pop_frac['mean_frac_unprec_all_b_y0'] = ds_pop_frac['frac_unprec_all_b_y0'].mean(dim='run')
    ds_pop_frac['std_frac_unprec_all'] = ds_pop_frac['frac_unprec_all_b'].std(dim='run')
    ds_pop_frac['std_frac_unprec_all_y0'] = ds_pop_frac['frac_unprec_all_b_y0'].std(dim='run')
    
    ds_pop_frac['frac_unprec_all_t'] = ds_pop_frac['unprec_all_t'] / ds_cohorts['population'].sum(dim=('birth_year','country'))
    ds_pop_frac['mean_frac_unprec_all_t'] = ds_pop_frac['frac_unprec_all_t'].mean(dim='run')
    ds_pop_frac['std_frac_unprec_all_t'] = ds_pop_frac['frac_unprec_all_t'].std(dim='run')        
    
    return ds_pop_frac

#%% ----------------------------------------------------------------
# get timing and EMF of exceedence of pic-defined extreme
def calc_exposure_emergence(
    ds_exposure,
    ds_exposure_pic,
    ds_age_emergence,
    gdf_country_borders,
):

    mmm_subset = [
        'mmm_15',
        'mmm_20',
        'mmm_NDC',
    ]

    # get years where mmm exposures under different trajectories exceed pic 99.99%
    ds_exposure_emergence = ds_exposure[mmm_subset].where(ds_exposure[mmm_subset] > ds_exposure_pic['ext'])
    ds_exposure_emergence_birth_year = ds_exposure_emergence.birth_year.where(ds_exposure_emergence.notnull()).min(dim='birth_year',skipna=True)#.astype('int')
    
    scen_subset = [
        '15',
        '20',
        'NDC',
    ]
    
    for scen in scen_subset:
        
        ds_exposure_emergence_birth_year['birth_year_age_{}'.format(scen)] = ds_age_emergence['age_emergence_{}'.format(scen)].where(
            ds_age_emergence['age_emergence_{}'.format(scen)].birth_year==ds_exposure_emergence_birth_year['mmm_{}'.format(scen)]
        ).mean(dim='run').min(dim='birth_year',skipna=True)    
    
    # move emergence birth years and EMFs to gdf for plotting
    gdf_exposure_emergence_birth_year = gpd.GeoDataFrame(ds_exposure_emergence_birth_year.to_dataframe().join(gdf_country_borders))
    
    return gdf_exposure_emergence_birth_year


#%% ----------------------------------------------------------------
def strj_emergence(
    d_isimip_meta,
    df_life_expectancy_5,
    ds_exposure_pic,
    ds_cohorts,
    by_emergence,
    flags,
):
    start_time = time.time()    
    
    # create new template datasets for pop_frac and age_emergence, make assignments for each run and GMT trajectory
    ds_pop_frac = xr.Dataset(
        data_vars={
            'unprec_exposed_b': (
                ['run','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(by_emergence),len(GMT_labels)),
                    fill_value=np.nan,
                ),
            ),
            'unprec_country_exposed_b': (
                ['run','country','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(ds_cohorts.country.data),len(by_emergence),len(GMT_labels)),
                    fill_value=np.nan,
                ),
            ),            
            'unprec_country_exposed_b_y0': (
                ['run','country','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(ds_cohorts.country.data),len(by_emergence),len(GMT_labels)),
                    fill_value=np.nan,
                ),
            ),                       
            'unprec_all_b': (
                ['run','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(by_emergence),len(GMT_labels)),
                    fill_value=np.nan
                ),
            ),
            'unprec_all_b_y0': (
                ['run','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(by_emergence),len(GMT_labels)),
                    fill_value=np.nan
                ),
            ),            
            'unprec_country_b_y0': (
                ['run','country','birth_year','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(ds_cohorts.country.data),len(by_emergence),len(GMT_labels)),
                    fill_value=np.nan
                ),
            ),            
            'unprec_all_t': (
                ['run','time','GMT'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(year_range),len(GMT_labels)),
                    fill_value=np.nan
                ),
            ),            
        },
        coords={
            'run': ('run', list(d_isimip_meta.keys())),
            'birth_year': ('birth_year', by_emergence),
            'country': ('country', ds_cohorts.country.data),
            'GMT': ('GMT', GMT_labels),
            'time': ('time', year_range),
        }
    )
    
    ds_age_emergence = xr.Dataset(
        data_vars={
            'age_emergence': (
                ['country','birth_year','run','GMT'], 
                np.full(
                    (len(ds_cohorts.country.data),len(by_emergence),len(list(d_isimip_meta.keys())),len(GMT_labels)),
                    fill_value=np.nan,
                )
            ),
        },
        coords={
            'country': ('country', ds_cohorts.country.data),
            'birth_year': ('birth_year', by_emergence),
            'run': ('run', list(d_isimip_meta.keys())),
            'GMT': ('GMT', GMT_labels),
        },        
    )
    
    # loop through sims and run emergence analysis
    for i in list(d_isimip_meta.keys()):
        
        with open('./data/{}/{}/exposure_cohort_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i), 'rb') as f:
            da_exposure_cohort = pk.load(f)
        with open('./data/{}/{}/exposure_peryear_perage_percountry_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i), 'rb') as f:
            da_exposure_peryear_perage_percountry = pk.load(f)
    
        for step in da_exposure_peryear_perage_percountry.GMT.values:
            
            print('Processing GMT step {} of {}, run {}'.format(step,len(GMT_labels),i))
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                
                # check for pickles before running full proc
                if not os.path.isfile('./data/{}/{}/ds_exposure_aligned_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step)):
                    
                    # align age + time selections of annual mean exposure along birthyears + time per country, birth cohort, run; to act as mask for birthyears when pic threshold is passed
                    da_exposure_aligned = calc_birthyear_align(
                        da_exposure_peryear_perage_percountry.sel(GMT=step),
                        df_life_expectancy_5,
                        by_emergence,
                    )
                    
                    # dataset of birthyear aligned exposure, add cumulative exposure 
                    ds_exposure_aligned_run_step = ds_exposure_align(
                        da_exposure_aligned,
                    )
                    
                    # pickle
                    with open('./data/{}/{}/ds_exposure_aligned_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'wb') as f:
                        pk.dump(ds_exposure_aligned_run_step,f)
                    
                else:
                    
                    # load pickle
                    with open('./data/{}/{}/ds_exposure_aligned_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'rb') as f:
                        ds_exposure_aligned_run_step = pk.load(f)
                
                # check for pickles before running
                if not os.path.isfile('./data/{}/{}/da_emergence_mask_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step)) or not os.path.isfile('./data/{}/{}/ds_age_emergence_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step)):
                    
                    # use birthyear aligned (cumulative) exposure and pic extreme to extract age of emergence and get mask to include all lived timesteps per birthyear that passed pic threshold
                    da_emergence_mask_run_step, ds_age_emergence_run_step = exposure_pic_masking(
                        ds_exposure_aligned_run_step,
                        ds_exposure_pic,
                    )
                    
                    # pickle
                    with open('./data/{}/{}/da_emergence_mask_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'wb') as f:
                        pk.dump(da_emergence_mask_run_step,f)
                    with open('./data/{}/{}/ds_age_emergence_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'wb') as f:
                        pk.dump(ds_age_emergence_run_step,f)
                    
                else:
                    
                    # load pickle
                    with open('./data/{}/{}/da_emergence_mask_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'rb') as f:
                        da_emergence_mask_run_step = pk.load(f)
                
                # check for pickles before running
                if not os.path.isfile('./data/{}/{}/ds_exposure_cohort_aligned_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step)):
                
                    # aligned cohort exposure age+time selections
                    da_exposure_cohort_aligned_run_step = calc_birthyear_align(
                        da_exposure_cohort.sel(GMT=step),
                        df_life_expectancy_5,
                        by_emergence,
                    )
                    
                    # convert aligned cohort exposure to dataset
                    ds_exposure_cohort_aligned_run_step = ds_exposure_align(
                        da_exposure_cohort_aligned_run_step,
                    )
                    
                    # pickle
                    with open('./data/{}/{}/ds_exposure_cohort_aligned_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'wb') as f:
                        pk.dump(ds_exposure_cohort_aligned_run_step,f)
                        
                else:
                    
                    # load pickle
                    with open('./data/{}/{}/ds_exposure_cohort_aligned_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'rb') as f:
                        ds_exposure_cohort_aligned_run_step = pk.load(f)
                
                # check for pop frac before running; don't load on else since loading at end for total dataset of all runs/GMT steps
                if not os.path.isfile('./data/{}/{}/ds_pop_frac_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step)):
                    
                    # population experiencing normal vs unprecedented exposure
                    ds_pop_frac_run_step = calc_unprec_exposure(
                        ds_exposure_cohort_aligned_run_step,
                        da_emergence_mask_run_step,
                        ds_cohorts,
                    )
                    
                    # pickle
                    with open('./data/{}/{}/ds_pop_frac_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'wb') as f:
                        pk.dump(ds_pop_frac_run_step,f)           
                        
                    # regional emergence analysis (make country selection for wetting/drying regions before country sum in calc_unprec_exposure)
    
    # loop through sims and and GMT levels and assign age emergence and pop frac to full datasets
    for i in list(d_isimip_meta.keys()):
    
        for step in da_exposure_peryear_perage_percountry.GMT.values:   
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]: 
            
                with open('./data/{}/{}/ds_age_emergence_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'rb') as f:
                    ds_age_emergence_run_step = pk.load(f)
                with open('./data/{}/{}/ds_pop_frac_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i,step), 'rb') as f:
                    ds_pop_frac_run_step = pk.load(f)                     
                
                # assign run/GMT to larger dataset
                ds_age_emergence.loc[{
                    'country':ds_cohorts.country.data,
                    'birth_year':by_emergence,
                    'run': i,
                    'GMT':step,
                }] = ds_age_emergence_run_step    
                
                # exposed population experiencing unprecedented exposure
                ds_pop_frac['unprec_exposed_b'].loc[{
                    'run':i,
                    'birth_year':by_emergence,
                    'GMT':step,
                }] = ds_pop_frac_run_step['unprec_exposed_b']
                
                # exposed population experiencing unprecedented exposure at the country level
                ds_pop_frac['unprec_country_exposed_b'].loc[{
                    'run':i,
                    'country':ds_cohorts.country.data,
                    'birth_year':by_emergence,
                    'GMT':step,
                }] = ds_pop_frac_run_step['unprec_country_exposed_b']
                
                # exposed population experiencing unprecedented exposure at the country level
                ds_pop_frac['unprec_country_exposed_b_y0'].loc[{
                    'run':i,
                    'country':ds_cohorts.country.data,
                    'birth_year':by_emergence,
                    'GMT':step,
                }] = ds_pop_frac_run_step['unprec_country_exposed_b_y0']                
                
                # population experiencing unprecedented exposure using birth-year lumping approach and summing birth year cohort sizes over time
                ds_pop_frac['unprec_all_b'].loc[{
                    'run':i,
                    'birth_year':by_emergence,
                    'GMT':step,
                }] = ds_pop_frac_run_step['unprec_all_b']
                
                # population experiencing unprecedented exposure using birth-year lumping approach and assuming cohort size == birth year size
                ds_pop_frac['unprec_all_b_y0'].loc[{
                    'run':i,
                    'birth_year':by_emergence,
                    'GMT':step,
                }] = ds_pop_frac_run_step['unprec_all_b_y0']  
                
                # country-level population experiencing unprecedented exposure (unprec_all_b_y0 before summing across countries)
                ds_pop_frac['unprec_country_b_y0'].loc[{
                    'country':ds_cohorts.country.data,
                    'run':i,
                    'birth_year':by_emergence,
                    'GMT':step,
                }] = ds_pop_frac_run_step['unprec_country_b_y0']                    
                
                # population unprecedented exposure over time
                ds_pop_frac['unprec_all_t'].loc[{
                    'run':i,
                    'time':year_range,
                    'GMT':step,
                }] = ds_pop_frac_run_step['unprec_all_t']                             
                              
    # run ensemble stats across runs
    ds_pop_frac = pop_frac_stats(
        ds_pop_frac,
        ds_cohorts,
    )
        
    # pickle pop frac for individual run
    with open('./data/{}/{}/pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_pop_frac,f)
    
    # pickle age emergence for individual run
    with open('./data/{}/{}/age_emergence_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_age_emergence,f)
            
    print("--- {} minutes ---".format(
        np.floor((time.time() - start_time) / 60),
        )
    )

    return ds_age_emergence, ds_pop_frac


# %%
