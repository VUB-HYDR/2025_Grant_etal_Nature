# ---------------------------------------------------------------
# Functions to compute exposure
# ----------------------------------------------------------------


from operator import index
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import pickle as pk
from scipy import interpolate
from scipy import stats as sts
import regionmask as rm
import glob
import time
import matplotlib.pyplot as plt
from copy import deepcopy as cp
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()


#%% ----------------------------------------------------------------
# linear regression
def lreg(x, y):
    # Wrapper around scipy linregress to use in apply_ufunc
    slope, intercept, r_value, p_value, std_err = sts.linregress(x, y)
    return np.array([slope, p_value, r_value])

#%% ----------------------------------------------------------------
# apply vectorized linear regression
def vectorize_lreg(da_y,
                   da_x=None):
    
    if da_x is not None:
        
        pass
    
    else:
        
        da_list = []
        for t in da_y.time.values:
            da_list.append(xr.where(da_y.sel(time=t).notnull(),t,da_y.sel(time=t)))
        da_x = xr.concat(da_list,dim='time')
        
    stats = xr.apply_ufunc(lreg, da_x, da_y,
                           input_core_dims=[['time'], ['time']],
                           output_core_dims=[["parameter"]],
                           vectorize=True,
                           dask="parallelized",
                           output_dtypes=['float64'],
                           output_sizes={"parameter": 3})
    slope = cp(stats.sel(parameter=0))
    return slope

#%% ----------------------------------------------------------------
# bootstrapping function 
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
# *improved function to compute extreme event exposure across a person's lifetime
def calc_life_exposure(
    df_exposure,
    df_life_expectancy,
    col,
):

    # initialise birth years 
    exposure_birthyears_percountry = np.empty(len(df_life_expectancy))

    for i, birth_year in enumerate(df_life_expectancy.index):

        life_expectancy = df_life_expectancy.loc[birth_year,col] 

        # define death year based on life expectancy
        death_year = birth_year + np.floor(life_expectancy)

        # integrate exposure over full years lived
        exposure_birthyears_percountry[i] = df_exposure.loc[birth_year:death_year,col].sum()

        # add exposure during last (partial) year
        exposure_birthyears_percountry[i] = exposure_birthyears_percountry[i] + \
            df_exposure.loc[death_year+1,col].sum() * \
                (life_expectancy - np.floor(life_expectancy))

    # a series for each column to somehow group into a dataframe
    exposure_birthyears_percountry = pd.Series(
        exposure_birthyears_percountry,
        index=df_life_expectancy.index,
        name=col,
    )

    return exposure_birthyears_percountry

#%% ----------------------------------------------------------------
# calculated weighted fieldmean per country mask
def calc_weighted_fldmean(
    da, 
    weights, 
    countries_mask, 
    ind_country, 
    flag_region,
):

    # one country provided, easy masking
    if not flag_region : 
        da_masked = da.where(countries_mask == ind_country)
    
    # if more countries are provided, combine the different masks 
    else: 
        
        if len(ind_country) > 1:
            
            mask = xr.DataArray(
                np.in1d(countries_mask,ind_country).reshape(countries_mask.shape),
                dims=countries_mask.dims,
                coords=countries_mask.coords,
            )
            da_masked = da.where(mask)
    
    da_weighted_fldmean = da_masked.weighted(weights).mean(dim=("lat", "lon"))
    
    return da_weighted_fldmean

#%% ----------------------------------------------------------------
# get member countries per region
def get_countries_of_region(
    region, 
    df_countries,
): 

    # Get list of member countries from region
    member_countries = df_countries.loc[df_countries['region']==region]['name'].values

    # not region but income group
    if len(member_countries) == 0: 
        member_countries = df_countries.loc[df_countries['incomegroup']==region]['name'].values

    # get all countries for the world
    if region == 'World':
        member_countries = df_countries['name'].values

    return member_countries    

#%% ----------------------------------------------------------------
# function to compute multi-model mean across ISIMIP simulations based on mf_exposure_mmm.m
def calc_exposure_mmm_xr(
    ds_le,
):
        
    # take stats
    mmm = ds_le['lifetime_exposure'].mean(dim='run')
    std = ds_le['lifetime_exposure'].std(dim='run')
    lqntl = ds_le['lifetime_exposure'].quantile(
        q=0.25,
        dim='run',
        method='inverted_cdf'
    )
    uqntl = ds_le['lifetime_exposure'].quantile(
        q=0.75,
        dim='run',
        method='inverted_cdf'
    )
    
    # get EMF of stats (divide by 60 yr old / 1960 cohort)
    mmm_EMF = mmm / mmm.sel(birth_year=1960)
    lqntl_EMF = lqntl / mmm.sel(birth_year=1960)
    uqntl_EMF = uqntl / mmm.sel(birth_year=1960)
    
    # assemble into dataset
    ds_le['mmm'] = mmm
    ds_le['std'] = std
    ds_le['lqntl'] = lqntl
    ds_le['uqntl'] = uqntl
    ds_le['mmm_EMF'] = mmm_EMF
    ds_le['lqntl_EMF'] = lqntl_EMF
    ds_le['uqntl_EMF'] = uqntl_EMF    
    
    return ds_le
    
#%% ----------------------------------------------------------------
# function to compute multi-model mean across ISIMIP simulations based on mf_exposure_mmm.m
def calc_exposure_mmm_pic_xr(
    d_exposure_pic,
    dim_1_name,
    var_tag,
):        
        
    # concat pic data array from dict of separate arrays
    da_exposure_pic = xr.concat(
        [v for v in d_exposure_pic.values()],
        dim='runs',    
    ).assign_coords({'runs':list(d_exposure_pic.keys())})

    # runs and lifetimes (lifetimes from boostrapping) redundant, so compile together
    da_exposure_pic = da_exposure_pic.stack(
        pic_lifetimes=['runs','lifetimes'],
    )

    # pic exposure stats for EMF
    da_exposure_pic_mmm = da_exposure_pic.mean(dim='pic_lifetimes')
    da_exposure_pic_std = da_exposure_pic.std(dim='pic_lifetimes')
    da_exposure_pic_lqntl = da_exposure_pic.quantile(
        q=0.25,
        dim='pic_lifetimes',
        method='inverted_cdf',
    )
    da_exposure_pic_uqntl = da_exposure_pic.quantile(
        q=0.75,
        dim='pic_lifetimes',
        method='inverted_cdf',
    )    

    # pic quantile for birth cohort exposure emergence
    da_exposure_pic_ext = da_exposure_pic.quantile(
        q=0.9999,
        dim='pic_lifetimes',
        # method='inverted_cdf',
    )
    
    # assemble into dataset
    ds_exposure_pic_stats = xr.Dataset(
        data_vars={
            'mmm_{}'.format(var_tag): ([dim_1_name],da_exposure_pic_mmm.data),
            'std_{}'.format(var_tag): ([dim_1_name],da_exposure_pic_std.data),
            'lqntl_{}'.format(var_tag): ([dim_1_name],da_exposure_pic_lqntl.data),
            'uqntl_{}'.format(var_tag): ([dim_1_name],da_exposure_pic_uqntl.data),
            'ext': ([dim_1_name],da_exposure_pic_ext.data),
        },
        coords={
            dim_1_name: (dim_1_name,da_exposure_pic[dim_1_name].data),
        }
    )

    return ds_exposure_pic_stats

#%% ----------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span
def calc_exposure_trends(
    d_isimip_meta,
    grid_area,
    gdf_country_borders,
    flags,
):

    # arrays of lat/lon values
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    
    # 3d mask for ar6 regions
    ar6_regs_3D = rm.defined_regions.ar6.land.mask_3D(lon,lat)
    
    # 3d mask for countries
    countries_3D = rm.mask_3D_geopandas(gdf_country_borders.reset_index(),lon,lat)
    
    # basin shapefiles
    gdf_basins = gpd.read_file('./data/shapefiles/Major_Basins_of_the_World.shp')
    gdf_basins = gdf_basins.loc[:,['NAME','geometry']]
    
    # merge basins with multiple entries
    basins_grouped = []
    bc = {k:0 for k in gdf_basins['NAME']} # bc for basin counter
    for b_name in gdf_basins['NAME']:
        if len(gdf_basins.loc[gdf_basins['NAME']==b_name]) > 1:
            if bc[b_name]==0:
                gdf_basin = gdf_basins.loc[gdf_basins['NAME']==b_name]
                basins_grouped.append(gdf_basin.dissolve())
            bc[b_name]+=1
        else:
            basins_grouped.append(gdf_basins.loc[gdf_basins['NAME']==b_name])
    gdf_basins = pd.concat(basins_grouped).reset_index().loc[:,['NAME','geometry']]
    basins_3D = rm.mask_3D_geopandas(gdf_basins,lon,lat) # 3d mask for basins

    # dataset for exposure trends
    ds_e = xr.Dataset(
        data_vars={
            'exposure_trend_ar6': (
                ['run','GMT','region','year'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(GMT_labels),len(ar6_regs_3D.region.data),len(np.arange(year_start,year_ref+1,20))),
                    fill_value=np.nan,
                ),
            ),
            'mean_exposure_trend_ar6': (
                ['GMT','region','year'],
                np.full(
                    (len(GMT_labels),len(ar6_regs_3D.region.data),len(np.arange(year_start,year_ref+1,20))),
                    fill_value=np.nan,
                ),
            ),                                
            'exposure_trend_country': (
                ['run','GMT','country','year'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(GMT_labels),len(countries_3D.region.data),len(np.arange(year_start,year_ref+1,20))),
                    fill_value=np.nan,
                ),
            ),
            'mean_exposure_trend_country': (
                ['GMT','country','year'],
                np.full(
                    (len(GMT_labels),len(countries_3D.region.data),len(np.arange(year_start,year_ref+1,20))),
                    fill_value=np.nan,
                ),
            ),         
            'exposure_trend_basin': (
                ['run','GMT','basin','year'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(GMT_labels),len(basins_3D.region.data),len(np.arange(year_start,year_ref+1,20))),
                    fill_value=np.nan,
                ),
            ),
            'mean_exposure_trend_basin': (
                ['GMT','basin','year'],
                np.full(
                    (len(GMT_labels),len(basins_3D.region.data),len(np.arange(year_start,year_ref+1,20))),
                    fill_value=np.nan,
                ),
            ),                                                             
        },
        coords={
            'region': ('region', ar6_regs_3D.region.data),
            'country': ('country', countries_3D.region.data),
            'basin': ('basin', basins_3D.region.data),
            'run': ('run', list(d_isimip_meta.keys())),
            'GMT': ('GMT', GMT_labels),
            'year': ('year', np.arange(year_start,year_ref+1,20))
        }
    )
    
    # loop over simulations
    for i in list(d_isimip_meta.keys()): 

        print('simulation {} of {}'.format(i,len(d_isimip_meta)))

        # load AFA data of that run
        with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
            da_AFA = pk.load(f)  
        
        # per GMT step, if max threshold criteria met, run gmt mapping and compute trends
        for step in GMT_labels:
                
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
            
                # reindexing original exposure array based on GMT-mapping indices
                da_AFA_step = da_AFA.reindex(
                    {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                ).assign_coords({'time':year_range}) 
                
                # get sums of exposed area per ar6, country & basin, convert m^2 to km^2
                da_AFA_ar6_weighted_sum = da_AFA_step.weighted(ar6_regs_3D*grid_area/10**6).sum(dim=('lat','lon'))
                da_AFA_country_weighted_sum = da_AFA_step.weighted(countries_3D*grid_area/10**6).sum(dim=('lat','lon'))
                da_AFA_basin_weighted_sum = da_AFA_step.weighted(basins_3D*grid_area/10**6).sum(dim=('lat','lon'))
        
                # regressions on separate 80 year periods of area sums
                for y in np.arange(year_start,year_ref+1,20):
                    
                    # ar6 regions
                    ds_e['exposure_trend_ar6'].loc[{
                        'run':i,
                        'GMT':step,
                        'region':ar6_regs_3D.region.data,
                        'year':y,
                    }] = vectorize_lreg(da_AFA_ar6_weighted_sum.loc[{'time':np.arange(y,y+81)}])
                    
                    # countries
                    ds_e['exposure_trend_country'].loc[{
                        'run':i,
                        'GMT':step,
                        'country':countries_3D.region.data,
                        'year':y,
                    }] = vectorize_lreg(da_AFA_country_weighted_sum.loc[{'time':np.arange(y,y+81)}])
                    
                    # basins
                    ds_e['exposure_trend_basin'].loc[{
                        'run':i,
                        'GMT':step,
                        'basin':basins_3D.region.data,
                        'year':y,
                    }] = vectorize_lreg(da_AFA_basin_weighted_sum.loc[{'time':np.arange(y,y+81)}])
    
    # take means of trends in exposed area
    ds_e['mean_exposure_trend_ar6'] = ds_e['exposure_trend_ar6'].mean(dim='run')
    ds_e['mean_exposure_trend_country'] = ds_e['exposure_trend_country'].mean(dim='run')
    ds_e['mean_exposure_trend_basin'] = ds_e['exposure_trend_basin'].mean(dim='run')

    # dump pickle of exposure trends
    with open('./data/{}/{}/exposure_trends_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_e,f)

    return ds_e
        
#%% ----------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span
def calc_lifetime_exposure(
    d_isimip_meta, 
    df_countries, 
    countries_regions, 
    countries_mask, 
    da_population, 
    df_life_expectancy_5,
    flags,
):

    # nan dataset of lifetime exposure
    ds_le = xr.Dataset(
        data_vars={
            'lifetime_exposure': (
                ['run','GMT','country','birth_year'],
                np.full(
                    (len(list(d_isimip_meta.keys())),len(GMT_labels),len(df_countries['name'].values),len(birth_years)),
                    fill_value=np.nan,
                ),
            )
        },
        coords={
            'country': ('country', df_countries['name'].values),
            'birth_year': ('birth_year', birth_years),
            'run': ('run', np.arange(1,len(list(d_isimip_meta.keys()))+1)),
            'GMT': ('GMT', GMT_labels)
        }
    )
    
    # loop over simulations
    for i in list(d_isimip_meta.keys()): 

        print('simulation {} of {}'.format(i,len(d_isimip_meta)))

        # load AFA data of that run
        with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
            da_AFA = pk.load(f)

        # --------------------------------------------------------------------
        # per country 

        # initialise dicts
        d_exposure_peryear_percountry = {}

        # get spatial average
        for j, country in enumerate(df_countries['name']):

            print('processing country '+str(j+1)+' of '+str(len(df_countries)), end='\r')
            
            # calculate mean per country weighted by population
            ind_country = countries_regions.map_keys(country)

            # historical + RCP simulations
            d_exposure_peryear_percountry[country] = calc_weighted_fldmean( 
                da_AFA,
                da_population, 
                countries_mask, 
                ind_country, 
                flag_region=False,
            )
                        
        # --------------------------------------------------------------------
        # convert dict to dataframe for vectorizing and integrate exposures then map to GMTs        
        frame = {k:v.values for k,v in d_exposure_peryear_percountry.items()}
        df_exposure = pd.DataFrame(frame,index=year_range)           
        
        # if max threshold criteria met, run gmt mapping
        for step in GMT_labels:
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
            
                # reindexing original exposure array based on GMT-mapping indices
                d_exposure_perrun_step = df_exposure.apply(
                    lambda col: calc_life_exposure(
                        df_exposure.reindex(df_exposure.index[d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]).set_index(df_exposure.index),
                        df_life_expectancy_5,
                        col.name,
                    ),
                    axis=0,
                )            
        
                # convert dataframe to data array of lifetime exposure (le) per country and birth year
                ds_le['lifetime_exposure'].loc[{
                    'run':i,
                    'GMT':step,
                }] = d_exposure_perrun_step.values.transpose()             

    # dump pickle of lifetime exposure
    with open('./data/{}/{}/lifetime_exposure_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(ds_le,f)

    return ds_le
        
#%% ----------------------------------------------------------------
# convert Area Fraction Affected (AFA) to 
# per-cohort number of extremes affecting one individual across life span
def calc_cohort_lifetime_exposure(
    d_isimip_meta,
    df_countries,
    countries_regions,
    countries_mask,
    da_population,
    da_cohort_size,
    flags,
):

    # loop over simulations
    for i in list(d_isimip_meta.keys()): 

        print('simulation {} of {}'.format(i,len(d_isimip_meta)))

        # load AFA data of that run
        with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
            da_AFA = pk.load(f)

        # --------------------------------------------------------------------
        # per country 

        # initialise dicts
        d_exposure_peryear_percountry = {}

        # get spatial average
        for j, country in enumerate(df_countries['name']):

            print('processing country '+str(j+1)+' of '+str(len(df_countries)), end='\r')
            
            # calculate mean per country weighted by population
            ind_country = countries_regions.map_keys(country)

            # historical + RCP simulations
            d_exposure_peryear_percountry[country] = calc_weighted_fldmean( 
                da_AFA,
                da_population, 
                countries_mask, 
                ind_country, 
                flag_region= False,
            )
            
        # convert dictionary to data array
        da_exposure_peryear_percountry = xr.DataArray(
            list(d_exposure_peryear_percountry.values()),
            coords={
                'country': ('country', list(d_exposure_peryear_percountry.keys())),
                'time': ('time', da_AFA.time.values),
            },
            dims=[
                'country',
                'time',
            ],
        )
            
        # GMT mapping for cohort exposure for stylized trajectories
        da_exposure_cohort_strj = xr.DataArray(
            coords={
                'country': ('country', list(d_exposure_peryear_percountry.keys())),
                'time': ('time', da_AFA.time.values),
                'ages': ('ages', da_cohort_size.ages.values),
                'GMT': ('GMT', GMT_labels),
            },
            dims=[
                'country',
                'time',
                'ages',
                'GMT',
            ],
        ) 
        
        # if max threshold criteria met, run gmt mapping              
        for step in GMT_labels:
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                
                # assign data to data array based on step in stylized trajectories
                da_exposure_cohort_strj.loc[
                    {'country':da_exposure_cohort_strj.country,
                        'time':da_exposure_cohort_strj.time,
                        'GMT':step}
                    ] = da_exposure_peryear_percountry.reindex(
                        {'time':da_exposure_peryear_percountry['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                    ).assign_coords({'time':year_range}) * da_cohort_size
                    
        with open('./data/{}/{}/exposure_cohort_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i), 'wb') as f:
            pk.dump(da_exposure_cohort_strj,f)
            
        # GMT mapping for stylized trajectories in dimension expansion of da_exposure_peryear_percountry
        da_exposure_peryear_perage_percountry_strj = xr.DataArray(
            coords={
                'country': ('country', list(d_exposure_peryear_percountry.keys())),
                'time': ('time', da_AFA.time.values),
                'ages': ('ages', da_cohort_size.ages.values),
                'GMT': ('GMT', GMT_labels),
            },
            dims=[
                'country',
                'time',
                'ages',
                'GMT',
            ],
        )
        
        for step in GMT_labels:
            
            if d_isimip_meta[i]['GMT_strj_valid'][step]:
                
                # assign data to data array based on step in stylized trajectories
                da_exposure_peryear_perage_percountry_strj.loc[
                    {'country':da_exposure_peryear_perage_percountry_strj.country,
                        'time':da_exposure_peryear_perage_percountry_strj.time,
                        'GMT':step}
                    ] = da_exposure_peryear_percountry.reindex(
                        {'time':da_exposure_peryear_percountry['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                    ).assign_coords({'time':year_range}) * xr.full_like(da_cohort_size,1)
        
        with open('./data/{}/{}/exposure_peryear_perage_percountry_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],i), 'wb') as f:
            pk.dump(da_exposure_peryear_perage_percountry_strj,f)                     
        
        
#%% ----------------------------------------------------------------
# convert PIC Area Fraction Affected (AFA) to 
# per-country number of extremes affecting one individual across life span
def calc_lifetime_exposure_pic(
    d_pic_meta, 
    df_countries, 
    countries_regions, 
    countries_mask, 
    da_population, 
    df_life_expectancy_5, 
    flags,
):

    d_exposure_perrun_pic = {}                 
    
    # loop over simulations
    for n,i in enumerate(list(d_pic_meta.keys())):

        print('simulation '+str(n+1)+ ' of '+str(len(d_pic_meta)))

        # load AFA data of that run
        with open('./data/{}/{}/isimip_AFA_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
            da_AFA_pic = pk.load(f)
        
        # get 1960 life expectancy
        life_expectancy_1960 = xr.DataArray(
            df_life_expectancy_5.loc[1960].values,
            coords={
                'country': ('country', df_life_expectancy_5.columns)
            }
        )            
        
        # --------------------------------------------------------------------
        # per country 
        # start_time = time.time()
        d_exposure_peryear_percountry_pic = {}
        
        # get spatial average
        for j, country in enumerate(df_countries['name']): # with other stuff running, this loop took 91 minutes
            # therefore consider first doing the weighted mean and then boot strapping? does that make sense?

            print('processing country '+str(j+1)+' of '+str(len(df_countries)), end='\r')
            # calculate mean per country weighted by population
            ind_country = countries_regions.map_keys(country)

            # corresponding picontrol - assume constant 1960 population density (this line takes about 16h by itself)
            d_exposure_peryear_percountry_pic[country] = calc_weighted_fldmean(
                da_AFA_pic, 
                da_population[0,:,:], # earliest year used for weights
                countries_mask, 
                ind_country, 
                flag_region= False,
            )
            
        da_exposure_pic = xr.DataArray(
            list(d_exposure_peryear_percountry_pic.values()),
            coords={
                'country': ('country', list(d_exposure_peryear_percountry_pic.keys())),
                'time': ('time', da_AFA_pic.time.values),
            },
            dims=[
                'country',
                'time',
            ],
        )

        # bootstrap native pic exposed area data ;pic_life_extent, nboots, resample_dim
        da_exposure_pic = xr.concat([resample(da_exposure_pic,resample_dim,pic_life_extent) for i in range(nboots)],dim='lifetimes')
        
        # --------------------------------------------------------------------
        # substitute calc_life_exposure because we are only doing the 1960 cohort
        d_exposure_perrun_pic[i] = da_exposure_pic.where(da_exposure_pic.time < 1960 + np.floor(life_expectancy_1960)).sum(dim='time') + \
            da_exposure_pic.where(da_exposure_pic.time == 1960 + np.floor(life_expectancy_1960)).sum(dim='time') * \
                (life_expectancy_1960 - np.floor(life_expectancy_1960))

    # save pickles
    with open('./data/{}/{}/exposure_pic_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
        pk.dump(d_exposure_perrun_pic,f)

    return d_exposure_perrun_pic

