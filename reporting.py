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
import openpyxl
import pickle as pk
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import regionmask as rm
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import interpolate
import cartopy.crs as ccrs
from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

#%% ----------------------------------------------------------------
# sample analytics for paper
# ------------------------------------------------------------------

#%% ----------------------------------------------------------------
# multi-hazard emergence estimates
# ------------------------------------------------------------------

def multi_hazard_emergence(
    grid_area,
    da_emergence_mean,
    da_gs_popdenom,
):
    with open('./data/pickles_v2/gridscale_cohort_global.pkl', 'rb') as file:
        da_gridscale_cohortsize = pk.load(file)   

    extremes_labels = {
        'burntarea': 'Wildfires',
        'cropfailedarea': 'Crop failures',
        'driedarea': 'Droughts',
        'floodedarea': 'Floods',
        'heatwavedarea': 'Heatwaves',
        'tropicalcyclonedarea': 'Tropical cyclones',
    }  

    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]

    density=6
    sim_frac=0.25
    gmt_indices_152535 = [24,15,6]
    gmt = 17 # gmt index to compare multihazard pf
    multiextrn = 3 #number of extremes for multihazard pf comparison
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)

    # pf for 1960 birth cohort in multi hazard case
    template_1960 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=17,birth_year=1960),
        False
    )

    for extr in extremes:

        p1960 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':1960,
        }]
        template_1960 = template_1960+p1960.where(p1960>sim_frac).notnull()

    p_u1960 = template_1960.where(mask.notnull())
    pf_3extr_1960 = da_gridscale_cohortsize.loc[{
        'birth_year':1960,
    }].where(p_u1960>=multiextrn).sum(dim=('lat','lon')) / da_gs_popdenom.loc[{'birth_year':1960}].sum(dim='country') * 100
    print('1960 {}-hazard pf is {} for GMT {}'.format(multiextrn,pf_3extr_1960['cohort_size'].item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))

    # pf for 2020 birth cohort in multi hazard case
    template_2020 = xr.full_like(
        da_emergence_mean.sel(hazard='heatwavedarea',GMT=17,birth_year=2020),
        False
    )

    for extr in extremes:

        p2020 = da_emergence_mean.loc[{
            'hazard':extr,
            'GMT':gmt,
            'birth_year':2020,
        }]
        template_2020 = template_2020+p2020.where(p2020>sim_frac).notnull()

    p_u2020 = template_2020.where(mask.notnull())
    pf_3extr_2020 = da_gridscale_cohortsize.loc[{
        'birth_year':2020,
    }].where(p_u2020>=multiextrn).sum(dim=('lat','lon')) / da_gs_popdenom.loc[{'birth_year':2020}].sum(dim='country') * 100
    print('2020 {}-hazard pf is {} for GMT {}'.format(multiextrn,pf_3extr_2020['cohort_size'].item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))


    # land area
    la_frac_eu_gteq3_2020 = xr.where(p_u2020>=multiextrn,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100
    la_frac_eu_gteq3_1960 = xr.where(p_u1960>=multiextrn,grid_area,0).sum(dim=('lat','lon')) / grid_area.where(mask==0).sum(dim=('lat','lon')) * 100


    print('1960 percentage of land area \n with emergence of {} extremes \n is {} in a {} GMT pathway'.format(multiextrn,la_frac_eu_gteq3_1960.item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))  
    print('2020 percentage of land area \n with emergence of {} extremes \n is {} in a {} GMT pathway'.format(multiextrn,la_frac_eu_gteq3_2020.item(),str(np.round(df_GMT_strj.loc[2100,gmt],1))))    

#%% ----------------------------------------------------------------
# grid scale cohort sizes per birth year (copied from jupyter and only useable there)
# ------------------------------------------------------------------

def gridscale_cohort_sizes(
    flags,
    da_population,
    gridscale_countries,   
):
    # dataset of by_py0
    ds_gridscale_cohortsize = xr.Dataset(
        data_vars={
            'cohort_size': (
                ['birth_year','lat','lon'],
                np.full(
                    (len(birth_years),len(da_population.lat.data),len(da_population.lon.data)),
                    fill_value=np.nan,
                ),
            ),
        },
        coords={
            'lat': ('lat', da_population.lat.data),
            'lon': ('lon', da_population.lon.data),
            'birth_year': ('birth_year', birth_years),
        }
    )

    # loop through countries and assign birth cohort size to dataset
    for cntry in gridscale_countries:
        print(cntry)
        # load demography pickle
        with open('./data/{}/gridscale_dmg_{}.pkl'.format(flags['version'],cntry), 'rb') as f:
            ds_dmg = pk.load(f)   
        # get population used in analysis
        da_cohort_cntry = ds_dmg['by_population_y0']
        # assign to bigger dataset
        ds_gridscale_cohortsize['cohort_size'].loc[{
            'birth_year':birth_years,
            'lat':da_cohort_cntry.lat.data,
            'lon':da_cohort_cntry.lon.data,
        }] = xr.where(
            da_cohort_cntry.notnull(),
            da_cohort_cntry,
            ds_gridscale_cohortsize['cohort_size'].loc[{'birth_year':birth_years,'lat':da_cohort_cntry.lat.data,'lon':da_cohort_cntry.lon.data}],
        )
    print('countries merged')
    with open('./data/{}/gridscale_cohort_global.pkl'.format(flags['version']), 'wb') as f:
        pk.dump(ds_gridscale_cohortsize,f)   
        
#%% ----------------------------------------------------------------
# grid exposure locations for all sims 
# ------------------------------------------------------------------        
        
def exposure_locs(
    flags,
    grid_area,
):        
    extremes = [
        flags['extr']
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        # 'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]

    lat = grid_area.lat.values
    lon = grid_area.lon.values
    mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)

    for extr in extremes:
        
        # with open('./data/{}/{}/isimip_metadata_{}_ar6_rm.pkl'.format(flags['version'],extr,extr), 'rb') as file:
        #     d_isimip_meta = pk.load(file)     
        # get metadata for extreme
        with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)        
            
        n = 0
        for i in list(d_isimip_meta.keys()): 

            print('simulation {} of {}'.format(i,len(d_isimip_meta)))

            # load AFA data of that run
            with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],extr,extr,str(i)), 'rb') as f:
                da_AFA = pk.load(f)           
            
            if n == 0:    
                da_sum = da_AFA.sum(dim='time').where(mask.notnull())
            else:
                da_sum = da_sum + da_AFA.sum(dim='time').where(mask.notnull())
            
            n+=1
            
        da_exposure_occurence = xr.where(da_sum>0,1,0)
        
        with open('./data/{}/{}/exposure_occurrence_{}.pkl'.format(flags['version'],extr,extr), 'wb') as file:
            pk.dump(da_exposure_occurence,file)      
            
            
#%% ----------------------------------------------------------------
# emergence locations in specific runs for 1.5, 2.5, 2.7 and 3.5
# used for geographically constrained PF to recompute with different 
# denominator (used for numerator with grid scale pop to compare against
# pop of regions pinged by exposure in da_exposure_occurrence)
# ------------------------------------------------------------------                    

def emergence_locs_perrun(
    flags,
    grid_area,
    gridscale_countries,
    countries_mask,
    countries_regions,
):

    gmt_indices_sample = [0,10,20]
    lat = grid_area.lat.values
    lon = grid_area.lon.values
    da_mask = rm.defined_regions.natural_earth_v5_0_0.land_110.mask(lon,lat)
    
    extremes = [
        flags['extr']
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        # 'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ] 
            
    # loop through extremes
    for extr in extremes:
        
        start_time = time.time()
        
        # get metadata for extreme
        with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)
            
        sims_per_step = {}
        for step in gmt_indices_sample:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)                           
        
        for step in gmt_indices_sample:
            
            ds_global_emergence = xr.Dataset(
                data_vars={
                    'emergence': (
                        ['run','birth_year','lat','lon'],
                        np.full(
                            (len(sims_per_step[step]),len(birth_years),len(da_mask.lat.data),len(da_mask.lon.data)),
                            fill_value=np.nan,
                        ),
                    ),                              
                },
                coords={
                    'lat': ('lat', da_mask.lat.data),
                    'lon': ('lon', da_mask.lon.data),
                    'birth_year': ('birth_year', birth_years),
                    'run': ('run', sims_per_step[step]),
                    'GMT': ('GMT', GMT_labels),
                }
            )        
            
            # loop through countries
            for cntry in gridscale_countries:
                
                da_cntry = xr.DataArray(
                    np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                    dims=countries_mask.dims,
                    coords=countries_mask.coords,
                )
                da_cntry = da_cntry.where(da_cntry,drop=True)                  
                
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
                    
                        with open('./data/{}/{}/{}/gridscale_emergence_mask_{}_{}_{}_{}_99.99.pkl'.format(flags['version'],extr,cntry,extr,cntry,i,step), 'rb') as f:
                            da_birthyear_emergence_mask = pk.load(f)
                            
                        ds_cntry_emergence['emergence'].loc[{
                            'run':i,
                            'birth_year':birth_years,
                            'lat':da_cntry.lat.data,
                            'lon':da_cntry.lon.data,
                        }] = da_birthyear_emergence_mask      
                        
                ds_cntry_emergence['emergence'] = ds_cntry_emergence['emergence'].where(da_cntry == 1)
                
                ds_global_emergence['emergence'].loc[{
                    'run':sims_per_step[step],
                    'birth_year':birth_years,
                    'lat':da_cntry.lat.data,
                    'lon':da_cntry.lon.data,
                }] = xr.where(
                    ds_cntry_emergence['emergence'].notnull(),
                    ds_cntry_emergence['emergence'],
                    ds_global_emergence['emergence'].loc[{
                        'run':sims_per_step[step],'birth_year':birth_years,'lat':da_cntry.lat.data,'lon':da_cntry.lon.data
                    }],
                )             
                
            with open('./data/{}/{}/emergence_locs_perrun_{}_{}.pkl'.format(flags['version'],extr,extr,step), 'wb') as f:
                pk.dump(ds_global_emergence['emergence'],f)        
                
        print("--- {} minutes for {} emergence loc ---".format(
            np.floor((time.time() - start_time) / 60),
            extr
            )
            )    
        
#%% ----------------------------------------------------------------
# population fraction estimates per run and for selected GMTs 
# when constraining denominator by geography ie exposed locations
# in our dataset
# ------------------------------------------------------------------            

def pf_geoconstrained(
    flags,
    countries_mask,
):

    gmt_indices_sample = [0,10,20]
    unprec_level="unprec_99.99"
    extremes = [
        flags['extr']
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        # 'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]  

    with open('./data/{}/gridscale_cohort_global.pkl'.format(flags['version']), 'rb') as file:
        ds_gridscale_cohortsize = pk.load(file)   
        
    da_gridscale_cohortsize = ds_gridscale_cohortsize['cohort_size']

    # loop through extremes
    for extr in extremes:

        start_time = time.time()

        # first get all regions that have exposure to extr in ensemble
        with open('./data/{}/{}/exposure_occurrence_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            da_exposure_occurrence = pk.load(file)          

        # get metadata for extreme
        with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)
            
        sims_per_step = {}
        for step in gmt_indices_sample:
            sims_per_step[step] = []
            print('step {}'.format(step))
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)  
                    
        ds_pf_geoconstrained = xr.Dataset(
            data_vars={
                'p_perrun': (
                    ['GMT','run','birth_year'],
                    np.full(
                        (len(gmt_indices_sample),len(sims_per_step[gmt_indices_sample[0]]),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),                
                'pf_perrun': (
                    ['GMT','run','birth_year'],
                    np.full(
                        (len(gmt_indices_sample),len(sims_per_step[gmt_indices_sample[0]]),len(birth_years)),
                        fill_value=np.nan,
                    ),
                ),                                        
            },
            coords={
                'birth_year': ('birth_year', birth_years),
                'run': ('run', sims_per_step[gmt_indices_sample[0]]),
                'GMT': ('GMT', gmt_indices_sample),
            }
        )       
        # numerator to exposure constrained PF
        for step in gmt_indices_sample:
            
            with open('./data/{}/{}/emergence_locs_perrun_{}_{}.pkl'.format(flags['version'],extr,extr,step), 'rb') as f:
                da_global_emergence = pk.load(f)
                
            da_global_emergence = xr.where(da_global_emergence==1,1,0)    

            for r in da_global_emergence.run.data:
                
                da_global_emergence.loc[{'run':r}] = da_global_emergence.loc[{'run':r}] * da_gridscale_cohortsize

            da_unprec_p = da_global_emergence.sum(dim=('lat','lon'))

            da_total_p = da_exposure_occurrence.where(countries_mask.notnull()) * da_gridscale_cohortsize
            da_total_p = da_total_p.sum(dim=('lat','lon'))

            da_pf = da_unprec_p / da_total_p

            ds_pf_geoconstrained['p_perrun'].loc[{
                'GMT':step,
                'run':da_pf.run.data,
                'birth_year':birth_years,
            }] = da_unprec_p
            
            ds_pf_geoconstrained['pf_perrun'].loc[{
                'GMT':step,
                'run':da_pf.run.data,
                'birth_year':birth_years,
            }] = da_pf        
        
        with open('./data/{}/{}/pf_geoconstrained_{}.pkl'.format(flags['version'],extr,extr), 'wb') as f:
            pk.dump(ds_pf_geoconstrained,f)  
        
        print("--- {} minutes for {} pf in under geo constraints ---".format(
            np.floor((time.time() - start_time) / 60),
            extr
            )
            )                

#%% ----------------------------------------------------------------
# read in geoconstrained pf and print for 1960 and 2020 across GMTs
# ------------------------------------------------------------------                           
                   
def print_pf_geoconstrained(
    flags,    
    da_gs_popdenom,
):

    gmt_indices_sample = [0,10,20]
    unprec_level="unprec_99.99"
    extremes = [
        flags['extr']
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        # 'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]

    for extr in extremes:
        
        with open('./data/{}/{}/pf_geoconstrained_{}.pkl'.format(flags['version'],extr,extr), 'rb') as f:
            ds_pf_geoconstrained = pk.load(f)      
            
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)      
            
        # get metadata for extreme
        with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as f:
            d_isimip_meta = pk.load(f)    
            
        # maybe not necessary since means are ignoring nans for runs not included in some steps
        sims_per_step = {}
        for step in gmt_indices_sample:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)             
        
        for step in gmt_indices_sample:
            
            pf_geo = ds_pf_geoconstrained['pf_perrun'].loc[{'GMT':step,'run':sims_per_step[step]}].mean(dim='run') * 100
            pf = ds_pf_gs[unprec_level].loc[{'GMT':step,'run':sims_per_step[step]}].fillna(0).sum(dim='country').mean(dim='run') / da_gs_popdenom.sum(dim='country') * 100
            
            print('{} under GMT step {} has geoconstrained pf of {} for 1960 and {} for 2020'.format(extr,step,pf_geo.loc[{'birth_year':1960}].item(),pf_geo.loc[{'birth_year':2020}].item()))
            print('{} under GMT step {} has regular pf of {} for 1960 and {} for 2020'.format(extr,step,pf.loc[{'birth_year':1960}].item(),pf.loc[{'birth_year':2020}].item()))
            
#%% ----------------------------------------------------------------
# checking for signifiance of change in means between 1960 and 2020 pf per event and for a GMT level
# low sensitivity to ttest_ind() or ttest_rel() choice
# ------------------------------------------------------------------        
    
def paired_ttest(
    flags,
    da_gs_popdenom,
):

    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]

    # GMT step representing CAT policy pledges for 2.7 degree warming
    gmtlevel=17
    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)               
        
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)         
        
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        
        da_plt = ds_pf_gs_extr['unprec'].loc[{
            'birth_year':birth_years,
            'GMT':gmtlevel,
            'run':sims_per_step[gmtlevel]
        }].sum(dim='country') # summing converts nans from invalid GMT/run combos to 0, use where below to remove these
        da_plt_gmt = da_plt.where(da_plt!=0) / da_gs_popdenom.sum(dim='country') * 100 
        
        list_extrs_pf.append(da_plt_gmt)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})

    for extr in extremes:
        
        # coefficient of  of variation
        mean = ds_pf_gs_extrs.sel(hazard=extr).mean(dim=('run','birth_year')).item()
        std = ds_pf_gs_extrs.sel(hazard=extr).std(dim=('run','birth_year')).item()
        cv = std / mean
        print('CV is {}'.format(cv))
        mean_1960 = ds_pf_gs_extrs.sel(hazard=extr,birth_year=1960).mean(dim=('run')).item()
        mean_2020 = ds_pf_gs_extrs.sel(hazard=extr,birth_year=2020).mean(dim=('run')).item()
        delta_mean = mean_2020 - mean_1960
        delta_ratio = delta_mean / mean
        print('delta mean ratio is {}'.format(delta_ratio))
        
        # 2 sample t test
        extr_1960=ds_pf_gs_extrs.sel(hazard=extr,birth_year=1960).values
        extr_2020=ds_pf_gs_extrs.sel(hazard=extr,birth_year=2020).values
        result = sts.ttest_rel(
            extr_1960, 
            extr_2020,
            nan_policy='omit',
        )
        print('{} p value for difference of means: {}'.format(extr,result.pvalue))
        print('')
            
#%% ----------------------------------------------------------------
# print latex table of ensemble members per hazard and gmt pathway
# ------------------------------------------------------------------      

def print_latex_table_ensemble_sizes(
    flags,
    df_GMT_strj,
):
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

    gmts = np.arange(0,21).astype('int')
    gmts2100 = np.round(df_GMT_strj.loc[2100,gmts].values,1)   
    gmt_dict = dict(zip(gmts,gmts2100))

    sims_per_step = {}
    for extr in extremes:
        
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)        

        sims_per_step[extr] = {}
        for step in gmts:
            sims_per_step[extr][step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[extr][step].append(i)      
                    
    headers = list(extremes_labels.values())
    data = {}
    for step in gmts:
        data[str(gmt_dict[step])] = [len(sims_per_step[extr][step]) for extr in extremes]

    textabular = f"l|{'r'*len(headers)}"
    texheader = " & " + " & ".join(headers) + "\\\\"
    texdata = "\\hline\n"
    for label in sorted(data):
        if label == "z":
            texdata += "\\hline\n"
        texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

    print("\\begin{tabular}{"+textabular+"}")
    print(texheader)
    print(texdata,end="")
    print("\\end{tabular}")                

#%% ----------------------------------------------------------------
# millions excess children between 1.5 and 2.7 deg warming by 2100
# living unprecedented exposure to events
# ------------------------------------------------------------------  

def print_millions_excess(
    flags,
    df_GMT_strj,
):

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

    pic_qntl_str=str(pic_qntl*100)
    gmts = GMT_labels
    gmts2100 = np.round(df_GMT_strj.loc[2100,gmts].values,1)   
    gmt_dict = dict(zip(gmts,gmts2100))
    sumlist=[]
    for extr in extremes:
        
        print(extr)
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)    
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)            

        sims_per_step = {}
        for step in gmts:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)  
        # millions children unprecedented in 1.5 pathway
        step=0
        unprec_15 = ds_pf_gs['unprec_{}'.format(pic_qntl_str)].sum(dim='country').loc[{'GMT':step,'run':sims_per_step[step],'birth_year':np.arange(2003,2021)}].sum(dim='birth_year').mean(dim='run') / 10**6
        print('in 1.5 degree pathway, {} chidren live unprecedented exposure to {}'.format(np.around(unprec_15.item()),extr))
        
        # millions children unprecedented in 2.7 pathway
        step=12
        unprec_27 = ds_pf_gs['unprec_{}'.format(pic_qntl_str)].sum(dim='country').loc[{'GMT':step,'run':sims_per_step[step],'birth_year':np.arange(2003,2021)}].sum(dim='birth_year').mean(dim='run') / 10**6
        print('in 2.7 degree pathway, {} children live unprecedented exposure to {}'.format(np.around(unprec_27.item()),extr))    
            
        # difference between 1.5 and 2.7 deg pathways
        print('{} more million children will live through unprecedented exposure to {}'.format(np.around((unprec_27.item()-unprec_15.item())),extr))
        print('')
        
        sumlist.append(np.around((unprec_27.item()-unprec_15.item())))
        
    print(np.sum(sumlist))
    
#%% ----------------------------------------------------------------
# ratio of pfs reporting
# ------------------------------------------------------------------  
def print_pf_ratios_and_abstract_numbers(
    df_GMT_strj,
    da_gs_popdenom,
):

    letters = ['a', 'b', 'c',\
                'd', 'e', 'f',\
                'g', 'h', 'i',\
                'j', 'k', 'l']
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': '$\mathregular{PF_{Wildfires}}$',
        'cropfailedarea': '$\mathregular{PF_{Crop failures}}$',
        'driedarea': '$\mathregular{PF_{Droughts}}$',
        'floodedarea': '$\mathregular{PF_{Floods}}$',
        'heatwavedarea': '$\mathregular{PF_{Heatwaves}}$',
        'tropicalcyclonedarea': '$\mathregular{PF_{Tropical cyclones}}$',
    }        

    # gmt choice
    GMT_low = 0 # 1.5 degrees
    GMT_high = 20 # 3.5 degrees
    GMT_cp = 12 # 2.7 degrees, whereas 17 is 3.2 in new scheme
    unprec_level="unprec_99.99"

    # loop through extremes and concat pop and pop frac
    list_extrs_pf = []
    for extr in extremes:
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        p = ds_pf_gs_extr[unprec_level].loc[{
            'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
        }].sum(dim='country')       
        p = p.where(p!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country') *100
        list_extrs_pf.append(p)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})    

    for extr in extremes:
        
        # looking across birth years (1960 to 2020 for current policies)
        pf_1960 = ds_pf_gs_extrs.loc[{
            'birth_year':1960,
            'GMT':GMT_cp,
            'hazard':extr
        }].item()
        
        pf_2020 = ds_pf_gs_extrs.loc[{
            'birth_year':2020,
            'GMT':GMT_cp,
            'hazard':extr
        }].item()    
        
        pf_2020_1960_ratio = np.around(pf_2020 / pf_1960,1)
        
        print('ratio in pf for {} in 2.7 degree \n scenario between 2020 and 1960 is {} (meaning, pf_2020/pf_1960)'.format(extr,pf_2020_1960_ratio))
        
        # looking across GMTs for 2020
        pf15 = ds_pf_gs_extrs.loc[{
            'birth_year':2020,
            'GMT':GMT_low,
            'hazard':extr
        }].item()
        
        pf27 = ds_pf_gs_extrs.loc[{
            'birth_year':2020,
            'GMT':GMT_cp,
            'hazard':extr
        }].item()    
        
        pf35 = ds_pf_gs_extrs.loc[{
            'birth_year':2020,
            'GMT':GMT_high,
            'hazard':extr
        }].item()
        
        pf_27_15_ratio = np.around(pf27 / pf15,1)    
        
        print('change in pf for {} and 2020 cohort \n between 1.5 and 2.7 pathways is {}'.format(extr,pf_27_15_ratio))
        
        print('1.5 degree scenario pf for {} is {}'.format(extr,pf15))
        print('3.5 degree scenario pf for {} is {}'.format(extr,pf35))
        
        print('')  
        

#%% ----------------------------------------------------------------
# print number of unprecedented people
# ------------------------------------------------------------------          
        
def print_absolute_unprecedented(
    ds_pf_gs,     
):
    
    step=0
    by=2020
    unprec=ds_pf_gs['unprec_99.99'].sum(dim='country').loc[{'run':sims_per_step[step],'GMT':step, 'birth_year':by}].mean(dim='run')
    print('{} million'.format(unprec.item()/10**6))               
    

#%% ----------------------------------------------------------------
# get pickle of cities that are valid for f1 concept plot
# ------------------------------------------------------------------  

def find_valid_cities(
     df_countries,
     da_cohort_size,
     countries_mask,
     countries_regions,
     d_isimip_meta,
     flags,
):
    if not os.path.isfile('./data/pickles_v2/valid_cities.pkl'):
        # excel file of cities, their coords and population
        df_cities = pd.read_excel('./data/city_locations/worldcities.xlsx')
        df_cities = df_cities.drop(columns=['city_ascii','iso2','iso3','admin_name','capital','id']).nlargest(n=200,columns=['population'])
        concept_bys = np.arange(1960,2021,30)

        # loop through countries
        cntry_concat = []
        for cntry in list(df_countries.index):
            
            print(cntry)
            da_smple_cht = da_cohort_size.sel(country=cntry) # cohort absolute sizes in sample country
            da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages') # cohort relative sizes in sample country
            da_cntry = xr.DataArray(
                np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                dims=countries_mask.dims,
                coords=countries_mask.coords,
            )
            da_cntry = da_cntry.where(da_cntry,drop=True)
            lat_weights = np.cos(np.deg2rad(da_cntry.lat))
            lat_weights.name = "weights" 

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
            with open('./data/pickles_v2/gridscale_dmg_{}.pkl'.format(cntry), 'rb') as f:
                ds_dmg = pk.load(f)   

            # load PIC pickle
            with open('./data/pickles_v2/{}/gridscale_le_pic_{}_{}.pkl'.format(flags['extr'],flags['extr'],cntry), 'rb') as f:
                ds_pic = pk.load(f)                   

            # loop over simulations
            for i in list(d_isimip_meta.keys()): 

                # print('simulation {} of {}'.format(i,len(d_isimip_meta)))

                # load AFA data of that run
                with open('./data/pickles_v2/{}/isimip_AFA_{}_{}.pkl'.format(flags['extr'],flags['extr'],str(i)), 'rb') as f:
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

            # select country from excel database of city coords
            df_cntry = df_cities.loc[df_cities['country']==cntry].copy()
            df_cntry['valid'] = np.nan

            # loop through cities in country
            for city_i in list(df_cntry.index):   

                # get city info from 
                # city = df_cntry.loc[city_i,'city']
                city_lat = df_cntry.loc[city_i,'lat']
                city_lon = df_cntry.loc[city_i,'lng']

                # pic
                da_pic_city_9999 = ds_pic['99.99'].sel({'lat':city_lat,'lon':city_lon},method='nearest').item()            

                # mean for city            
                da_test_city = ds_spatial['cumulative_exposure'].sel({'lat':city_lat,'lon':city_lon},method='nearest').mean(dim='run')
                da_test_city = da_test_city.rolling(time=5,min_periods=5).mean()   

                # sequence booleans for showing that gmt+1 is greater than gmt (relevant for 2.5 and 3.5)
                sequence_bools = {}
                for i,gmt in enumerate(GMT_indices_plot):

                    sequence_bools[gmt] = []

                    for by in da_test_city.birth_year:

                        da_by = da_test_city.sel(birth_year=by).max(dim='time')
                        bool_entry = da_by.sel(GMT=gmt) > da_by.sel(GMT=GMT_indices_plot[i-1])
                        sequence_bools[gmt].append(bool_entry.item())

                # pre-industrial comparison to make sure 1960 lifetime exposure for 1.5, 2.5 and 3.5 is below pic 99.99
                pic_bools = []
                by=1960
                for i,gmt in enumerate(GMT_indices_plot):

                    da_by = da_test_city.sel(birth_year=by,GMT=gmt).max(dim='time')
                    bool_entry = da_by < da_pic_city_9999
                    pic_bools.append(bool_entry.item())        

                # check that sequence bools for 2.5 and 3.5 and pic bools are all True
                sequence_bools_highgmts = sequence_bools[15]+sequence_bools[24]
                all_bools = sequence_bools_highgmts + pic_bools
                if np.all(all_bools):
                    df_cntry.loc[city_i,'valid'] = True
                else:
                    df_cntry.loc[city_i,'valid'] = False

            # only keep cities that match criteria
            df_cntry = df_cntry.drop(df_cntry.index[df_cntry['valid']==False])
            cntry_concat.append(df_cntry)    

        df_valid_cities = pd.concat(cntry_concat)
        df_valid_cities = df_valid_cities.sort_values(by=['population'],ascending=False)
        print(df_valid_cities)    
        # pickle selection of cities
        with open('./data/pickles_v2/valid_cities.pkl', 'wb') as f:
            pk.dump(df_valid_cities,f)   
            
    else:
        
        with open('./data/pickles_v2/valid_cities.pkl', 'rb') as f:
            df_valid_cities = pk.load(f)        
            
            
#%% ----------------------------------------------------------------
# generating large latex tables on CF data per country, birth year and 1.5, 2.5 and 3.5 degree scenario
# ------------------------------------------------------------------

def print_latex_table_unprecedented(
    flags,
    da_gs_popdenom,
):

    # input
    unprec_level="unprec_99.99"      
    bys=np.arange(1960,2021,10)
    # bys=np.arange(1960,2021,1)
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'wildfires',
        'cropfailedarea': 'crop failures',
        'driedarea': 'droughts',
        'floodedarea': 'floods',
        'heatwavedarea': 'heatwaves',
        'tropicalcyclonedarea': 'tropical cyclones',
    }  

    # data
    for extr in extremes:
        
        # open dictionary of metadata for sim means and CF data per extreme
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)     
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)           

        sims_per_step = {}
        sims_per_step[extr] = {}
        for step in GMT_indices_plot:
            sims_per_step[extr][step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[extr][step].append(i)      
        
        da_p_gs_plot = ds_pf_gs[unprec_level].loc[{
            'GMT':GMT_indices_plot,
        }]
        df_list_gs = []
        for step in GMT_indices_plot:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[extr][step],'GMT':step}].mean(dim='run')
            da_cf_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom * 100
            df_cf_gs_plot_step = da_cf_gs_plot_step.to_dataframe(name='CF').reset_index()
            df_cf_gs_plot_step['P'] = da_p_gs_plot_step.to_dataframe(name='P').reset_index().loc[:,['P']] / 1000
            df_list_gs.append(df_cf_gs_plot_step)
        df_cf_gs_plot = pd.concat(df_list_gs)

        df_cf_gs_plot['CF'] = df_cf_gs_plot['CF'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P'] = df_cf_gs_plot['P'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P (CF)'] = df_cf_gs_plot.apply(lambda x: '{} ({})'.format(str(x.P),str(x.CF)), axis=1)

        # print latex per step
        for step in GMT_indices_plot:
            
            print('')
            print('Running latex print of CF for {} under {} pathway'.format(extremes_labels[extr],gmt_legend[step]))
            print('')
            
            df_latex = df_cf_gs_plot[df_cf_gs_plot['GMT']==step].copy()
            df_cntry_by = df_latex.loc[:,['country','birth_year','P (CF)']].set_index('country')

            for by in bys:
                df_cntry_by[by] = df_cntry_by[df_cntry_by['birth_year']==by].loc[:,['P (CF)']]
                
            df_cntry_by = df_cntry_by.drop(columns=['birth_year','P (CF)']).drop_duplicates() 

            # latex
            caption = '\\caption{{\\textbf{{Absolute population (in thousands) of cohorts living unprecedented exposure to {0} and CF\\textsubscript{{{0}}} (\\%) per country and birth year in a {1}\\degree C pathway}}}}\\\\'.format(extremes_labels[extr],gmt_legend[step])
            headers = list(df_cntry_by.columns.astype('str'))
            headers = ['Country'] + headers
            data = {}
            for row in list(df_cntry_by.index):
                if len(str(row).split()) > 1:
                    newrow = ' \\\ '.join(str(row).split())
                    newrow = '\makecell[l]{{{}}}'.format(newrow)    
                    data[str(newrow)] = list(df_cntry_by.loc[row,:].values)
                else:
                    data[str(row)] = list(df_cntry_by.loc[row,:].values)

            textabular = f" l |{' c '*(len(headers)-1)}"
            texheader = " & ".join(headers) + "\\\\"
            texdata = "\\hline\n"

            for label in data:
                if label == "z":
                    texdata += "\\hline\n"
                texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

            print('\\small')
            print('\\begin{longtable}{'+textabular+'}')
            print(caption)
            print(texheader)
            print(texdata,end='')
            print('\\end{longtable}')
            print('\\normalsize') 
            print('\\clearpage')             
#%% ----------------------------------------------------------------
# generating large latex tables on CF data per country, birth year and 1.5, 2.5 and 3.5 degree scenario
# ------------------------------------------------------------------

def print_latex_table_unprecedented_sideways(
    flags,
    da_gs_popdenom,
):

    # input
    # bys=np.arange(1960,2021,10)
    bys=np.arange(1960,2021,1)
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    extremes = [
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    extremes_labels = {
        'burntarea': 'wildfires',
        'cropfailedarea': 'crop failures',
        'driedarea': 'droughts',
        'floodedarea': 'floods',
        'heatwavedarea': 'heatwaves',
        'tropicalcyclonedarea': 'tropical cyclones',
    }  

    # data
    for extr in extremes:
        
        # open dictionary of metadata for sim means and CF data per extreme
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)     
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)           

        sims_per_step = {}
        sims_per_step[extr] = {}
        for step in GMT_indices_plot:
            sims_per_step[extr][step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[extr][step].append(i)      
        
        da_p_gs_plot = ds_pf_gs['unprec'].loc[{
            'GMT':GMT_indices_plot,
        }]
        df_list_gs = []
        for step in GMT_indices_plot:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[extr][step],'GMT':step}].mean(dim='run')
            da_cf_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom * 100
            df_cf_gs_plot_step = da_cf_gs_plot_step.to_dataframe(name='CF').reset_index()
            df_cf_gs_plot_step['P'] = da_p_gs_plot_step.to_dataframe(name='P').reset_index().loc[:,['P']] / 1000
            df_list_gs.append(df_cf_gs_plot_step)
        df_cf_gs_plot = pd.concat(df_list_gs)

        df_cf_gs_plot['CF'] = df_cf_gs_plot['CF'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P'] = df_cf_gs_plot['P'].fillna(0).round(decimals=0).astype('int') 
        df_cf_gs_plot['P (CF)'] = df_cf_gs_plot.apply(lambda x: '{} ({})'.format(str(x.P),str(x.CF)), axis=1)

        # print latex per step
        for step in GMT_indices_plot:
            
            print('')
            print('Running latex print of CF for {} under {} pathway'.format(extremes_labels[extr],gmt_legend[step]))
            print('')
            
            df_latex = df_cf_gs_plot[df_cf_gs_plot['GMT']==step].copy()
            df_cntry_by = df_latex.loc[:,['country','birth_year','P (CF)']].set_index('country')

            for by in bys:
                df_cntry_by[by] = df_cntry_by[df_cntry_by['birth_year']==by].loc[:,['P (CF)']]
                
            df_cntry_by = df_cntry_by.drop(columns=['birth_year','P (CF)']).drop_duplicates() 

            # latex
            caption = '\\caption{{\\textbf{{Absolute population (in thousands) of cohorts living unprecedented exposure to {0} and CF\\textsubscript{{{0}}} (\\%) per country and birth year in a {1}\\degree C pathway}}}}\\\\'.format(extremes_labels[extr],gmt_legend[step])
            headers = list(df_cntry_by.columns.astype('str'))
            headers = ['\'{}'.format(y[2:]) for y in headers]
            headers = ['Country'] + headers
            data = {}
            for row in list(df_cntry_by.index):
                if len(str(row).split()) > 1:
                    newrow = ' \\\ '.join(str(row).split())
                    newrow = '\makecell[l]{{{}}}'.format(newrow)    
                    data[str(newrow)] = list(df_cntry_by.loc[row,:].values)
                else:
                    data[str(row)] = list(df_cntry_by.loc[row,:].values)

            textabular = f" l |{' c '*(len(headers)-1)}"
            texheader = " & ".join(headers) + "\\\\"
            texdata = "\\hline\n"

            for label in data:
                if label == "z":
                    texdata += "\\hline\n"
                texdata += f"{label} & {' & '.join(map(str,data[label]))} \\\\\n"

            print('\\small')
            print('\\begin{longtable}{'+textabular+'}')
            print(caption)
            print(texheader)
            print(texdata,end='')
            print('\\end{longtable}')
            print('\\normalsize') 
            print('\\clearpage')             
#%% ----------------------------------------------------------------
# pyramid info
# ------------------------------------------------------------------

def print_pyramid_info(
    flags,
):

    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    # GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_integers = [0,10,12,20] # 1.5, 2.5, 2.7, and 3.5
    qntl_range = '20'
    vln_types=('grdi','gdp')
    
    # per vulnerability indicator
    for vln_type in vln_types:
        print('')
        print(vln_type)
        print('')
        with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
            d_pyramid_plot = pk.load(f)    
        for e in extremes:    
            for GMT in GMT_integers:
                poor_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}poorest'.format(qntl_range)]) # "_a" for panel "a"
                poor_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}poorest'.format(qntl_range)])
                rich_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
                rich_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}richest'.format(qntl_range)])
                pvalues_poor = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_poor'.format(qntl_range)])
                pvalues_rich = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_rich'.format(qntl_range)])
                
                print('')
                print('{} degree C pathway'.format(df_GMT_strj.loc[2100,GMT]))
                print('')
                
                print('ULE population for poorest is: \n {}'.format(poor_unprec))
                print('percentage of ULE for poorest is: \n {}'.format(poor_unprec / poor_pop * 100))
                print('unprecedented population for richest is: \n {}'.format(rich_unprec))
                print('percentage of ULE for richest is: \n {}'.format(rich_unprec / rich_pop * 100))
                print('p values significant: \n {}'.format(pvalues_poor < sl))
                
#%% ----------------------------------------------------------------
# f2 numbers
# ------------------------------------------------------------------                

def print_f2_info(
    ds_pf_gs,
    flags,
    df_GMT_strj,
    da_gs_popdenom,
    gdf_country_borders,
):
    plot_var='unprec_99.99'
    gmt_indices_152535 = [20,10,0]
    map_letters = {20:'g',10:'f',0:'e'}    
    gmt_legend={
        GMT_indices_plot[0]:'1.5',
        GMT_indices_plot[1]:'2.5',
        GMT_indices_plot[2]:'3.5',
    }
    
    # box plot stuff
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
    
    print('panel a: box plot time series numbers')
    for gmt in (0,10,20):
        print('GMT is {}'.format(df_GMT_strj.loc[2100,gmt]))
        df_reporting = df_pf_gs_plot[(df_pf_gs_plot['hazard']==extr)&(df_pf_gs_plot['GMT']==gmt)&(df_pf_gs_plot['birth_year']==2020)]
        print('median absolute number of ULE (in millions) is {}'.format(da_p_gs_plot.loc[{'run':sims_per_step[gmt],'GMT':gmt,'birth_year':2020}].fillna(0).sum(dim='country').median(dim='run').item()/10**6))
        print('median pf is {}'.format(df_reporting['pf'].median()))
        print('mean absolute number of ULE is {}'.format(da_p_gs_plot.loc[{'run':sims_per_step[gmt],'GMT':gmt,'birth_year':2020}].fillna(0).sum(dim='country').mean(dim='run').item()/10**6))
        print('mean pf is {}'.format(df_reporting['pf'].mean()))
        
        
    # map stuff
    by=2020
    da_p_gs_plot = ds_pf_gs_plot[plot_var].loc[{
        'GMT':gmt_indices_152535,
        'birth_year':by,
    }]
    df_list_gs = []
    for step in gmt_indices_152535:
        da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].median(dim='run')
        da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
        df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
        df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
        df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
    gdf = cp(gdf_country_borders.reset_index())
    gdf_p = cp(gdf_country_borders.reset_index())

    print('panel e,f,g: country numbers')
    for step in (0,10,20):
        print('GMT is {}'.format(df_GMT_strj.loc[2100,step]))
        gdf_p['pf']=df_p_gs_plot['pf'][df_p_gs_plot['GMT']==step].values      
        print('number of countries with pf > 50% is : {}'.format(len(gdf_p['pf'][gdf_p['pf']>50])))  
        print('number of countries with pf > 90% is : {}'.format(len(gdf_p['pf'][gdf_p['pf']>90])))
        print('number of countries with pf = 100% is : {}'.format(len(gdf_p['pf'][gdf_p['pf']==100])))  

#%% ----------------------------------------------------------------
# f3 numbers
# ------------------------------------------------------------------
        
def print_f3_info(
    flags,
    da_gs_popdenom
):
    
    extremes = [
        'heatwavedarea',     
        'cropfailedarea', 
        'burntarea', 
        'driedarea', 
        'floodedarea', 
        'tropicalcyclonedarea',
    ]
    unprec_level="unprec_99.99"

    list_extrs_pf = []
    for extr in extremes:
        with open('./data/{}/{}/isimip_metadata_{}_ar6_new_rm.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)  
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        p = ds_pf_gs_extr[unprec_level].loc[{
            'GMT':np.arange(GMT_indices_plot[0],GMT_indices_plot[-1]+1).astype('int'),
        }].sum(dim='country')       
        p = p.where(p!=0).mean(dim='run') / da_gs_popdenom.sum(dim='country') *100
        list_extrs_pf.append(p)
        
    ds_pf_gs_extrs = xr.concat(list_extrs_pf,dim='hazard').assign_coords({'hazard':extremes})

    for extr in extremes:
        print(extr)
        print('pf for 2020 under 3.5 degree pathway is {}'.format(ds_pf_gs_extrs.loc[{'hazard':extr,'birth_year':2020,'GMT':0}].item()))            

#%% ----------------------------------------------------------------
# save the children info
# ------------------------------------------------------------------

# 1.	Difference of exposure to all six climate extremes at 1.5, 2.7 and 3.5°C for children born in 2020 (in absolute numbers & percent)
# 2.	Number of countries and regions affected by unprecedented heatwave exposure for more than 50% of the children born in 2020
# 3.	Difference of unprecedented exposure to heatwaves between the most and least vulnerable for different birth cohorts (intergenerational and socioeconomic inequality)

# input args for function
# gridscale_countries
# d_isimip_meta
# da_gs_popdenom
# df_GMT_strj
# gdf_country_borders
def save_the_children_stuff(
    gridscale_countries,
    flags,
    da_gs_popdenom,
    df_GMT_strj,
    gdf_country_borders,
    countries_mask,
    countries_regions,
    da_cohort_size_1960_2020,
    
):
    # ------------------------------------------------------------------
    # 1 Difference of exposure to all six climate extremes at 1.5, 2.7 and 3.5°C for children born in 2020 (in absolute numbers & percent)

    unprec_level="unprec_99.99"  
    gmt_share = [0,12,20] # 1.5, 2.7 and 3.5
    extremes = [
        'heatwavedarea',     
        'cropfailedarea', 
        'burntarea', 
        'driedarea', 
        'floodedarea', 
        'tropicalcyclonedarea',
    ]

    # pop fraction dataset with extra country index for "Globe" and other features adjusted for excel export
    new_countries = np.append(gridscale_countries,'Globe')
    ds_pf_share = xr.Dataset(
        data_vars={
            'unprec': (
                ['country','GMT','hazard'],
                np.full(
                    (len(new_countries),len(gmt_share),len(extremes)),
                    fill_value=np.nan,
                ),
            ),      
            'unprec_mill': (
                ['country','GMT','hazard'],
                np.full(
                    (len(new_countries),len(gmt_share),len(extremes)),
                    fill_value=np.nan,
                ),
            ),              
            'unprec_frac': (
                ['country','GMT','hazard'],
                np.full(
                    (len(new_countries),len(gmt_share),len(extremes)),
                    fill_value=np.nan,
                ),
            ),              
        },
        coords={
            'country': ('country', new_countries),
            'GMT': ('GMT', gmt_share),
            'hazard': ('hazard', extremes)
        }
    )       

    # loop through extremes and export
    for extr in extremes:
        with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)            
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            ds_pf_gs_extr = pk.load(file)    
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)               
        for gmt in gmt_share:
            p = ds_pf_gs_extr[unprec_level].loc[{
                'birth_year':2020,
                'GMT':gmt,
                'run':sims_per_step[gmt]
            }]
            # p = p.where(p!=0).mean(dim='run') # Jan 8th note: haven't selected specific GMTs or their relevant runs with sims_per_step[step] here, which is slightly skewing numbers
            p = p.mean(dim='run')
            p_global = p.sum(dim='country')  
            p_frac = p / da_gs_popdenom.loc[{'birth_year':2020}]*100
            p_global_frac = p_global / da_gs_popdenom.loc[{'birth_year':2020}].sum(dim='country') *100
            # assign country level:
            ds_pf_share['unprec'].loc[{
                'country':new_countries[:-1],
                'GMT':gmt,
                'hazard':extr,
            }] = p
            ds_pf_share['unprec_mill'].loc[{
                'country':new_countries[:-1],
                'GMT':gmt,
                'hazard':extr,
            }] = p / 10**6    
            ds_pf_share['unprec_frac'].loc[{
                'country':new_countries[:-1],
                'GMT':gmt,
                'hazard':extr,
            }] = p_frac
            # assign global level:
            ds_pf_share['unprec'].loc[{
                'country':new_countries[-1],
                'GMT':gmt,
                'hazard':extr,
            }] = p_global
            ds_pf_share['unprec_mill'].loc[{
                'country':new_countries[-1],
                'GMT':gmt,
                'hazard':extr,
            }] = p_global / 10**6        
            ds_pf_share['unprec_frac'].loc[{
                'country':new_countries[-1],
                'GMT':gmt,
                'hazard':extr,
            }] = p_global_frac        
        # organize/export to excel
        df = ds_pf_share['unprec'].loc[{'hazard':extr}].to_dataframe().reset_index(level='country')
        df = df.drop(labels=['hazard'],axis=1).pivot_table(values='unprec',index=df.index,columns='country')
        df_mill = ds_pf_share['unprec_mill'].loc[{'hazard':extr}].to_dataframe().reset_index(level='country')
        df_mill = df_mill.drop(labels=['hazard'],axis=1).pivot_table(values='unprec_mill',index=df_mill.index,columns='country')
        df_frac = ds_pf_share['unprec_frac'].loc[{'hazard':extr}].to_dataframe().reset_index(level='country')
        df_frac = df_frac.drop(labels=['hazard'],axis=1).pivot_table(values='unprec_frac',index=df_frac.index,columns='country')
        df.index = df_GMT_strj.loc[2100,df_frac.index.values]
        df.index.names = ['GMT']    
        df_mill.index = df_GMT_strj.loc[2100,df_frac.index.values]
        df_mill.index.names = ['GMT']       
        df_frac.index = df_GMT_strj.loc[2100,df_frac.index.values]
        df_frac.index.names = ['GMT']
        df.to_excel('./data/save_the_children/data_1_redo/unprecedented_absolute_{}.xlsx'.format(extr))
        df_mill.to_excel('./data/save_the_children/data_1_redo/unprecedented_millions_{}.xlsx'.format(extr))
        df_frac.to_excel('./data/save_the_children/data_1_redo/unprecedented_percent_{}.xlsx'.format(extr))    
    # for extr in extremes:
    #     with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
    #         ds_pf_gs_extr = pk.load(file)    
    #     p = ds_pf_gs_extr[unprec_level].loc[{
    #         'birth_year':2020,
    #         'GMT':gmt_share,
    #     }]
    #     p = p.where(p!=0).mean(dim='run') # Jan 8th note: haven't selected specific GMTs or their relevant runs with sims_per_step[step] here, which is slightly skewing numbers
    #     p_global = p.sum(dim='country')  
    #     p_frac = p / da_gs_popdenom.loc[{'birth_year':2020}]*100
    #     p_global_frac = p_global / da_gs_popdenom.loc[{'birth_year':2020}].sum(dim='country') *100
    #     # assign country level:
    #     ds_pf_share['unprec'].loc[{
    #         'country':new_countries[:-1],
    #         'GMT':gmt_share,
    #         'hazard':extr,
    #     }] = p
    #     ds_pf_share['unprec_mill'].loc[{
    #         'country':new_countries[:-1],
    #         'GMT':gmt_share,
    #         'hazard':extr,
    #     }] = p / 10**6    
    #     ds_pf_share['unprec_frac'].loc[{
    #         'country':new_countries[:-1],
    #         'GMT':gmt_share,
    #         'hazard':extr,
    #     }] = p_frac
    #     # assign global level:
    #     ds_pf_share['unprec'].loc[{
    #         'country':new_countries[-1],
    #         'GMT':gmt_share,
    #         'hazard':extr,
    #     }] = p_global
    #     ds_pf_share['unprec_mill'].loc[{
    #         'country':new_countries[-1],
    #         'GMT':gmt_share,
    #         'hazard':extr,
    #     }] = p_global / 10**6        
    #     ds_pf_share['unprec_frac'].loc[{
    #         'country':new_countries[-1],
    #         'GMT':gmt_share,
    #         'hazard':extr,
    #     }] = p_global_frac        
    #     # organize/export to excel
    #     df = ds_pf_share['unprec'].loc[{'hazard':extr}].to_dataframe().reset_index(level='country')
    #     df = df.drop(labels=['hazard'],axis=1).pivot_table(values='unprec',index=df.index,columns='country')
    #     df_mill = ds_pf_share['unprec_mill'].loc[{'hazard':extr}].to_dataframe().reset_index(level='country')
    #     df_mill = df_mill.drop(labels=['hazard'],axis=1).pivot_table(values='unprec_mill',index=df_mill.index,columns='country')
    #     df_frac = ds_pf_share['unprec_frac'].loc[{'hazard':extr}].to_dataframe().reset_index(level='country')
    #     df_frac = df_frac.drop(labels=['hazard'],axis=1).pivot_table(values='unprec_frac',index=df_frac.index,columns='country')
    #     df.index = df_GMT_strj.loc[2100,df_frac.index.values]
    #     df.index.names = ['GMT']    
    #     df_mill.index = df_GMT_strj.loc[2100,df_frac.index.values]
    #     df_mill.index.names = ['GMT']       
    #     df_frac.index = df_GMT_strj.loc[2100,df_frac.index.values]
    #     df_frac.index.names = ['GMT']
    #     # df.to_excel('./data/save_the_children/data_1/unprecedented_absolute_{}.xlsx'.format(extr))
    #     df_mill.to_excel('./data/save_the_children/data_1_redo/unprecedented_millions_{}.xlsx'.format(extr))
    #     # df_frac.to_excel('./data/save_the_children/data_1/unprecedented_percent_{}.xlsx'.format(extr))
        
    # ------------------------------------------------------------------
    # 2 Number of countries and regions affected by unprecedented heatwave exposure for more than 50% of the children born in 2020

    plot_var='unprec_99.99'
    gmt_indices_152535 = [20,12,10,0]  

    # pf threshold
    pf_threshold=25
    # box plot stuff
    df_list_gs = []
    # extr='heatwavedarea'
    extremes = [ # this array of extremes strings should be the same as the setup function
        'burntarea', 
        'cropfailedarea', 
        'driedarea', 
        'floodedarea', 
        # 'heatwavedarea', 
        'tropicalcyclonedarea',
    ]
    for extr in extremes:
        with open('./data/{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)              
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            ds_pf_gs = pk.load(file)

        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)
            
        # map stuff
        by=2020
        da_p_gs_plot = ds_pf_gs[plot_var].loc[{
            'GMT':gmt_indices_152535,
            'birth_year':by,
        }]
        df_list_gs = []
        for step in gmt_indices_152535:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].median(dim='run')
            da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
            df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
            df_p_gs_plot_step = df_p_gs_plot_step.assign(GMT_label = lambda x: np.round(df_GMT_strj.loc[2100,x['GMT']],1).values.astype('str'))
            df_list_gs.append(df_p_gs_plot_step)
        df_p_gs_plot = pd.concat(df_list_gs)
        df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
        gdf = cp(gdf_country_borders.reset_index())
        gdf_p = cp(gdf_country_borders.reset_index())

        concat_list=[]    
        for step in (0,10,12,20):
            print('GMT is {}'.format(df_GMT_strj.loc[2100,step]))
            gdf_p['pf']=df_p_gs_plot['pf'][df_p_gs_plot['GMT']==step].values
            print('number of countries with pf > {}% is : {}'.format(pf_threshold,len(gdf_p['pf'][gdf_p['pf']>pf_threshold])))  
            gmt_gdf_concat=cp(gdf_p)
            gmt_gdf_concat=gmt_gdf_concat.drop(labels='geometry',axis=1)
            gmt_gdf_concat=gmt_gdf_concat.pivot_table(values='pf',columns='name')     
            gmt_gdf_concat = gmt_gdf_concat.reset_index().drop(labels='index',axis=1)
            gmt_gdf_concat.index = [df_GMT_strj.loc[2100,step]]
            gmt_gdf_concat.index.names = ['GMT']
            gmt_gdf_concat.columns.names = ['country']
            gmt_gdf_concat['countries above 50%'] = len(gdf_p['pf'][gdf_p['pf']>pf_threshold])
            concat_list.append(gmt_gdf_concat)
            
        gdf_export = pd.concat(concat_list)
        gdf_export.to_excel('./data/save_the_children/data_2/countries_over_{}_{}.xlsx'.format(pf_threshold,extr))
        
    # ------------------------------------------------------------------
    # 3 Difference of unprecedented exposure to heatwaves between the most and least vulnerable for different birth cohorts (intergenerational and socioeconomic inequality)

    sl=0.05 # significance testing level for asterisks
    extremes = [ # this array of extremes strings should be the same as the setup function
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    # GMT_integers = [0,10,12,17,20] # 1.5, 2.5, 2.7, 3.2 and 3.5
    GMT_integers = [0,10,12,20] # 1.5, 2.5, 2.7, and 3.5
    qntl_range = '20'
    vln_types=('grdi','gdp')

    # per vulnerability indicator
    for vln_type in vln_types:
        print('')
        print(vln_type)
        print('')
        with open('./data/{}/pyramid_data_{}.pkl'.format(flags['version'],vln_type), 'rb') as f:
            d_pyramid_plot = pk.load(f) 
        for e in extremes:    
            for GMT in GMT_integers:
                df = pd.DataFrame(
                    index=birth_years,
                    columns=[
                        'Unprecedented population (in millions) among the least vulnerable', 
                        'Unprecedented percentage of the least vulnerable',
                        'Unprecedented population (in millions) among the most vulnerable', 
                        'Unprecedented percentage of the most vulnerable',
                    ]
                )         
                poor_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}poorest'.format(qntl_range)]) # "_a" for panel "a"
                poor_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}poorest'.format(qntl_range)])
                rich_unprec = np.asarray(d_pyramid_plot[e][GMT]['unprec_pop_quantiles_{}richest'.format(qntl_range)])
                rich_pop = np.asarray(d_pyramid_plot[e][GMT]['population_quantiles_{}richest'.format(qntl_range)])
                pvalues_poor = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_poor'.format(qntl_range)])
                pvalues_rich = np.asarray(d_pyramid_plot[e][GMT]['ttest_{}pc_pvals_rich'.format(qntl_range)])
                
                print('')
                print('{} degree C pathway'.format(df_GMT_strj.loc[2100,GMT]))
                print('')
                print('ULE population for poorest is: \n {}'.format(poor_unprec))
                df.loc[:,'Unprecedented population (in millions) among the most vulnerable'] = poor_unprec
                print('percentage of ULE for poorest is: \n {}'.format(poor_unprec / poor_pop * 100))
                df.loc[:,'Unprecedented percentage of the most vulnerable'] = poor_unprec / poor_pop * 100
                print('unprecedented population for richest is: \n {}'.format(rich_unprec))
                df.loc[:,'Unprecedented population (in millions) among the least vulnerable'] = rich_unprec
                print('percentage of ULE for richest is: \n {}'.format(rich_unprec / rich_pop * 100))
                df.loc[:,'Unprecedented percentage of the least vulnerable'] = rich_unprec / rich_pop * 100
                print('p values significant: \n {}'.format(pvalues_poor < sl))
                df.index.names = ['Birth cohort']
                df.to_excel('./data/save_the_children/data_3/{}_{}.xlsx'.format(vln_type,df_GMT_strj.loc[2100,GMT]))
                
    # ------------------------------------------------------------------
    # 4 Millions unprecedented  from 2003-2020 for all hazards

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

    pic_qntl_str=str(pic_qntl*100)
    gmts = GMT_labels
    gmts2100 = np.round(df_GMT_strj.loc[2100,gmts].values,1)   
    gmt_dict = dict(zip(gmts,gmts2100))
    concat_list_0 = []
    concat_list_12 = []
    concat_list_20 = []
    for extr in extremes:
        
        print(extr)
        with open('./data/pickles_v2/{}/isimip_metadata_{}_{}_{}.pkl'.format(extr,extr,flags['gmt'],flags['rm']), 'rb') as file:
            d_isimip_meta = pk.load(file)    
        with open('./data/pickles_v2/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)            

        sims_per_step = {}
        for step in gmts:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)  
        # millions children unprecedented in 1.5 pathway
        step=0
        unprec_15 = ds_pf_gs['unprec_{}'.format(pic_qntl_str)].sum(dim='country').loc[{'GMT':step,'run':sims_per_step[step],'birth_year':np.arange(2003,2021)}].mean(dim='run') / 10**6
        concat_list_0.append(unprec_15)
        
        # millions children unprecedented in 2.7 pathway
        step=12
        unprec_27 = ds_pf_gs['unprec_{}'.format(pic_qntl_str)].sum(dim='country').loc[{'GMT':step,'run':sims_per_step[step],'birth_year':np.arange(2003,2021)}].mean(dim='run') / 10**6
        concat_list_12.append(unprec_27)
        
        # millions children unprecedented in 3.5 pathway
        step=20
        unprec_35 = ds_pf_gs['unprec_{}'.format(pic_qntl_str)].sum(dim='country').loc[{'GMT':step,'run':sims_per_step[step],'birth_year':np.arange(2003,2021)}].mean(dim='run') / 10**6
        concat_list_20.append(unprec_35)    
            
        print('')
        
    concat_da_0 = np.round(xr.concat(concat_list_0,dim='hazard').assign_coords({'hazard':extremes}),1)
    df_0 = concat_da_0.to_dataframe().reset_index(level='hazard')
    df_0 = df_0.pivot_table(values='unprec_99.99',index='birth_year',columns='hazard')
    df_0.to_excel('./data/save_the_children/data_4/millions_unprec_1.5.xlsx')

    concat_da_12 = np.round(xr.concat(concat_list_12,dim='hazard').assign_coords({'hazard':extremes}),1)
    df_12 = concat_da_12.to_dataframe().reset_index(level='hazard')
    df_12 = df_12.pivot_table(values='unprec_99.99',index='birth_year',columns='hazard')
    df_12.to_excel('./data/save_the_children/data_4/millions_unprec_2.7.xlsx')

    concat_da_20 = np.round(xr.concat(concat_list_20,dim='hazard').assign_coords({'hazard':extremes}),1)
    df_20 = concat_da_20.to_dataframe().reset_index(level='hazard')
    df_20 = df_20.pivot_table(values='unprec_99.99',index='birth_year',columns='hazard')
    df_20.to_excel('./data/save_the_children/data_4/millions_unprec_3.5.xlsx')   

    # ------------------------------------------------------------------
    # 5 GRDI request

    # grdi request
    ds_grdi = xr.open_dataset('./data/deprivation/grdi_con_nanreplace_isimipgrid.nc4')
    grdi = ds_grdi['grdi']
    cntry_concat = []

    for cntry in gridscale_countries:
        
        da_cntry = xr.DataArray(
            np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
            dims=countries_mask.dims,
            coords=countries_mask.coords,
        )
        da_cntry = da_cntry.where(da_cntry,drop=True)    
        grdi_c = grdi.where(da_cntry)
        pop = da_cohort_size_1960_2020.where(da_cntry)
        
        by_concat = []
        
        for by in birth_years:
        
            pop_by = da_cohort_size_1960_2020.sel(birth_year=by)
            
            grdi_c_by = grdi_c.where(pop_by.notnull())
            pop_by = pop_by.where(grdi_c_by.notnull())
            
            wgrdi=grdi_c_by.fillna(0).weighted(pop_by.fillna(0)).mean(('lat','lon'))    
            by_concat.append(wgrdi)
        
        da_wgrdi_cntry = xr.DataArray(
            data=by_concat,
            coords={'birth_year':birth_years}
        )
        
        cntry_concat.append(da_wgrdi_cntry)
        
    da_wgrdi_countries = np.round(xr.concat(cntry_concat,dim='country').assign_coords({'country':gridscale_countries}),2)
    df_wgrdi = da_wgrdi_countries.to_dataframe(name='grdi').reset_index()
    df_wgrdi = df_wgrdi.pivot_table(values='grdi',index='birth_year',columns='country')
    df_wgrdi.to_excel('./data/save_the_children/data_5/grdi_per_country.xlsx')   

    # ------------------------------------------------------------------
    # 6 Vector request (ended up just giving them the pdf from plot_si.py function)

    # vector request
    # since wer're looking at country level means across runs, denominator is important and 0s need to be accounted for in non-emergence
    # so we only take sims or runs valid per GMT level and make sure nans are 0
    extremes = [
        # 'burntarea', 
        # 'cropfailedarea', 
        # 'driedarea', 
        # 'floodedarea', 
        'heatwavedarea', 
        # 'tropicalcyclonedarea',
    ]
    unprec_level="unprec_99.99"
    by=2020
    gmt_indices_152535 = [0,10,20]
    df_list_gs = []
    for extr in extremes:
        with open('./data/{}/{}/isimip_metadata_{}_ar6_new_rm.pkl'.format(flags['version'],extr,extr), 'rb') as file:
            d_isimip_meta = pk.load(file)         
        with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],extr,extr), 'rb') as f:
            ds_pf_gs = pk.load(f)  
        da_p_gs_plot = ds_pf_gs[unprec_level].loc[{
            'GMT':gmt_indices_152535,
            'birth_year':by,
        }]          
        sims_per_step = {}
        for step in GMT_labels:
            sims_per_step[step] = []
            for i in list(d_isimip_meta.keys()):
                if d_isimip_meta[i]['GMT_strj_valid'][step]:
                    sims_per_step[step].append(i)        
        for step in gmt_indices_152535:
            da_p_gs_plot_step = da_p_gs_plot.loc[{'run':sims_per_step[step],'GMT':step}].mean(dim='run')
            da_p_gs_plot_step = da_p_gs_plot_step / da_gs_popdenom.loc[{'birth_year':by}] * 100
            df_p_gs_plot_step = da_p_gs_plot_step.to_dataframe(name='pf').reset_index()
            df_p_gs_plot_step['extreme'] = extr
            df_list_gs.append(df_p_gs_plot_step)
    df_p_gs_plot = pd.concat(df_list_gs)
    df_p_gs_plot['pf'] = df_p_gs_plot['pf'].fillna(0)  
    gdf_p = cp(gdf_country_borders.reset_index())
    gdf_p = gdf_p.rename({'name':'country'})
    df_geom_pf = gdf_p.merge(df_p_gs_plot,left_on='name',right_on='country',how='left')
    df_geom_pf = df_geom_pf.drop(['extreme','birth_year','region','name'],axis=1)

    # get geodataframes per warming target for heatwaves
    df_geom_pf_15 = df_geom_pf[df_geom_pf['GMT']==0].drop(['GMT'],axis=1)
    df_geom_pf_25 = df_geom_pf[df_geom_pf['GMT']==10].drop(['GMT'],axis=1)
    df_geom_pf_35 = df_geom_pf[df_geom_pf['GMT']==20].drop(['GMT'],axis=1)

    # save to shapefiles
    df_geom_pf_15.to_file('./data/save_the_children/data_6/CF_heatwaves_1.5.shp')
    df_geom_pf_25.to_file('./data/save_the_children/data_6/CF_heatwaves_2.5.shp')
    df_geom_pf_35.to_file('./data/save_the_children/data_6/CF_heatwaves_3.5.shp')         

#%% ----------------------------------------------------------------
# save the children info for website
# ------------------------------------------------------------------

def website_exposure_means(
    flags,
    gridscale_countries,
    GMT_labels,
    GMT_indices_plot,
    year_range,
    countries_mask,
    countries_regions,
    da_population,
    df_life_expectancy_5,
    d_isimip_meta,
):
    
    # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
    if not os.path.exists('./data/{}/website_{}.pkl'.format(flags['version'],flags['extr'])):
    
        # use da_population as input for pop weighted mean
        # lifetime exposure dataset (pop weighted mean of pixel scale lifetime exposure per country, run, GMT and birthyear)
        ds_e = xr.Dataset(
            data_vars={
                'exposure_popweight': (
                    ['country','GMT','time'],
                    np.full(
                        (len(gridscale_countries),len(GMT_labels),len(year_range)),
                        fill_value=np.nan,
                    ),
                ),
                'exposure_latweight': (
                    ['country','GMT','time'],
                    np.full(
                        (len(gridscale_countries),len(GMT_labels),len(year_range)),
                        fill_value=np.nan,
                    ),
                )
            },
            coords={
                'country': ('country', gridscale_countries),
                'time': ('time', year_range),
                'GMT': ('GMT', GMT_labels)
            }
        )  

        for i,cntry in enumerate(gridscale_countries):

            print('country # {} of {}, {}'.format(i,len(gridscale_countries),cntry))

            # country mask and weights for latitude (probably won't use but will use population instead)
            da_cntry = xr.DataArray(
                np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                dims=countries_mask.dims,
                coords=countries_mask.coords,
            )
            da_cntry = da_cntry.where(da_cntry,drop=True)
            da_cntry_population = da_population.where(da_cntry,drop=True)
            lat_weights = np.cos(np.deg2rad(da_cntry.lat))
            lat_weights.name = "weights"      
            
            popweight_sample = []
            latweight_sample = []

            # loop over GMT trajectories
            for step in GMT_labels:
                
                print('gmt step {}'.format(step))

                # loop over simulations
                for i in list(d_isimip_meta.keys()): 

                    print('simulation {} of {}'.format(i,len(d_isimip_meta)))

                    # load AFA data of that run
                    with open('./data/{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'rb') as f:
                        da_AFA = pk.load(f)
                        
                    # mask to sample country and reduce spatial extent
                    da_AFA = da_AFA.where(da_cntry==1,drop=True)
                                        
                    # run GMT-mapping of years if valid
                    if d_isimip_meta[i]['GMT_strj_valid'][step]:
                        
                        # GMT-mapping
                        da_AFA_step = da_AFA.reindex(
                            {'time':da_AFA['time'][d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step]]}
                        ).assign_coords({'time':year_range})
                        
                        # lat-weighted mean
                        da_AFA_step_lw = da_AFA_step.weighted(lat_weights).mean(('lat','lon'))       
                        latweight_sample.append(da_AFA_step_lw)
                        
                        # pop-weighted mean
                        da_AFA_step_pw = da_AFA_step.weighted(da_cntry_population).mean(('lat','lon'))     
                        popweight_sample.append(da_AFA_step_pw)
                            
                # assign 
                ds_e['exposure_latweight'].loc[{
                    'country':cntry,
                    'time':year_range,
                    'GMT':step,
                }] = xr.concat(latweight_sample,dim='run').mean(dim='run')
                ds_e['exposure_popweight'].loc[{
                    'country':cntry,
                    'time':year_range,
                    'GMT':step,
                }] = xr.concat(popweight_sample,dim='run').mean(dim='run')
                
                # then do birth year dependent and country dependent cumulative exposure
                if step in GMT_indices_plot:
                    
                    for by in birth_years:
                        
                        le = df_life_expectancy_5.loc[by,cntry] # life expectancy
                        le_fractional = le - np.floor(le) # fractional exposure of death year
                        lifespan = np.arange(by,by+le)
                        
                        # cumulative exposure from birth to death year
                        ce = ds_e['exposure_popweight'].loc[{
                            'country':cntry,
                            'time':lifespan,
                            'GMT':step,
                        }].cumsum(dim='time')
                        
                        # the fractional expsoure from incomplete final year lived
                        e_fractional = ds_e['exposure_popweight'].loc[{
                            'country':cntry,
                            'time':lifespan[-1]+1,
                            'GMT':step,
                        }] * le_fractional
                        
                        # add that fractional exposure to the final year of the rounded lifespan
                        ce.loc[{'time':lifespan[-1]}] = ce.loc[{'time':lifespan[-1]}] + e_fractional
                        
                        # place ce into cumulative exposure dataset
                        ds_ce['cumulative_exposure_popweight'].loc[{
                            'country':cntry,
                            'time':lifespan,
                            'birth_year':by,
                            'GMT':step,
                        }] = ce
                    
                    
        # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
        with open('./data/{}/website_{}.pkl'.format(flags['version'],flags['extr']), 'wb') as f:
            pk.dump(ds_e,f)         
            
    else:
        
        # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
        with open('./data/{}/website_{}.pkl'.format(flags['version'],flags['extr']), 'rb') as f:
            ds_e = pk.load(f)

    # adding on cumulative exposure
    if not os.path.exists('./data/{}/website_cumulative_exposure_{}.nc'.format(flags['version'],flags['extr'])):
        
        # cumulative exposure
        ds_ce = xr.Dataset(
            data_vars={
                'cumulative_exposure_popweight': (
                    ['country','GMT','birth_year','time'],
                    np.full(
                        (len(gridscale_countries),len(GMT_indices_plot),len(birth_years),len(year_range)),
                        fill_value=np.nan,
                    ),
                ),
            },
            coords={
                'country': ('country', gridscale_countries),
                'birth_year': ('birth_year', birth_years),
                'time': ('time', year_range),
                'GMT': ('GMT', GMT_indices_plot)
            }
        )              
        
        # loop thru countries
        for i,cntry in enumerate(gridscale_countries):
            for step in GMT_labels:
                if step in GMT_indices_plot:
                    for by in birth_years:
                        
                        le = df_life_expectancy_5.loc[by,cntry] # life expectancy
                        le_fractional = le - np.floor(le) # fractional exposure of death year
                        lifespan = np.arange(by,by+le)
                        
                        # cumulative exposure from birth to death year
                        ce = ds_e['exposure_popweight'].loc[{
                            'country':cntry,
                            'time':lifespan,
                            'GMT':step,
                        }].cumsum(dim='time')
                        
                        # the fractional expsoure from incomplete final year lived
                        e_fractional = ds_e['exposure_popweight'].loc[{
                            'country':cntry,
                            'time':lifespan[-1]+1,
                            'GMT':step,
                        }] * le_fractional
                        
                        # add that fractional exposure to the final year of the rounded lifespan
                        ce.loc[{'time':lifespan[-1]}] = ce.loc[{'time':lifespan[-1]}] + e_fractional
                        
                        # place ce into cumulative exposure dataset
                        ds_ce['cumulative_exposure_popweight'].loc[{
                            'country':cntry,
                            'time':lifespan,
                            'birth_year':by,
                            'GMT':step,
                        }] = ce       
                        
        # to netcdf
        ds_ce.to_netcdf('./data/{}/website_cumulative_exposure_{}.nc'.format(flags['version'],flags['extr']),format='NETCDF4')
        
    else:
        
        ds_ce = xr.open_dataset('./data/{}/website_cumulative_exposure_{}.nc'.format(flags['version'],flags['extr']))
            
    return ds_e, ds_ce

#%% ----------------------------------------------------------------
# save the children info for website
# ------------------------------------------------------------------

def website_pic_threshold(
    flags,
    gridscale_countries,
    countries_mask,
    countries_regions,
    da_population,
    d_pic_meta,
):
    
    # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
    if not os.path.exists('./data/{}/website_{}_pic_thresholds.nc'.format(flags['version'],flags['extr'])):
    
        # use da_population as input for pop weighted mean
        # lifetime exposure dataset (pop weighted mean of pixel scale lifetime exposure per country, run, GMT and birthyear)
        ds_pic_qntl_website = xr.Dataset(
            data_vars={
                'pic_popweight': (
                    ['country'],
                    np.full(
                        (len(gridscale_countries),),
                        fill_value=np.nan,
                    ),
                ),
                'pic_latweight': (
                    ['country'],
                    np.full(
                        (len(gridscale_countries)),
                        fill_value=np.nan,
                    ),
                )
            },
            coords={
                'country': ('country', gridscale_countries),
            }
        )  

        for i,cntry in enumerate(gridscale_countries):

            print('country # {} of {}, {}'.format(i,len(gridscale_countries),cntry))

            # country mask and weights for latitude (probably won't use but will use population instead)
            da_cntry = xr.DataArray(
                np.in1d(countries_mask,countries_regions.map_keys(cntry)).reshape(countries_mask.shape),
                dims=countries_mask.dims,
                coords=countries_mask.coords,
            )
            da_cntry = da_cntry.where(da_cntry,drop=True)    
            da_cntry_population = da_population.where(da_cntry,drop=True)
            lat_weights = np.cos(np.deg2rad(da_cntry.lat))
            lat_weights.name = "weights"      
            
            popweight_sample = []
            latweight_sample = []

            # loop over simulations
            for i in list(d_pic_meta.keys()):

                print('simulation {} of {}'.format(i,len(d_pic_meta)))
                
                with open('./data/{}/{}/{}/gridscale_pic_qntls_{}_{}.pkl'.format(flags['version'],flags['extr'],cntry,flags['extr'],cntry), 'rb') as f:
                    ds_pic_qntl = pk.load(f)                           
                        
                # lat-weighted mean
                da_pic_lw = ds_pic_qntl['99.99'].weighted(lat_weights).mean(('lat','lon'))       
                latweight_sample.append(da_pic_lw)
                
                # pop-weighted mean
                da_pic_pw = ds_pic_qntl['99.99'].weighted(da_cntry_population.sel(time=1960)).mean(('lat','lon'))     
                popweight_sample.append(da_pic_pw)
                            
            # assign and run means
            ds_pic_qntl_website['pic_latweight'].loc[{
                'country':cntry,
            }] = xr.concat(latweight_sample,dim='run').mean(dim='run')
            ds_pic_qntl_website['pic_popweight'].loc[{
                'country':cntry,
            }] = xr.concat(popweight_sample,dim='run').mean(dim='run')
                    
        # save to nc
        ds_pic_qntl_website.to_netcdf('./data/{}/website_{}_pic_thresholds.nc'.format(flags['version'],flags['extr']),format='NETCDF4')
            
    else:
        
        # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
        ds_pic_qntl_website = xr.open_dataset('./data/{}/website_{}_pic_thresholds.nc'.format(flags['version'],flags['extr']))
            
    return ds_pic_qntl_website

#%% ----------------------------------------------------------------
# save the children info for website piped to excel
# ------------------------------------------------------------------

def ds_e_to_excel(
    flags,
    ds_e,
    df_GMT_strj,
):

    # then exort to excel, limit each country's time axis based on life expectancy in 2020 ...
    excel_file_pw = './data/{}/website_exposure_population_weighted_{}.xlsx'.format(flags['version'],flags['extr'])
    excel_file_lw = './data/{}/website_exposure_latitude_weighted_{}.xlsx'.format(flags['version'],flags['extr'])
    
    # write pop weighted results to excel
    with pd.ExcelWriter(excel_file_pw, engine="openpyxl") as writer:
        for gmt in ds_e.GMT.values:
            gmt_label = np.round(df_GMT_strj.loc[2100,gmt],1).astype('str')
            df = ds_e['exposure_popweight'].sel(GMT=gmt).to_dataframe().reset_index(level='country')
            df_rearrange = df.pivot_table(values='exposure_popweight',index=df.index,columns='country')
            df_rearrange.to_excel(writer, sheet_name=gmt_label)
            
    # write lat weigthed results to excel
    with pd.ExcelWriter(excel_file_lw, engine="openpyxl") as writer:
        for gmt in ds_e.GMT.values:
            gmt_label = np.round(df_GMT_strj.loc[2100,gmt],1).astype('str')
            df = ds_e['exposure_latweight'].sel(GMT=gmt).to_dataframe().reset_index(level='country')
            df_rearrange = df.pivot_table(values='exposure_latweight',index=df.index,columns='country')
            df_rearrange.to_excel(writer, sheet_name=gmt_label)