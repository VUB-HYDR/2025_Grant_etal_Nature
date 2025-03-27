#!/usr/bin/env python3
# ---------------------------------------------------------------
# Main script to postprocess and visualise lifetime exposure data
#
# Python translation of the MATLAB scripts of Thiery et al. (2021)
# https://github.com/VUB-HYDR/2021_Thiery_etal_Science
# ----------------------------------------------------------------

#%% ----------------------------------------------------------------
# Summary and notes

# to save the enironment used (with your path to the env directory): 
# conda env export -p C:\Users\ivand\anaconda3\envs\exposure_env > exposure_env.yml


# Data types are defined in the variable names starting with:  
#     df_     : DataFrame    (pandas)
#     gdf_    : GeoDataFrame (geopandas)
#     da_     : DataArray    (xarray)
#     d_      : dictionary  
#     sf_     : shapefile
#     ...dir  : directory

#               
#%%  ----------------------------------------------------------------
# import and path
# ----------------------------------------------------------------

import xarray as xr
import pickle as pk
import time
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import mapclassify as mc
from copy import deepcopy as cp
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy as cr
import geopandas as gpd
# import seaborn as sns # must comment this out for things to work on the server
scriptsdir = os.getcwd()


#%% ----------------------------------------------------------------
# flags
# ----------------------------------------------------------------

# extreme event
global flags

flags = {}
flags['extr'] = 'heatwavedarea' # 0: all
                                # 1: burntarea
                                # 2: cropfailedarea
                                # 3: driedarea
                                # 4: floodedarea
                                # 5: heatwavedarea
                                # 6: tropicalcyclonedarea
                                # 7: waterscarcity
flags['gmt'] = 'ar6_new'    # original: use Wim's stylized trajectory approach with max trajectory a linear increase to 3.5 deg                               
                            # ar6: substitute the linear max wth the highest IASA c7 scenario (increasing to ~4.0), new lower bound, and new 1.5, 2.0, NDC (2.8), 3.0
                            # ar6_new: works off ar6, but ensures only 1.5-3.5 with perfect intervals of 0.1 degrees (less proc time and data volume)
flags['rm'] = 'rm'       # no_rm: no smoothing of RCP GMTs before mapping
                         # rm: 21-year rolling mean on RCP GMTs 
flags['version'] = 'pickles_v2'     # pickles: original version, submitted to Nature
                                        # inconsistent GMT steps (not perfect 0.1 degree intervals)
                                        # GMT steps ranging 1-4 (although study only shows ~1.5-3.5, so runs are inefficient)
                                        # only 99.99% percentile for PIC threshold
                                    # pickles_v2: version generated after submission to Nature in preparation for criticism/review
                                        # steps fixed in load_manip to be only 1.5-3.5, with clean 0.1 degree intervals
                                        # 5 percentiles for PIC threshold and emergence for each
                                    # pickles_v3: version generated after the 2021 toolchains were taken away from hydra. could not longer use old pickles effectively
flags['run'] = 0          # 0: do not process ISIMIP runs (i.e. load runs pickle)
                            # 1: process ISIMIP runs (i.e. produce and save runs as pickle)
flags['mask'] = 0           # 0: do not process country data (i.e. load masks pickle)
                            # 1: process country data (i.e. produce and save masks as pickle)
flags['lifetime_exposure_cohort'] = 0       # 0: do not process ISIMIP runs to compute exposure across cohorts (i.e. load exposure pickle)
                                            # 1: process ISIMIP runs to compute exposure across cohorts (i.e. produce and save exposure as pickle)                            
flags['lifetime_exposure_pic'] = 0   # 0: do not process ISIMIP runs to compute picontrol exposure (i.e. load exposure pickle)
                                     # 1: process ISIMIP runs to compute picontrol exposure (i.e. produce and save exposure as pickle)
flags['emergence'] = 0      # 0: do not process ISIMIP runs to compute cohort emergence (i.e. load cohort exposure pickle)
                            # 1: process ISIMIP runs to compute cohort emergence (i.e. produce and save exposure as pickle)
flags['birthyear_emergence'] = 0    # 0: only run calc_birthyear_align with birth years from 1960-2020
                                    # 1: run calc_birthyear_align with birth years from 1960-2100                             
flags['gridscale'] = 0     # 0: do not process grid scale analysis, load pickles
                            # 1: process grid scale analysis
flags['gridscale_le_test'] = 0      # 0: do not process the grid scale analysis testing diff versions of constant life expectancy
                                    # 1: process grid scale analysis testing diff versions of constant life expectancy                             
flags['gridscale_country_subset'] = 0      # 0: run gridscale analysis on all countries
                                           # 1: run gridscale analysis on subset of countries determined in "get_gridscale_regions" 
flags['global_emergence_recollect'] = 0        # 0: do not process or load pickles of global emergence masks
                                    # 1: process or load pickles if they're present (note that pickles are huge on hydra)
flags['pic_quantiles'] = 0          # 0: do not load sensitivity tests of pic lifetime exposure for a range of percentiles as thresholds for ULE (only ran for heatwaves)
                                    # 1: load that^ (not a lot of memory)                                    
flags['global_avg_emergence'] = 0                                                                                                
flags['gdp_deprivation'] = 0        # 0: do not process/load lifetime GDP/GRDI average
                                    # 1: load lifetime GDP average analysis        
flags['vulnerability'] = 0          # 0: do not process subsets of d_collect_emergence vs gdp & deprivation quantiles
                                    # 1: process/load d_collect_emergence vs gdp & deprivation quantiles for vulnerability analysis
flags['plot_ms'] = 0 # 1 yes plot, 0 no plot
flags['plot_si'] = 0
flags['reporting'] = 0  
flags['testing'] = 0   
flags['website'] = 1


#%% ----------------------------------------------------------------
# settings
# ----------------------------------------------------------------

from settings import *
ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins = init()

# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
set_extremes(flags)

#%% ----------------------------------------------------------------
# load and manipulate demographic, GMT and ISIMIP data
# ----------------------------------------------------------------

from load_manip import *

# --------------------------------------------------------------------
# Load global mean temperature projections
global df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj

df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_strj = load_GMT(
    year_start,
    year_end,
    year_range,
    flags,
)

# --------------------------------------------------------------------
# Load and manipulate life expectancy, cohort and mortality data

if flags['mask']: # load data and do calculations

    print('Processing country info')

    d_countries = all_country_data(flags)

else: # load processed country data

    print('Loading processed country and region data')

    # load country pickle
    d_countries = pk.load(open('./data/{}/country_info.pkl'.format(flags['version']), 'rb'))
    
# unpack country information
df_countries = d_countries['info_pop']
gdf_country_borders = d_countries['borders']
da_population = d_countries['population_map']
df_birthyears = d_countries['birth_years']
df_life_expectancy_5 = d_countries['life_expectancy_5']
da_cohort_size = d_countries['cohort_size']
countries_regions, countries_mask = d_countries['mask']    
 
# --------------------------------------------------------------------
# load ISIMIP model data
global grid_area
grid_area = xr.open_dataarray('./data/isimip/clm45_area.nc4')

print('running load_isimip')
d_isimip_meta,d_pic_meta = load_isimip(
    extremes,
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_strj,
    flags,
)

sims_per_step = {}
for step in GMT_labels:
    sims_per_step[step] = []
    for i in list(d_isimip_meta.keys()):
        if d_isimip_meta[i]['GMT_strj_valid'][step]:
            sims_per_step[step].append(i)    
                 
#%% ----------------------------------------------------------------
# grid scale emergence
# ------------------------------------------------------------------

from gridscale import *

# list of countries to run gridscale analysis on (sometimes doing subsets across basiss/regions in floods/droughts)
gridscale_countries = get_gridscale_regions(
    grid_area,
    flags,
    gdf_country_borders,
)   

# birth year aligned cohort sizes for gridscale analysis (summed over lat/lon per country)
if not os.path.isfile('./data/{}/gs_cohort_sizes.pkl'.format(flags['version'])):

    print('getting da_gs_popdenom')
    da_gs_popdenom = get_gridscale_popdenom(
        gridscale_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        da_population,
        df_life_expectancy_5,
    )

    # pickle birth year aligned cohort sizes for gridscale analysis (summed per country)
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'wb') as f:
        pk.dump(da_gs_popdenom,f)  
        
else:
    
    # load pickle birth year aligned cohort sizes for gridscale analysis (summed per country, i.e. not lat/lon explicit)
    print('loading da_gs_popdenom')
    with open('./data/{}/gs_cohort_sizes.pkl'.format(flags['version']), 'rb') as f:
        da_gs_popdenom = pk.load(f)               

# run gridscale emergence analysis
if flags['gridscale']:
    
    print('calculating emergence')
    ds_pf_gs = gridscale_emergence(
        d_isimip_meta,
        d_pic_meta,
        flags,
        gridscale_countries,
        da_cohort_size,
        countries_regions,
        countries_mask,
        df_life_expectancy_5,
        da_population,
    )    
    
else:
    
    # # load pickled aggregated pop frac datasets
    print('loading emergence')
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
        ds_pf_gs = pk.load(f)
        
if flags['gridscale_le_test']:
    
    ds_pf_gs_le_test = gridscale_emergence_life_expectancy_constant(
        d_isimip_meta,
        d_pic_meta,
        flags,
        gridscale_countries,
        da_cohort_size,
        countries_regions,
        countries_mask,
        df_life_expectancy_5,
        da_population,
    )        
    
else:
    
    with open('./data/{}/{}/gridscale_aggregated_pop_frac_le_test_{}.pkl'.format(flags['version'],flags['extr']+'_le_test',flags['extr']), 'rb') as f:
        ds_pf_gs_le_test = pk.load(f)    

# read in all global emergence masks (d_global_emergence is then used for vulnerability assessment, but only possible on hpc because it is large for some hazards)
if flags['global_emergence_recollect']:

    # temporarily commented out extremes in this function outside heatwaved area to test new means extraction below
    d_global_emergence = collect_global_emergence(
        grid_area,
        flags,
        countries_mask,
        countries_regions,
        gridscale_countries,
        df_GMT_strj,
    )
    
if flags['pic_quantiles']:
    
    # temporarily commented out extremes in this function outside heatwaved area to test new means extraction below
    d_global_pic_qntls = collect_pic_qntls(
        grid_area,
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )  
    
    d_global_pic_qntls_extra = collect_pic_qntls_extra(
        grid_area,
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )    
    

if flags['global_avg_emergence']:
      
    # run averaging on d_global_emergence to produce SI figure of emergence fractions
    ds_emergence_mean = get_mean_emergence(
        df_GMT_strj,
        flags,
        da_population,
        d_global_emergence,
    )    
    
# load/proc GDP and deprivation data
if flags['gdp_deprivation']:
    
    ds_gdp, ds_grdi = load_gdp_deprivation(
        flags,
        grid_area,
        da_population,
        countries_mask,
        countries_regions,
        gridscale_countries,
        df_life_expectancy_5,
    )
    
# vulnerability subsetting
if flags['vulnerability']:  
    
    print('running vulnerability analysis')

    # get spatially explicit cohort sizes for all birth years in analysis
    da_cohort_size_1960_2020 = get_spatially_explicit_cohorts_1960_2020(
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
        da_cohort_size,
        da_population,
    )
    
    # adds data arrays to ds_gdp and ds_grdi with ranked vulnerability binned by population (i.e. ranges of ranked vulnerability, physically distributed, grouped/binned by population size)            
    ds_gdp_qntls, ds_grdi_qntls = get_vulnerability_quantiles(
        flags,
        grid_area,
        da_cohort_size_1960_2020,
        ds_gdp,
        ds_grdi,
    )
        
    # just a dummy d_global_emergence to run emergence_by_vulnerability
    try:
        d_global_emergence
    except NameError:
        print('to save memory on my laptop, d_global_emergence is not unpickled. defining a dummy var for emergence_by_vulnerability')
        d_global_emergence={}
    else:
        pass

    # dataset of emergence numbers selected by quantiles of vulnerability, both with grdi and gdp
    ds_vulnerability = emergence_by_vulnerability(
        flags,
        df_GMT_strj,
        ds_gdp_qntls,
        ds_grdi_qntls,
        da_cohort_size_1960_2020,
        d_global_emergence,
    )


#%% ----------------------------------------------------------------
# main text plots
# ------------------------------------------------------------------       

if flags['plot_ms']:

    from plot_ms import *

    # f1 of ms, conceptual figure of city grid cell
    plot_conceptual(
        da_cohort_size,
        countries_mask,
        countries_regions,
        d_isimip_meta,
        flags,
        df_life_expectancy_5,
    )
    
    # f2 alternative with absolute pops below box plots and no pie charts
    # further, returning robinson boundaries for use in pyramid plot maps for consistent map extents (that exclude antarctica)
    gdf_robinson_bounds = plot_combined_population(
        df_GMT_strj,
        ds_pf_gs,
        da_gs_popdenom,
        gdf_country_borders,
        sims_per_step,
        flags,
        df_countries,
    )        

    # f3 of heatmaps across all hazards
    plot_heatmaps_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )

    # f4 pyramid plotting
    pyramid_combined(
        ds_grdi_qntls,
        ds_gdp_qntls,
        da_cohort_size_1960_2020,
        gdf_robinson_bounds,
        df_GMT_strj,
        flags,
    )

    
#%% ----------------------------------------------------------------
# supplementary text plots
# ------------------------------------------------------------------  

if flags['plot_si']:

    from plot_si import *
    
    # heatmaps but with simulations limited to common sims (to avoid dry GCM jumps)
    plot_sf1_heatmaps_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )    
    
    # pf box plots for 1.5, 2.5 and 3.5 degree world across birth years
    plot_sf2_boxplots_allhazards(
        da_gs_popdenom,
        df_GMT_strj,
        flags,
    )      
    
    # pf time series for 2.7 degree world across birth years
    plot_sf3_pf_by_tseries_allhazards(
        flags,
        df_GMT_strj,
        da_gs_popdenom,
    )          
    
    # pf maps for 1..5, 2.5, 3.5 for all hazards
    plot_sf4_pf_maps_allhazards(
        da_gs_popdenom,
        gdf_country_borders,
        flags,
    )        
    
    # emergence fraction plot for hazards between 1960 and 2020 in a 2.7 degree world
    plot_sf5_emergence_fracs(
        grid_area,
        ds_emergence_mean,
    )        
    
    # plot locations where exposure occurs at all in our dataset
    plot_sf6_exposure_locations(
        grid_area,
        countries_mask,
        flags,
    )        
    
    # plot heatmaps of pf for country level emergence
    plot_sf7_heatmaps_allhazards_countryemergence(
        df_GMT_strj,
        flags,
    )     
    
    # plot gmt time series for projections (rcp) and for which we map projections onto (ar6)
    plot_sf8_gmt_pathways(
        df_GMT_strj,
        d_isimip_meta,
    )    
    
    # plot gmt time series for projections (rcp) and for which we map projections onto (ar6) for powerpoint
    plot_sfX_gmt_mapping(
        df_GMT_strj,
        d_isimip_meta,
    )    

    # pf time series for 2020 birth year across GMTs
    plot_pf_gmt_tseries_allhazards(
        df_GMT_strj,
        da_gs_popdenom,
        flags,
    )
    
    # plot tseries box plots for 1.5, 2.5 and 3.5 when denominator contrained by exposure extent
    plot_geoconstrained_boxplots(
        flags,
    )    
    
    # plot pie charts of all hazards
    plot_allhazards_piecharts(
        da_gs_popdenom,
        df_countries,
        flags,
    )
    
    # plot cohort sizes in stacked bar chart
    plot_cohort_sizes(
        df_countries,
        da_gs_popdenom,
    )    
    
    # plot hexagon landfracs (will change to only show landfracs for SI)
    plot_hexagon_landfrac(
        d_global_emergence,
    )    
    
    # plot heatmaps of delta CF between main text f3 (heatwavedarea panel) and 
    plot_life_expectancy_testing(
        df_GMT_strj,
        GMT_indices_plot,
        da_gs_popdenom,
        flags,
    )    
    
    # plot pic threshold sensitivity analysis results for heatwavedarea as box plots (only in response letter as of August 26)
    plot_pic_sensitivity_test(
        flags,
        d_global_pic_qntls,
        d_global_pic_qntls_extra,
    )
    
    # plot maps of population distribution per gdp quantile  (only in response letter as of August 26)
    population_per_gdp_quantile(
        da_cohort_size_1960_2020,
        ds_gdp_qntls,
        gdf_robinson_bounds
    )    
    
    # same as above but for grdi
    population_per_grdi_quantile(
        da_cohort_size_1960_2020,
        ds_grdi_qntls,
        gdf_robinson_bounds
    )        

#%% ----------------------------------------------------------------
# sample analytics for paper
# ------------------------------------------------------------------

if flags['reporting']:
    
    print('running reporting functions')
    
    from reporting import *
    
    # estimates of land area and (potential) pf for 1960 and 2020 emergencve of multiple hazards
    multi_hazard_emergence(
        grid_area,
        da_emergence_mean,
        da_gs_popdenom,
    )
    
    # get birth year cohort sizes at grid scale
    gridscale_cohort_sizes(
        flags,
        da_population,
        gridscale_countries,   
    )    
    
    # per hazard, locations where exposure occurs across whole ensemble
    exposure_locs(
        grid_area,
    )
    
    # per run for 1.5, 2.5, 2.7 and 3.5, collect maps of emergence locations to be used in geographically constrained pf estimates
    emergence_locs_perrun(
        flags,
        grid_area,
        gridscale_countries,
        countries_mask,
        countries_regions,
    )    
    
    # compute geographically constrained pf
    pf_geoconstrained()
    
    # print geographically constrained pf vs regular pf
    print_pf_geoconstrained(
        flags,
        da_gs_popdenom,
    )    

    # checking for signifiance of change in means between 1960 and 2020 pf per event and for a GMT level
    paired_ttest(
        flags,
        da_gs_popdenom,
    )
    
    # print latex table on ensemble members per hazard
    print_latex_table_ensemble_sizes(
        flags,
        df_GMT_strj,
    )   
    
    # children (i.e. those born between 2003-2020) living unprec exposure between 1.5 and 2.7 degrees warming (for numbers in conclusion of paper)
    print_millions_excess(
        flags,
        df_GMT_strj,
    )     

    # print pf info    
    print_pf_ratios_and_abstract_numbers(
        df_GMT_strj,
        da_gs_popdenom,
    )    
    
    # get number of million people unprecedented: (will change this stuff to run for all extremes and birth years for table in paper)
    print_absolute_unprecedented(
        ds_pf_gs,
    )
 
    # get cities that work for 
    find_valid_cities(
        df_countries,
        da_cohort_size,
        countries_mask,
        countries_regions,
        d_isimip_meta,
        flags,
    )
    
    # latex tables of CF per extr and GMT pathway
    print_latex_table_unprecedented(
        flags,
        da_gs_popdenom,
    )    
    
    print_latex_table_unprecedented_sideways(
        flags,
        da_gs_popdenom,
    )    
    
    # data for box plots of heatwaves (f2)
    print_f2_info(
        ds_pf_gs,
        flags,
        df_GMT_strj,
        da_gs_popdenom,
        gdf_country_borders,
    )
    
    # data for f3
    print_f3_info(
        flags,
        da_gs_popdenom
    )
    
    # data for pyramid stuff (f4)
    print_pyramid_info(
        flags,
    )
    
                
# data for website
if flags['website']:
    
    from reporting import website_exposure_means, ds_e_to_excel, website_pic_threshold
    
    # get annual lat- and pop-weighted means of exposure across GMT, country and time (no cummulative summing, no birth-year life expectancy integration)
    ds_e,ds_ce = website_exposure_means(
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
    )
    
    # export to excel
    ds_e_to_excel(
        flags,
        ds_e,
        df_GMT_strj,
    )
    
    # # export to nc
    # ds_e_to_nc(
    #     flags,
    #     ds_e,
    #     df_GMT_strj,
    # )    
    
    # get pic thresholds per country
    website_pic_threshold(
        flags,
        gridscale_countries,
        countries_mask,
        countries_regions,
        da_population,
        d_pic_meta,
    )
                    