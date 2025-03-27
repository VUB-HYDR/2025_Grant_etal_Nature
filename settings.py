# ----------------------------------------------------------------
# Settings
# These are global variables to be used throughout the whole project
# ----------------------------------------------------------------

import numpy as np

#%% ----------------------------------------------------------------
def init(): 

    # initialise age and associated time period of interest
    global ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_inc
    ages = np.arange(60,-1,-1)
    age_young = 0
    age_ref = np.nanmax(ages)
    age_range = np.arange(0,105)
    year_ref = 2020
    year_start = year_ref - age_ref
    birth_years = np.arange(year_start,year_ref+1)     
    year_end = 2113 # based on maximum life expectancy reported in UN WPP
    year_range = np.arange(year_start,year_end+1)
    
    # PIC sampling information
    global pic_life_extent, nboots, resample_dim, pic_by, pic_qntl
    pic_life_extent=82 # +1 from max 1960 life expectancy
    nboots=10000 # number of bootstrapped lifetimes
    resample_dim='time' # for bootstrapping lifetimes, sample over time
    pic_by=1960 # use 1960 birth year demography data for pic lifetimes
    pic_qntl=0.9999 # quantile for pic extreme threshold
    # pic_qntl_list = [pic_qntl,0.999,0.99,0.975,0.95,0.9]
    pic_qntl_list = [pic_qntl]
    # pic_qntl_labels = ['99.99', '99.9', '99.0', '97.5', '95.0', '90.0']
    pic_qntl_labels = ['99.99']

    # initialise age groups
    # (https://www.carbonbrief.org/analysis-why-children-must-emit-eight-times-less-co2-than-their-grandparents)
    # (https://www.pewresearch.org/fact-tank/2019/01/17/where-millennials-end-and-generation-z-begins/)
    global agegroups
    agegroups = {
        'Boomers' : (1950, 1965),
        'Gen X' : (1965, 1981),
        'Millenials' : (1981, 1997),
        'Gen Z' : (1997, 2020)
    }


    # initialise reference period for computing GMT anomalies
    global year_start_GMT_ref, year_end_GMT_ref
    year_start_GMT_ref = 1850
    year_end_GMT_ref = 1900

    global extremes_legend_dict
    extremes_legend_dict = {
        'burntarea' : 'Wildfires',
        'cropfailedarea' : 'Crop failures', 
        'driedarea' : 'Droughts', 
        'floodedarea' : 'River floods',
        'heatwavedarea' : 'Heatwaves',
        'tropicalcyclonedarea' : 'Tropical cyclones',
        'all' : 'All'               
    }
    
    
    # initialise model names
    global model_names
    model_names = {
        'burntarea' : [
            'CARAIB', 
            'LPJ-GUESS',
            'LPJmL',
            'ORCHIDEE',
            'VISIT'], 
        'cropfailedarea' : [
            'GEPIC', 
            'LPJmL',
            'PEPIC'],
        'driedarea' : [
            'CLM45',
            'H08', 
            'LPJmL', 
            'JULES-W1', 
            'MPI-HM', 
            'ORCHIDEE', 
            'PCR-GLOBWB', 
            'WaterGAP2'],
        'floodedarea' : [
            'CLM45', 
            'H08', 
            'LPJmL', 
            'JULES-W1', 
            'MPI-HM', 
            'ORCHIDEE', 
            'PCR-GLOBWB', 
            'WaterGAP2'],
        'heatwavedarea' : [
            'HWMId99'],        
        # 'heatwavedarea' : [
        #     'HWMId-humidex', 
        #     'HWMId99-humidex40', 
        #     'HWMId97p5-humidex40', 
        #     'HWMId99-tasmax35', 
        #     'HWMId97p5-tasmax35', 
        #     'HWMId99', 'HWMId97p5', 
        #     'humidex40d3', 
        #     'humidex40d5', 
        #     'humidex45d3', 
        #     'humidex45d5', 
        #     'CWMId99'],
        'tropicalcyclonedarea' : ['KE-TG-meanfield']
    }


    # Set threshold maximum T difference between RCP and GMT trajectories
    # i.e. any run with T difference exceeding this threshold is excluded
    # year-to-year jumps in GMT larger than 0.1, so using a 0.1 maxdiff threshold erronously removes runs
    # used to be 0.5, but then rcp2.6 is used for high-warming levels
    # Anything between 0.1 and 0.2 removes RCP2.6 in NDC scenarios (see histograms of maxdiff_NDC)
    # take 0.2 to have more data in BE scenarios and hence smooth EMF curves in BE plot
    global RCP2GMT_maxdiff_threshold
    RCP2GMT_maxdiff_threshold = 0.2 # [K]
    
    # set GMT info for stylized trajectories
    GMT_max = 3.5 # this gets overwritten by the end of GMT_40 (but this probably changes with new approach for clean intervals between just 1.5 and 3.5 that we will use on rerun)
    GMT_min = 1.5
    GMT_inc = 0.1
    scen_thresholds = {
        '3.0': [2.9,3.0],
        'NDC': [2.35,2.4],
        '2.0': [1.95,2.0],
        '1.5': [1.45, 1.5],
    }
    # GMT_labels = np.arange(0,29).astype('int')
    GMT_labels = np.arange(0,21).astype('int')
    GMT_window = 21
    GMT_current_policies = [12,17] # this is for 2.7 and 3.2 degrees warming targets, unknown which is best to use

    # set kernel x-values
    global kernel_x
    kernel_x = np.arange(1,50.5,0.5)
    
    # grid scale variables
    global sample_birth_years
    sample_birth_years = np.arange(1960,2021,10) # for grid scale spatial datasets  
    sample_countries = [
        'Canada',
        'United States',
        'China',
        'Russian Federation'
    ]
    # GMT_indices_plot = [0,10,19,28]
    # new GMT_indices_plot for box plots showing 1.5, 2.5 and 3.5
    # GMT_indices_plot = [6,15,24]c commented out because I cleaned up GMTs
    GMT_indices_plot = [0,10,20]
    birth_years_plot = np.arange(1960,2021,20)   
    
    # basins for flood trends
    basins=['Mississippi','Zaire','Nile','Amazon','Danube','Parana','Niger','Ganges-Brahmaputra','Ob','St. Lawrence','Yangtze']
    
    # plotting vars
    letters = ['a', 'b', 'c',
           'd', 'e', 'f',
           'g', 'h', 'i',
           'j', 'k', 'l',
           'm', 'n', 'o',
           'p', 'q', 'r',
           's', 't', 'u',
           'v', 'w', 'x',
           'y', 'z']
    
    return ages, age_young, age_ref, age_range, year_ref, year_start, birth_years, year_end, year_range, GMT_max, GMT_min, GMT_inc, RCP2GMT_maxdiff_threshold, year_start_GMT_ref, year_end_GMT_ref, scen_thresholds, GMT_labels, GMT_window, GMT_current_policies, pic_life_extent, nboots, resample_dim, pic_by, pic_qntl, pic_qntl_list, pic_qntl_labels, sample_birth_years, sample_countries, GMT_indices_plot, birth_years_plot, letters, basins

#%% ----------------------------------------------------------------
# set extremes based on flag (this needs to happen here as it uses the flags dict defined above)
def set_extremes(flags):

    global extremes
    
    if not flags['extr'] == 'all': # processing for single extreme
        extremes = [flags['extr']]
    else: 
        extremes = [
            'burntarea', 
            'cropfailedarea', 
            'driedarea', 
            'floodedarea', 
            'heatwavedarea', 
            'tropicalcyclonedarea'
        ]
