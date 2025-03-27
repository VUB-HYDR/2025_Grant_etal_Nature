# 2025_Grant_etal_Nature

Python code for lifetime exposure analysis with ISIMIP simulations for "Global emergence of unprecedented lifetime exposure to climate extremes". This was originally translated from the MATLAB code of [Thiery et al. (2021)](https://github.com/VUB-HYDR/2021_Thiery_etal_Science) and extended toward assessing gridscale, unprecedented exposure.


## Environment
The python modules used in this repository can be installed using [exposure_env.yml](./exposure_env.yml). This may take up to an hour to compile in the Anaconda prompt. If this doesn't work, you will need to manually collate imported modules in the .py scripts and generate your own env with this list. Try using Python 3.9 for this.

```
conda env create -f exposure_env.yml

```

## Sample data
Sample data for our analysis for heatwaves is available [here](https://zenodo.org/records/15097896). You need to copy/unzip the "data" folder and its contents to the same directory as the code for the repo to work.

## Running
Once your python environment is set up, running this analysis for heatwaves should take 3-6 hours. Simply choose the "heatwavedarea" flag and set all "flags" run options except for "lifetime_exposure_cohort", "lifetime_exposure_pic", "emergence" and "birthyear_emergence" to full compute (1). This will produce a number of python pickle files for intermediate computations and final results. Final results are mostly present as xarray datasets/dataarrays. Note that some plotting functions will not work, as they require outputs of analyses for other extreme event categories for which sample data is not provided.

## License
This project is licensed under the MIT License. See also the 
[LICENSE](LICENSE) 
file.



## References
Thiery, W., Lange, S., Rogelj, J., Schleussner, C. F., Gudmundsson, L., Seneviratne, S. I., Andrijevic, M., Frieler, K., Emanuel, K., Geiger, T., Bresch, D. N., Zhao, F., Willner, S. N., Büchner, M., Volkholz, J., Bauer, N., Chang, J., Ciais, P., Dury, M., … Wada, Y. (2021). Intergenerational inequities in exposure to climate extremes. Science, 374(6564), 158–160. https://doi.org/10.1126/science.abi7339

