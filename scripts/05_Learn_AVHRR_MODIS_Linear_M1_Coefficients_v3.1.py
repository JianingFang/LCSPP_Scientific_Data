import xarray as xr
import os
import numpy as np
import calendar
from datetime import date, timedelta
import warnings
from copy import deepcopy
from scipy.stats import linregress
import sys


# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 1
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS
rid = 0 

# Path to read and write data
AVHRR_MVC_DIR = "AVHRR/data/AVHRR_MVC/"
ERA5_PATH = "data/ERA5/"
SZA_PATH = "AVHRR/data/AVHRR_SZA_CORRECTED_v3.1/"
LINEAR_PATH = "AVHRR/data/AVHRR_LINEAR_CORRECTED_v3.1/"
MODIS_MVC_DIR = "AVHRR/data/MODIS_MVC/"

IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

if IDX > 23:
    ref_var = "red"
    bid = IDX - 24
else:
    ref_var = "nir"
    bid = IDX

# Calibrate a model for each of the biweekly period 
biweekly_identifiers = []
for month in np.arange(1, 13):
    for half_month_index in ["a", "b"]:
        biweekly_identifiers.append('{:0>2}'.format(month) + half_month_index)
bi = biweekly_identifiers[bid]

# Obtain the filenames for all the input datasets
MODIS_MVC_files = []
AVHRR_MVC_files = []


for year in np.arange(2014, 2023):
    for month in np.arange(1, 13):
        for half_month_index in ["a", "b"]:
            MODIS_MVC_files.append(os.path.join(MODIS_MVC_DIR, "MODIS_MVC_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            AVHRR_MVC_files.append(os.path.join(AVHRR_MVC_DIR, "AVHRR_MVC_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            
MODIS_MVC_files_valid  = [f for f in np.array(MODIS_MVC_files) if bi in f]
AVHRR_MVC_files_valid  = [f for f in np.array(AVHRR_MVC_files) if bi in f]

time_ma = np.full((len(MODIS_MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
for t in range(len(MODIS_MVC_files_valid)):
    time_ma[t, :, :] = t        
    

avhrr_ref_ma = np.full((len(AVHRR_MVC_files_valid), 3000, 7200), np.nan, dtype=np.float32)
modis_ref_ma = np.full((len(MODIS_MVC_files_valid), 3000, 7200), np.nan, dtype=np.float32)

modis_slope_ma = np.full(avhrr_ref_ma.shape[1:], np.nan, dtype=np.float32)
modis_intercept_ma = np.full(avhrr_ref_ma.shape[1:], np.nan, dtype=np.float32)
modis_r2_ma = np.full(avhrr_ref_ma.shape[1:], np.nan, dtype=np.float32)
    
for i in range(len(AVHRR_MVC_files_valid)):
    avhrr_mvc_ds = xr.open_dataset(AVHRR_MVC_files_valid[i])
    avhrr_ref_ma[i, :, :] = avhrr_mvc_ds["{}_brdf_normalized_sza45".format(ref_var)].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    modis_mvc_ds = xr.open_dataset(MODIS_MVC_files_valid[i])
    modis_ref_ma[i, :, :] = modis_mvc_ds["{}_brdf_normalized_sza45".format(ref_var)].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    # added snow percent mask here, Oct 9 2023
    modis_ref_ma[i, :, :][modis_mvc_ds["percent_snow"].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :] > 0] = np.nan

    
    
for i in range(SINGLE_HEIGHT):
    for j in range(7200):
        
        ref_avhrr_window = avhrr_ref_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
        ref_modis_window = modis_ref_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
        
        if np.sum(np.invert(np.isnan(ref_avhrr_window)) & np.invert(np.isnan(ref_avhrr_window))) > 8:
        
            time_window = time_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()

            ref_avhrr99 = np.nanpercentile(ref_avhrr_window, 99)
            ref_avhrr01 = np.nanpercentile(ref_avhrr_window, 1)
            ref_avhrr_window[(ref_avhrr_window > ref_avhrr99) | (ref_avhrr_window < ref_avhrr01)] = np.nan

            ref_modis99 = np.nanpercentile(ref_modis_window, 99)
            ref_modis01 = np.nanpercentile(ref_modis_window, 1)
            ref_modis_window[(ref_modis_window > ref_modis99) | (ref_modis_window < ref_modis01)] = np.nan

            ref_avhrr_w_valid = np.invert(np.isnan(ref_avhrr_window))
            ref_modis_w_valid = np.invert(np.isnan(ref_modis_window))

            all_w_valid = ref_avhrr_w_valid & ref_modis_w_valid
            if len(np.unique(time_window[all_w_valid])) > 5:
                ref_avhrr_vals = ref_avhrr_window[all_w_valid]
                ref_modis_vals = ref_modis_window[all_w_valid]
                if len(np.unique(ref_avhrr_vals)) > 5:
                    slope, intercept, r_value, p_value, std_err = linregress(ref_avhrr_vals,ref_modis_vals)
                    modis_slope_ma[i, j] = slope
                    modis_intercept_ma[i, j] = intercept
                    modis_r2_ma[i, j] = r_value**2
np.save(os.path.join(LINEAR_PATH, "snow_m1_slope_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])), modis_slope_ma)
np.save(os.path.join(LINEAR_PATH, "snow_m1_intercept_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])), modis_intercept_ma)
np.save(os.path.join(LINEAR_PATH, "snow_m1_r2_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])), modis_r2_ma)
