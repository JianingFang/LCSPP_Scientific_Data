import xarray as xr
import os
import numpy as np
import calendar
from datetime import date, timedelta
import warnings
from copy import deepcopy
from sklearn import linear_model


# Path to read and write data
MVC_DIR = "AVHRR/data/AVHRR_MVC/"
ERA5_PATH = "data/ERA5/"
SZA_PATH = "AVHRR/data/AVHRR_SZA_CORRECTED_v3.2/"

# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 1
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS
rid = 0 

# read mean red and nir values averaged across calibration PICS sites.
# This is expected to take account of the per-sensor biases


# Get index for the variable and period to process
# Note SLURM_ARRAY_TASK_ID should be between 1-48
IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

if IDX > 23:
    ref_var = "red"
    bid = IDX - 24
else:
    ref_var = "nir"
    bid = IDX

time_arr = np.load("../../data/time_arr_avhrr.npy")
ref_mean = np.load("../../data/{}_pics_avhrr_scaled_cal_mean.npy".format(ref_var))
valid_period = np.load("../../data/valid_period_avhrr.npy")
time_arr_valid = time_arr[valid_period]


# Calibrate a model for each of the biweekly period 
biweekly_identifiers = []
for month in np.arange(1, 13):
    for half_month_index in ["a", "b"]:
        biweekly_identifiers.append('{:0>2}'.format(month) + half_month_index)
bi = biweekly_identifiers[bid]

# Obtain the filenames for all the input datasets
MVC_files = []
t2m_files = []
ssrd_files = []
precip_files = []
for year in np.arange(1982, 2014):
    for month in np.arange(1, 13):
        for half_month_index in ["a", "b"]:
            MVC_files.append(os.path.join(MVC_DIR, "AVHRR_MVC_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            t2m_files.append(os.path.join(ERA5_PATH, "ERA5_CMG_t2m_biweekly_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            ssrd_files.append(os.path.join(ERA5_PATH, "ERA5_CMG_ssrd_biweekly_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            precip_files.append(os.path.join(ERA5_PATH, "ERA5_CMG_precip_biweekly_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            
            
MVC_files_valid  = [f for f in np.array(MVC_files)[valid_period] if bi in f]
t2m_files_valid  = [f for f in np.array(t2m_files)[valid_period] if bi in f]
ssrd_files_valid  = [f for f in np.array(ssrd_files)[valid_period] if bi in f]
precip_files_valid  = [f for f in np.array(precip_files)[valid_period] if bi in f]


bi_list = []        
for i in range(len(MVC_files)):
    if bi in MVC_files[i] and valid_period[i]:
        bi_list.append(i)
        
bi_idx = np.array(bi_list)
           

# Allocate memory for the arrays to store values
ref_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
sza_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
ref_mean_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
t2m_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
ssrd_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
precip_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)

time_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
for t in range(len(MVC_files_valid)):
    time_ma[t, :, :] = t

# Allocate memory for the calibrated reflectance
corrected_ref_ma = np.full((len(MVC_files_valid), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)

# Read in input datasets to populate arrays
for i in range(len(MVC_files_valid)):
    mvc_ds = xr.open_dataset(MVC_files_valid[i])
    t2m = np.flip(xr.open_dataset(t2m_files_valid[i]).Band1.values, axis=0)
    ssrd = np.flip(xr.open_dataset(ssrd_files_valid[i]).Band1.values, axis=0)
    precip = np.flip(xr.open_dataset(precip_files_valid[i]).Band1.values, axis=0)
    ref_ma[i, :, :] = mvc_ds["{}_brdf_normalized_sza45".format(ref_var)].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    sza_ma[i, :, :] = mvc_ds.sza.values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    ref_mean_ma[i, :, :] = ref_mean[bi_idx[i]]
    t2m_ma[i, :, :] = t2m[rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    ssrd_ma[i, :, :] = ssrd[rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    precip_ma[i, :, :] = precip[rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    

# Now let's fit the model!
for i in range(SINGLE_HEIGHT):
    for j in range(7200):
        ref_p_valid = np.invert(np.isnan(ref_ma[:, i, j]))
        sza_p_valid = np.invert(np.isnan(sza_ma[:, i, j]))
        ref_mean_p_valid = np.invert(np.isnan(ref_mean_ma[:, i, j]))
        t2m_p_valid = np.invert(np.isnan(t2m_ma[:, i, j]))
        ssrd_p_valid = np.invert(np.isnan(ssrd_ma[:, i, j]))
        precip_p_valid = np.invert(np.isnan(precip_ma[:, i, j]))
        all_p_valid =  ref_p_valid &  sza_p_valid & ref_mean_p_valid & t2m_p_valid & ssrd_p_valid & precip_p_valid
        
        # Need to have at least 10 datapoints to do the regression
        if np.sum(all_p_valid) > 10:

            
            sza_vals = sza_ma[:, i, j][all_p_valid]
            ref_mean_vals = ref_mean_ma[:, i, j][all_p_valid] 
            t2m_vals = t2m_ma[:, i, j][all_p_valid]
            ssrd_vals = ssrd_ma[:, i, j][all_p_valid] 
            precip_vals = precip_ma[:, i, j][all_p_valid] 

            X = np.array([sza_vals, ref_mean_vals, t2m_vals, ssrd_vals, precip_vals]).T
            X = X - np.mean(X, axis=0)
            X = X / np.std(X, axis=0)

            Y = ref_ma[:, i, j][all_p_valid]
            reg = linear_model.LinearRegression()
            reg.fit(X, Y)
            # remove the effects due to SZA and PICS mean
            corrected_ref_ma[:, i, j][all_p_valid] = Y - X[:, 0] * reg.coef_[0] - X[:, 1] * reg.coef_[1]
        else:
            # If fewer than 10 data points per pixel are available, try to see if we can get 10
            # different time in the 5x5 moving window
            ref_window = ref_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
            sza_window = sza_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
            ref_mean_window = ref_mean_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
            t2m_window = t2m_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
            ssrd_window = ssrd_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
            precip_window = precip_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
            time_window = time_ma[:, max(i-2, 0):min(i+2, SINGLE_HEIGHT), max(j-2, 0):min(j+2, 7200)].flatten()
            
            ref_w_valid = np.invert(np.isnan(ref_window))
            sza_w_valid = np.invert(np.isnan(sza_window))
            ref_mean_w_valid = np.invert(np.isnan(ref_mean_window))
            t2m_w_valid = np.invert(np.isnan(t2m_window))
            ssrd_w_valid = np.invert(np.isnan(ssrd_window))
            precip_w_valid = np.invert(np.isnan(precip_window))

            all_w_valid = ref_w_valid &  sza_w_valid & ref_mean_w_valid & t2m_w_valid & ssrd_w_valid & precip_w_valid
            if len(np.unique(time_window[all_w_valid])) > 10:
                sza_vals = sza_window[all_w_valid]
                ref_mean_vals = ref_mean_window[all_w_valid]
                t2m_vals = t2m_window[all_w_valid]
                ssrd_vals = ssrd_window[all_w_valid]
                precip_vals = precip_window[all_w_valid]
                
                X = np.array([sza_vals, ref_mean_vals, t2m_vals, ssrd_vals, precip_vals]).T
                X_mean = np.mean(X, axis=0)
                X_std = np.std(X, axis=0)
                
                X = (X - X_mean) / X_std
                Y = ref_window[all_w_valid]
                reg = linear_model.LinearRegression()
                reg.fit(X, Y)
                
                sza_vals_p = sza_ma[:, i, j][all_p_valid]
                ref_mean_vals_p = ref_mean_ma[:, i, j][all_p_valid] 
                t2m_vals_p = t2m_ma[:, i, j][all_p_valid]
                ssrd_vals_p = ssrd_ma[:, i, j][all_p_valid] 
                precip_vals_p = precip_ma[:, i, j][all_p_valid] 

                X_p = np.array([sza_vals_p, ref_mean_vals_p, t2m_vals_p, ssrd_vals_p, precip_vals_p]).T
                X_p = (X_p - X_mean) / X_std
                Y_p = ref_ma[:, i, j][all_p_valid]
                
                # remove the effects due to SZA and PICS mean
                corrected_ref_ma[:, i, j][all_p_valid] = Y_p - X_p[:, 0] * reg.coef_[0] - X_p[:, 1] * reg.coef_[1]

# save calibrated array to disk
np.save(os.path.join(SZA_PATH, "SZA_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])), corrected_ref_ma)
