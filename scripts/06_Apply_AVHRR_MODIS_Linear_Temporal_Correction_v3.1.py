import xarray as xr
import os
import numpy as np
import calendar
from datetime import date, timedelta
import warnings
from copy import deepcopy
import rasterio as rio

# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 1
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS
rid = 0 



# Path to read and write data
SZA_PATH = "AVHRR/data/AVHRR_SZA_CORRECTED_v3.2/"
LINEAR_PATH = "AVHRR/data/AVHRR_LINEAR_CORRECTED_v3.2/"
AVHRR_MVC_DIR = "AVHRR/data/AVHRR_MVC/"


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

AVHRR_MVC_files = []
for year in np.arange(2014, 2023):
    for month in np.arange(1, 13):
        for half_month_index in ["a", "b"]:
            AVHRR_MVC_files.append(os.path.join(AVHRR_MVC_DIR, "AVHRR_MVC_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            
AVHRR_MVC_files_valid  = [f for f in np.array(AVHRR_MVC_files) if bi in f]
avhrr_m1_ref_ma = np.full((len(AVHRR_MVC_files_valid), 3000, 7200), np.nan, dtype=np.float32)
for i in range(len(AVHRR_MVC_files_valid)):
    avhrr_mvc_ds = xr.open_dataset(AVHRR_MVC_files_valid[i])
    avhrr_m1_ref_ma[i, :, :] = avhrr_mvc_ds["{}_brdf_normalized_sza45".format(ref_var)].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]


igbp_2014=rio.open('HDF4_EOS:EOS_GRID:"../../data/MCD12C1/MCD12C1.A2014001.061.2022165213124.hdf":MOD12C1:Majority_Land_Cover_Type_1')
igbp_2014_data = igbp_2014.read()[0]
land = igbp_2014_data > 0

avhrr_ref_ma = np.load(os.path.join(SZA_PATH, "SZA_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])))
slope_ma = np.load(os.path.join(LINEAR_PATH, "slope_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])))
intercept_ma = np.load(os.path.join(LINEAR_PATH, "intercept_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])))
slope_ma[np.isnan(slope_ma) & land[0:3000, :]] = 1.0
intercept_ma[np.isnan(intercept_ma) & land[0:3000, :]] = 0.0
linear_corrected = avhrr_ref_ma * slope_ma + intercept_ma

slope_m1_ma = np.load(os.path.join(LINEAR_PATH, "m1_slope_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])))
intercept_m1_ma = np.load(os.path.join(LINEAR_PATH, "m1_intercept_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])))
slope_m1_ma[np.isnan(slope_m1_ma) & land[0:3000, :]] = 1.0
intercept_m1_ma[np.isnan(intercept_m1_ma) & land[0:3000, :]] = 0.0
linear_corrected_m1 = avhrr_m1_ref_ma * slope_m1_ma + intercept_m1_ma

linear_corrected_all = np.concatenate([linear_corrected, linear_corrected_m1])

if ref_var == "nir":
    linear_corrected_all[linear_corrected_all > 0.8] = np.nan # this will help to remove most of the high latitude bad data points in winter
np.save(os.path.join(LINEAR_PATH, "corrected_{}_{}.npy".format(ref_var, biweekly_identifiers[bid])), linear_corrected_all)
