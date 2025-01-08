import xarray as xr
import numpy as np
import os
import rioxarray as rxr
from tqdm import tqdm
import calendar
from datetime import date

IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

if IDX > 23:
    ref_var = "red"
    other_ref_var = "nir"
    bid = IDX - 24
else:
    ref_var = "nir"
    other_ref_var = "red"
    bid = IDX
    
# Calibrate a model for each of the biweekly period 
biweekly_identifiers = []
for month in np.arange(1, 13):
    for half_month_index in ["a", "b"]:
        biweekly_identifiers.append('{:0>2}'.format(month) + half_month_index)
bi = biweekly_identifiers[bid]

DEM = np.flip(xr.open_dataset(os.path.join("AVHRR/data/", "DEM_CMG.nc")).Band1, axis=0)
DEM.values[DEM.values < -414] = -414 # remove abberant values during interpolation

AI = np.flip(xr.open_dataset(os.path.join("AVHRR/data/", "AI_CMG.nc")).Band1, axis=0)
AI.values[AI.values < 0.001] = 0.001 # remove abberant values during interpolation
AI.values[AI.values > 10] = 10 # remove abberant values during interpolation
LOG_AI = np.log(AI)

LINEAR_PATH = "AVHRR/data/AVHRR_LINEAR_CORRECTED_v3.2/"
MODIS_MVC_DIR = "AVHRR/data/MODIS_MVC/"
MERRA2_DIR = "../../data/MERRA2/"
ERA5_PATH = "data/ERA5/"
ML_PATH = "AVHRR/data/ML_CORRECT_v3.2/"

# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 1
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS
rid = 0 

# Obtain the filenames for all the input datasets
MODIS_MVC_files = []

for year in np.arange(2001, 2024):
    mr = np.arange(1, 13)
    for month in mr:
        for half_month_index in ["a", "b"]:
            MODIS_MVC_files.append(os.path.join(MODIS_MVC_DIR, "MODIS_MVC_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
MVC_files_valid  = [f for f in np.array(MODIS_MVC_files) if bi in f]

avhrr_fn = [f for f in sorted(os.listdir(LINEAR_PATH)) if "corrected_{}_{}".format(ref_var, bi) in f][0]
avhrr_other_fn = [f for f in sorted(os.listdir(LINEAR_PATH)) if "corrected_{}_{}".format(other_ref_var, bi) in f][0]


test_years = np.arange(1, len(MVC_files_valid), 8)
train_years = np.array([n for n in np.arange(len(MVC_files_valid)) if n not in test_years])


avhrr_ref_ma = np.load(os.path.join(os.path.join(LINEAR_PATH, avhrr_fn)))
avhrr_ref_ma = avhrr_ref_ma[-len(MVC_files_valid):, :, :]

avhrr_other_ref_ma = np.load(os.path.join(os.path.join(LINEAR_PATH, avhrr_other_fn)))
avhrr_other_ref_ma = avhrr_other_ref_ma[-len(MVC_files_valid):, :, :]
avhrr_ref_ma[np.isnan(avhrr_other_ref_ma)] = np.nan

modis_ref_ma = np.full(avhrr_ref_ma.shape, np.nan, dtype=np.float32)

for i in range(len(MVC_files_valid)):
    mvc_ds = xr.open_dataset(MVC_files_valid[i])
    modis_ref_ma[i, :, :] = mvc_ds["{}_brdf_normalized_sza45".format(ref_var)].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
    # added snow percent mask here, Oct 9 2023
    modis_ref_ma[i, :, :][mvc_ds["percent_snow"].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :] > 0] = np.nan
    
avhrr_ref_ma_train = avhrr_ref_ma[train_years, :, :]
avhrr_ref_ma_test = avhrr_ref_ma[test_years, :, :]
avhrr_other_ref_ma_train = avhrr_other_ref_ma[train_years, :, :]
avhrr_other_ref_ma_test = avhrr_other_ref_ma[test_years, :, :]

dif_ref_ma = modis_ref_ma - avhrr_ref_ma

dif_ref_ma_train = dif_ref_ma[train_years, :, :]
dif_ref_ma_test = dif_ref_ma[test_years, :, :]

valid_ma_train = np.invert(np.isnan(dif_ref_ma_train))
valid_count_train = np.sum(valid_ma_train)

valid_ma_test = np.invert(np.isnan(dif_ref_ma_test))
valid_count_test = np.sum(valid_ma_test)

rand_select_train = np.random.rand(np.int32(valid_count_train)) < 0.4
rand_select_test = np.random.rand(np.int32(valid_count_test)) < 0.4

selected_dif_train = dif_ref_ma_train[valid_ma_train][rand_select_train]
selected_dif_test = dif_ref_ma_test[valid_ma_test][rand_select_test]

selected_avhrr_ref_train = avhrr_ref_ma_train[valid_ma_train][rand_select_train]
selected_avhrr_ref_test = avhrr_ref_ma_test[valid_ma_test][rand_select_test]

selected_avhrr_other_ref_train = avhrr_other_ref_ma_train[valid_ma_train][rand_select_train]
selected_avhrr_other_ref_test = avhrr_other_ref_ma_test[valid_ma_test][rand_select_test]

LOG_AI_ma_train = np.repeat(np.expand_dims(LOG_AI, axis=0), len(train_years), axis=0)
LOG_AI_ma_test = np.repeat(np.expand_dims(LOG_AI, axis=0), len(test_years), axis=0)
selected_logai_train = LOG_AI_ma_train[valid_ma_train][rand_select_train]
selected_logai_test = LOG_AI_ma_test[valid_ma_test][rand_select_test]

DEM_ma_train = np.repeat(np.expand_dims(DEM, axis=0), len(train_years), axis=0)
DEM_ma_test = np.repeat(np.expand_dims(DEM, axis=0), len(test_years), axis=0)

selected_dem_train = DEM_ma_train[valid_ma_train][rand_select_train]
selected_dem_test = DEM_ma_test[valid_ma_test][rand_select_test]

AOD_valid = np.array([f for f in sorted(os.listdir(os.path.join(MERRA2_DIR, "processed"))) if "CMG" in f and bi in f])
AOD_valid=AOD_valid[-len(MVC_files_valid):]

aod_ma = np.full(avhrr_ref_ma.shape, np.nan, dtype=np.float32)

for i in range(len(AOD_valid)):
    aod_ds = xr.open_dataset(os.path.join(MERRA2_DIR, "processed", AOD_valid[i]))
    aod_ma[i, :, :] = np.flip(aod_ds.Band1.values, axis=0)
    
aod_ma_train = aod_ma[train_years, :, :]
aod_ma_test = aod_ma[test_years, :, :]
selected_aod_train = aod_ma_train[valid_ma_train][rand_select_train]
selected_aod_test = aod_ma_test[valid_ma_test][rand_select_test]

cld_valid = np.array([f for f in sorted(os.listdir(ERA5_PATH)) if "_true_CMG_biweekly_cloud_cover_" in f and bi in f])
cld_valid=cld_valid[-len(MVC_files_valid):]

cld_ma = np.full(avhrr_ref_ma.shape, np.nan, dtype=np.float32)

for i in range(len(cld_valid)):
    cld_ds = xr.open_dataset(os.path.join(ERA5_PATH, cld_valid[i]))
    cld_ma[i, :, :] = np.flip(cld_ds.Band1.values, axis=0)
    
cld_ma_train = cld_ma[train_years, :, :]
cld_ma_test = cld_ma[test_years, :, :]
selected_cld_train = cld_ma_train[valid_ma_train][rand_select_train]
selected_cld_test = cld_ma_test[valid_ma_test][rand_select_test]

sd_valid = np.array([f for f in sorted(os.listdir(ERA5_PATH)) if "_true_CMG_biweekly_depth_" in f and bi in f])
sd_valid=sd_valid[-len(MVC_files_valid):]

sd_ma = np.full(avhrr_ref_ma.shape, np.nan, dtype=np.float32)

for i in range(len(sd_valid)):
    sd_ds = xr.open_dataset(os.path.join(ERA5_PATH, sd_valid[i]))
    sd_ma[i, :, :] = np.flip(sd_ds.Band1.values, axis=0)
    
sd_ma_train = sd_ma[train_years, :, :]
sd_ma_test = sd_ma[test_years, :, :]
selected_sd_train = sd_ma_train[valid_ma_train][rand_select_train]
selected_sd_test = sd_ma_test[valid_ma_test][rand_select_test]

X_train=np.array([selected_avhrr_ref_train, selected_avhrr_other_ref_train, selected_logai_train, selected_dem_train, selected_aod_train, selected_cld_train, selected_sd_train])
X_test=np.array([selected_avhrr_ref_test, selected_avhrr_other_ref_test, selected_logai_test, selected_dem_test, selected_aod_test, selected_cld_test, selected_sd_test])

np.save(os.path.join(ML_PATH, "X_train_{}_{}".format(ref_var, bi)), X_train)
np.save(os.path.join(ML_PATH, "X_test_{}_{}".format(ref_var, bi)), X_test)
np.save(os.path.join(ML_PATH, "Y_train_{}_{}".format(ref_var, bi)), selected_dif_train)
np.save(os.path.join(ML_PATH, "Y_test_{}_{}".format(ref_var, bi)), selected_dif_test)
