import xarray as xr
import numpy as np
from scipy.io import loadmat
import os
import rioxarray as rxr
import matplotlib.pyplot as plt
from tqdm import tqdm
import calendar
from datetime import date
import rasterio as rio
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

    
# Calibrate a model for each of the biweekly period 
biweekly_identifiers = []
for month in np.arange(1, 13):
    for half_month_index in ["a", "b"]:
        biweekly_identifiers.append('{:0>2}'.format(month) + half_month_index)
bi = biweekly_identifiers[IDX]

DEM = np.flip(xr.open_dataset(os.path.join("AVHRR/data/", "DEM_CMG.nc")).Band1, axis=0)
DEM.values[DEM.values < -414] = -414 # remove abberant values during interpolation

AI = np.flip(xr.open_dataset(os.path.join("AVHRR/data/", "AI_CMG.nc")).Band1, axis=0)
AI.values[AI.values < 0.001] = 0.001 # remove abberant values during interpolation
AI.values[AI.values > 10] = 10 # remove abberant values during interpolation
LOG_AI = np.log(AI)

LINEAR_PATH = "AVHRR/data/AVHRR_LINEAR_CORRECTED_v3.1/"
MODIS_MVC_DIR = "AVHRR/data/MODIS_MVC/"
MERRA2_DIR = "../../data/MERRA2/"
ERA5_PATH = "data/ERA5/"
ML_PATH = "AVHRR/data/ML_CORRECT_v3.1/"
ML_CORRECTED = "AVHRR/data/AVHRR_ML_CORRECTED_v3.1/"

valid_period = np.load("../../data/valid_period.npy")


# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 1
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS
rid = 0 

# Obtain the filenames for all the input datasets
yearmonth_list = []

for year in np.arange(1982, 2023):
    for month in np.arange(1, 13):
        for half_month_index in ["a", "b"]:
            yearmonth_list.append(str(year) + '{:0>2}'.format(month) + half_month_index)


yearmonth_list_valid  = [f for f in np.array(yearmonth_list)[np.array(valid_period)] if bi in f]
avhrr_red_fn = [f for f in sorted(os.listdir(LINEAR_PATH)) if "corrected_{}_{}".format("red", bi) in f][0]
avhrr_nir_fn = [f for f in sorted(os.listdir(LINEAR_PATH)) if "corrected_{}_{}".format("nir", bi) in f][0]

avhrr_red_ma = np.load(os.path.join(os.path.join(LINEAR_PATH, avhrr_red_fn)))
avhrr_nir_ma = np.load(os.path.join(os.path.join(LINEAR_PATH, avhrr_nir_fn)))




def try_gpu(i=0): 
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# feedforward model construct function
def construct_model(input_dim, hidden_dim, n_hidden_layers, drop_out=None):
    layers=[]
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    if drop_out:
        layers.append(nn.Dropout(p=0.2))
    for i in range(n_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim,hidden_dim))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(p=drop_out))
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers).to(device=try_gpu())

red_model_name = "red_layer_3_neuron_64_09-03-2023_11-52-12_lr0.001_batchsize1024"
nir_model_name = "nir_layer_3_neuron_64_09-03-2023_11-55-02_lr0.001_batchsize1024"


model_dir="../notebooks/models"
red_net = construct_model(7, 64, 3);
red_net.load_state_dict(torch.load(os.path.join(model_dir, red_model_name), map_location=torch.device('cpu')))
red_net.eval();
red_net=red_net.to(device="cpu")

nir_net = construct_model(7, 64, 3);
nir_net.load_state_dict(torch.load(os.path.join(model_dir, nir_model_name), map_location=torch.device('cpu')))
nir_net.eval();
nir_net=nir_net.to(device="cpu")


red_scaler_mean = np.load(os.path.join(ML_PATH, "{}_train_scaler_mean_v3.1.npy".format("red")))
red_scaler_var = np.load(os.path.join(ML_PATH, "{}_train_scaler_var_v3.1.npy".format("red")))
red_scaler = StandardScaler()
red_scaler.mean_ = red_scaler_mean
red_scaler.var_ = red_scaler_var
red_scaler.scale_ = np.sqrt(red_scaler_var)

nir_scaler_mean = np.load(os.path.join(ML_PATH, "{}_train_scaler_mean_v3.1.npy".format("nir")))
nir_scaler_var = np.load(os.path.join(ML_PATH, "{}_train_scaler_var_v3.1.npy".format("nir")))
nir_scaler = StandardScaler()
nir_scaler.mean_ = nir_scaler_mean
nir_scaler.var_ = nir_scaler_var
nir_scaler.scale_ = np.sqrt(nir_scaler_var)


assert len(yearmonth_list_valid) == avhrr_red_ma.shape[0]
assert len(yearmonth_list_valid) == avhrr_nir_ma.shape[0]

avhrr_ref_invalid = np.isnan(avhrr_red_ma) | np.isnan(avhrr_nir_ma)
avhrr_red_ma[avhrr_ref_invalid] = np.nan
avhrr_nir_ma[avhrr_ref_invalid] = np.nan


for i in range(len(yearmonth_list_valid)):
    aod_fn = np.array([f for f in sorted(os.listdir(os.path.join(MERRA2_DIR, "processed"))) if "CMG" in f and yearmonth_list_valid[i] in f])[0]
    aod_ds = xr.open_dataset(os.path.join(MERRA2_DIR, "processed", aod_fn))
    aod_ma = np.flip(aod_ds.Band1.values, axis=0)

    cld_fn = np.array([f for f in sorted(os.listdir(ERA5_PATH)) if "_true_CMG_biweekly_cloud_cover_" in f and yearmonth_list_valid[i] in f])[0]
    cld_ds = xr.open_dataset(os.path.join(ERA5_PATH, cld_fn))
    cld_ma = np.flip(cld_ds.Band1.values, axis=0)

    sd_fn = np.array([f for f in sorted(os.listdir(ERA5_PATH)) if "_true_CMG_biweekly_snow_depth_" in f and yearmonth_list_valid[i] in f])[0]
    sd_ds = xr.open_dataset(os.path.join(ERA5_PATH, sd_fn))
    sd_ma = np.flip(sd_ds.Band1.values, axis=0)
    
    all_valid = np.invert(np.isnan(avhrr_red_ma[i, :, :]) | np.isnan(avhrr_nir_ma[i, :, :]) | np.isnan(LOG_AI.values) | np.isnan(DEM.values) | np.isnan(aod_ma) | np.isnan(cld_ma) | np.isnan(sd_ma))
    
    red_predictors = np.array([avhrr_red_ma[i, :, :][all_valid], 
                               avhrr_nir_ma[i, :, :][all_valid],
                               LOG_AI.values[all_valid],
                               DEM.values[all_valid],
                               aod_ma[all_valid],
                               cld_ma[all_valid],
                               sd_ma[all_valid]]).T
    nir_predictors = np.array([avhrr_nir_ma[i, :, :][all_valid],
                               avhrr_red_ma[i, :, :][all_valid], 
                               LOG_AI.values[all_valid],
                               DEM.values[all_valid],
                               aod_ma[all_valid],
                               cld_ma[all_valid],
                               sd_ma[all_valid]]).T
    red_scaled_predictors = red_scaler.transform(red_predictors).astype(np.float32)
    nir_scaled_predictors = nir_scaler.transform(nir_predictors).astype(np.float32)
    
    red_predicted_dif = red_net(torch.tensor(red_scaled_predictors)).detach().numpy()[:, 0]
    nir_predicted_dif = nir_net(torch.tensor(nir_scaled_predictors)).detach().numpy()[:, 0]

    red_predicted_dif_ma = np.full(avhrr_red_ma[i, :, :].shape, np.nan, dtype=np.float32)
    red_predicted_dif_ma[all_valid] = red_predicted_dif
    corrected_red_ma = avhrr_red_ma[i, :, :] + red_predicted_dif_ma
    np.save(os.path.join(ML_CORRECTED, "ml_{}_{}.npy".format("red", yearmonth_list_valid[i])), corrected_red_ma)
    
    nir_predicted_dif_ma = np.full(avhrr_nir_ma[i, :, :].shape, np.nan, dtype=np.float32)
    nir_predicted_dif_ma[all_valid] = nir_predicted_dif
    corrected_nir_ma = avhrr_nir_ma[i, :, :] + nir_predicted_dif_ma
    np.save(os.path.join(ML_CORRECTED, "ml_{}_{}.npy".format("nir", yearmonth_list_valid[i])), corrected_nir_ma)

