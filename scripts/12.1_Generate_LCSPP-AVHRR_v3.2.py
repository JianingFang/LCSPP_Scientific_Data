import numpy as np
import xarray as xr
import os
import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp
import torch
from torch import nn
from datetime import date, time, timedelta
import datetime
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LogNorm
import astral
from astral import sun
import pandas as pd
from scipy import interpolate
import netCDF4 as nc

LCREF_v3_DIR = "../../data/LCREF_v3.2/"
LCSPP_v3_DIR = "../../data/LCSPP_v3.2/"
lcspp_processed_dir = os.path.join("../../data/", "processed")

DEM = np.flip(xr.open_dataset(os.path.join("AVHRR/data/", "DEM_CMG.nc")).Band1, axis=0)
DEM.values[DEM.values < -414] = -414 # remove abberant values during interpolation
elevation = np.full((3600, 7200), np.nan, dtype=np.float32)
elevation[0:3000, :]=DEM * 0.001

latitude_time_diff=np.load("../../data/processed/latitude_time_diff_sep_6_2014.npy")
f=interpolate.interp1d(latitude_time_diff[0], latitude_time_diff[1], kind="linear")

def compute_cos_sza_for_fitted_overpass(latitude, date_of_interest):
    overpass_time_diff=float(f(latitude))
    overpass_time_delta=datetime.timedelta(hours=overpass_time_diff)
    cos_sza=np.cos(astral.sun.zenith(astral.LocationInfo(latitude=latitude, longitude=0).observer,
                             dateandtime=datetime.datetime.combine(date_of_interest, datetime.time(13,36)) + overpass_time_delta,
                             with_refraction = True) / 180 * np.pi)
    return cos_sza

def compute_daily_sza_for_fitted_overpass(latitude, date_of_interest):
    overpass_time_diff=float(f(latitude))
    overpass_time_delta=datetime.timedelta(hours=overpass_time_diff)
    eval_points=np.arange(-0.5,0.501, 1/(6*24))
    daily_sza_points=np.array([np.cos(astral.sun.zenith(astral.LocationInfo(latitude=latitude, longitude=0).observer,
                             dateandtime=datetime.datetime.combine(date_of_interest, datetime.time(13,36)) + overpass_time_delta + datetime.timedelta(days=p),
                             with_refraction = True) / 180 * np.pi) for p in eval_points])
    daily_sza_points[daily_sza_points < 0] = 0
    return np.mean(daily_sza_points)

# compute TOA shortwave radiation
def compute_R_toa(cos_sza, doy):
    S0 = 1360.8
    alpha = 0.98
    return S0 * alpha * (1 + 0.033 * np.cos(2 * np.pi * doy / 365)) * cos_sza
# compute clear sky shortwave radiation. See Zhang et al. 2017 CSIF paper for reference
def compute_total_surface_shortwave_radiation(cos_sza, doy, elevation):
    R_toa = compute_R_toa(cos_sza, doy)
    a0 = 0.4237 - 0.00821 * (6 - elevation)**2
    a1 = 0.5055 + 0.00595 * (6.5 - elevation)**2
    k = 0.2711 + 0.01858 * (2.5 - elevation)**2
    tau_b = a0 + a1 * np.exp(-k/cos_sza)
    tau_d = 0.271 - 0.294 * tau_b
    R_sb = R_toa * tau_b
    R_sd = R_toa * tau_d
    R_t = R_sb + R_sd
    R_t[cos_sza < 0] = 0
    return R_t

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

def generate_lcspp_prediction(file, ssrd_file, IDX):
    ds=xr.open_dataset(file)
    # the ml calibrated avhrr
    valid_flag = (np.invert(np.isnan(ds.red.values))) & (np.invert(np.isnan(ds.nir.values))) & (ds.nir.values > 0) & (ds.nir.values < 1) & (ds.red.values > 0) & (ds.red.values < 1) 
    valid_flag = valid_flag[0]
    red_valid = ds.red.values[0][valid_flag]
    nir_valid = ds.nir.values[0][valid_flag]
    lon_valid = np.tile(ds.lon.values, (3600,1))[valid_flag]
    lat_valid = np.tile(ds.lat.values, (7200,1)).T[valid_flag]

    # compute cos(sza)
    doi = pd.to_datetime(ds.time[0].values).date()
    doy = doi.timetuple().tm_yday
    computed_cos_sza=np.full(3600, np.nan)
    computed_cos_sza[164:-164]=np.array([compute_cos_sza_for_fitted_overpass(l, doi) for l in ds.lat.values[164:-164]])
    computed_cos_sza_array=np.tile(computed_cos_sza, (7200,1)).T

    computed_cos_daily_sza=np.full(3600, np.nan)
    computed_cos_daily_sza[164:-164]=np.array([compute_daily_sza_for_fitted_overpass(l, doi) for l in ds.lat.values[164:-164]])
    computed_cos_daily_sza_array=np.tile(computed_cos_daily_sza, (7200,1)).T


    computed_cos_sza_valid = computed_cos_sza_array[valid_flag]
    
    
    computed_cos_daily_sza_valid = computed_cos_daily_sza_array[valid_flag]

    # create data matrices 
    data_matrix = np.array([red_valid, nir_valid, computed_cos_sza_valid]).T
    scaled_data_matrix = scaler.transform(data_matrix)
  
    # make predictions
    with torch.no_grad():
        predicted=net(torch.tensor(scaled_data_matrix).float().to(try_gpu())).cpu().numpy().squeeze()

    # map back to 2D array
    lcspp=np.zeros((3600, 7200))
    lcspp[valid_flag]=predicted
    lcspp[np.invert(valid_flag)]=np.nan
    


    cos_sza=np.zeros((3600, 7200))
    cos_sza[valid_flag]=computed_cos_sza_valid
    cos_sza[np.invert(valid_flag)]=np.nan
    lcspp[(cos_sza <= 0) & np.invert(np.isnan(lcspp))] = 0
    
    cos_daily_sza=np.zeros((3600, 7200))
    cos_daily_sza[valid_flag]=computed_cos_daily_sza_valid
    cos_daily_sza[np.invert(valid_flag)]=np.nan
    
    lcspp_clear_daily = sif / cos_sza * cos_daily_sza
    lcspp_clear_daily[(cos_sza <= 0) & np.invert(np.isnan(lcspp_clear_daily))] = 0
    lcspp_clear_daily[(cos_daily_sza <= 0) & np.invert(np.isnan(lcspp_clear_daily))] = 0
    
    era_5_rad = np.flip(xr.open_dataset(ssrd_file).Band1.values, axis=0)
    R_t = compute_total_surface_shortwave_radiation(cos_sza, doy, elevation)
    all_daily = sif /R_t * era_5_rad
    all_daily[R_t <= 0] = 0
    all_daily[np.isnan(lcspp)] = np.nan
    

    lcspp_clear_inst_array=xr.DataArray(np.expand_dims(lcspp, axis=0),
                 coords=[ds.time.values, ds.lat, ds.lon],
                 dims=["time", "lat", "lon"])
        
    
    lcspp_clear_daily_array=xr.DataArray(np.expand_dims(lcspp_clear_daily, axis=0),
                 coords=[ds.time.values, ds.lat, ds.lon],
                 dims=["time", "lat", "lon"])
    
    lcspp_all_daily_array=xr.DataArray(np.expand_dims(all_daily, axis=0),
                 coords=[ds.time.values, ds.lat, ds.lon],
                 dims=["time", "lat", "lon"])
    """

    cos_sza_array=xr.DataArray(np.expand_dims(cos_sza, axis=0),
                 coords=[ds.time.values, ds.lat, ds.lon],
                 dims=["time", "lat", "lon"])

    cos_daily_sza_array=xr.DataArray(np.expand_dims(cos_daily_sza, axis=0),
                 coords=[ds.time.values, ds.lat, ds.lon],
                 dims=["time", "lat", "lon"])
    """
    

    
    lcspp_ds=xr.Dataset({"lcspp_clear_inst":lcspp_clear_inst_array,
                       "lcspp_clear_daily":lcspp_clear_daily_array,
                       "lcspp_all_daily":lcspp_all_daily_array})
    lcspp_ds = lcspp_ds.astype(np.float32) 

    lcspp_qa_ma = np.full((1, 3600, 7200), 100, dtype=np.uint8)
    lcspp_qa_ma[((ds.red_qa==0) & (ds.nir_qa==0)).values] = 0
    lcspp_qa_ma[(((ds.red_qa<=1) & (ds.nir_qa<=1)).values) & (lcspp_qa_ma!=0)] = 1
    lcspp_qa_ma[(lcspp_qa_ma>1) & np.invert(np.isnan(lcspp_ds.lcspp_clear_inst.values))] = 2
    lcspp_qa_ma[lcspp_qa_ma==100]=3
    lcspp_qa_da.attrs={"long_name":"Gapfill Quality for LCSPP", "description":"0=observation, 1=good-quality gapfill by HANTS, 2=low-quality gapfill with mean seasonal cycle, 3=no data"}
    lcspp_qa_da = xr.DataArray(lcspp_qa_ma, coords={"time":lcspp_ds.time,
                                                            "lat":lcspp_ds.lat,
                                                            "lon":lcspp_ds.lon})
    lcspp_ds["lcspp_qa"] = lcspp_qa_da


    if IDX < 19: 
        DOI = "10.5281/zenodo.7916850" #for 1982-2000
    else:
        DOI = "10.5281/zenodo.11906675" # for 2001-2023

    lcspp_ds["lcspp_all_daily"].attrs={"long_name":"all-sky daily average LCSPP weighted by ERA5 SSRD"}
    lcspp_ds["lcspp_clear_inst"].attrs={"long_name":"instantaneous clear-sky predicted LCSPP", "units":"mW m-2 nm-1 sr-1"}
    lcspp_ds["lcspp_clear_daily"].attrs={"long_name": "daily clear-sky predicted LCSPP adjusted by cosine solar zenith angle", "units":"mW m-2 nm-1 sr-1"}
    
    lcspp_ds.attrs = {"title": "Long-term Continuous SIF-informed Photosynthesis Proxy reconstructed with calibrated AVHRR surface reflectance (LCSPP-AVHRR)", 
                    "spatial_resolution": "0.050000 degrees per pixel",
                    "geospatial_lat_min": "-90",
                    "geospatial_lat_max": "90",
                    "geospatial_lon_min":"-180",
                    "geospatial_lon_max":"180",
                    "product_version": "v3.2",
                    "doi":DOI,
                    "filename_notation": "a: day1-day15 of the month, b:day16-last day of the month",
                    "contacts": "Jianing Fang (jf3423@columbia.edu), Xu Lian (xl3179@columbia.edu)",
                    "date_source": "LCREF_AVHRR_v3.2, MCD43C1 v061, OCO-2 SIF Lite V11r, ERA5 Reanalysis",
                    "created_date":datetime.date.today().strftime("%m/%d/%Y")}    
    
    lcspp_ds.to_netcdf(os.path.join(LCSPP_v3_DIR, "LCSPP_AVHRR_v3.2_" + file.split("/")[-1][5:]))
    lcspp_ds.close()
    ds.close()
    

scaler = StandardScaler()

lcspp_train_mean = np.load("../../data/lcspp_train_mean.npy")
lcspp_train_var = np.load("../../data/lcspp_train_var.npy")
lcspp_train_scale = np.load("../../data/lcspp_train_scale.npy")

scaler.mean_  = lcspp_train_mean
scaler.var_  = lcspp_train_var
scaler.scale_  = lcspp_train_scale

hidden_dim=64
n_hidden_layers=2
net= construct_model(3, hidden_dim, n_hidden_layers)
model_name="layer_2_neuron_64_10-11-2023_10-25-03_lr0.001_batchsize1024"
model_dir="../notebooks/models"
net.load_state_dict(torch.load(os.path.join(model_dir, model_name), map_location=torch.device('cpu')))
net.eval();

IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
AVHRR_FILE_YEAR_LIST_AVAILABLE = np.array([os.path.join(LCREF_v3_DIR, f) for f in sorted(os.listdir(LCREF_v3_DIR))]).reshape(-1, 24)[IDX]
SSRD_YEAR_LIST = np.array(sorted([os.path.join("../../../data/ERA5", f) for f in os.listdir("../../../data/ERA5") if "CMG_ssrd_biweekly_" in f])).reshape(-1, 24)[IDX]

for l in range(len(AVHRR_FILE_YEAR_LIST_AVAILABLE)):
    generate_lcspp_prediction(AVHRR_FILE_YEAR_LIST_AVAILABLE[l], SSRD_YEAR_LIST[l], IDX)
