import xarray as xr
import matplotlib.pyplot as plt
import os
import netCDF4 as nc
import numpy as np
import datetime
import rioxarray as rxr
import calendar
from datetime import date, timedelta

AVHRR_DIR="../../data/LTDR/"
AVHRR_files = [os.path.join(AVHRR_DIR, f) for f in sorted(os.listdir(AVHRR_DIR)) if ".hdf" in f]
SCALE = 0.0001
MVC_DIR = "AVHRR/data/AVHRR_MVC/"


def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

def get_date(hml):
    year_start = int(hml[0].split("/")[-1].split(".")[1][1:5])
    doy_start = int(hml[0].split("/")[-1].split(".")[1][5:8])
    doy_end = int(hml[-1].split("/")[-1].split(".")[1][5:8])
    
    return np.datetime64(date(year_start, 1, 1) + timedelta(days=(doy_start+doy_end)/2-1))

def get_filename(hml):
    year_start = int(hml[0].split("/")[-1].split(".")[1][1:5])
    doy_start = int(hml[0].split("/")[-1].split(".")[1][5:8])
    date_mvc = date(year_start, 1, 1) + timedelta(days=doy_start-1)
    if date_mvc.day <10:
        hm_symbol="a"
    else:
        hm_symbol="b"
    return "AVHRR_MVC_" + str(year_start) + '{:0>2}'.format(date_mvc.month) + hm_symbol + ".nc"


def get_avhrr_data(f):
    ds = rxr.open_rasterio(f)
    unpacked_qa = unpackbits(ds.QA.values[0, :, :], 16)
    iscloudy = unpacked_qa[:,:, 1].reshape(3600, 7200)==1
    iscloudyshad = unpacked_qa[:,:, 2].reshape(3600, 7200)==1
    iswater = unpacked_qa[:,:, 3].reshape(3600, 7200)==1
    isnight = unpacked_qa[:,:, 6].reshape(3600, 7200)==1
    isc1inv = unpacked_qa[:,:, 8].reshape(3600, 7200)==1
    isc2inv = unpacked_qa[:,:, 9].reshape(3600, 7200)==1
    isnobrdf = unpacked_qa[:,:, 14].reshape(3600, 7200)==0

    ismask_red = (iswater | iscloudy | iscloudyshad | isnight | isc1inv | isnobrdf)
    ismask_nir = (iswater | iscloudy | iscloudyshad | isnight | isc2inv | isnobrdf)

    red = ds.SREFL_CH1[0, :, :].values.astype(np.float32) * SCALE
    red[ismask_red] = np.nan
    nir = ds.SREFL_CH2[0, :, :].values.astype(np.float32) * SCALE
    nir[ismask_nir] = np.nan
    
    ndvi = (nir - red) / (nir + red)

    sza = ds.SZEN.values[0, :, :].astype(np.float32)
    sza[sza==-9999]=np.nan
    sza = sza * 0.01
    
    return red, nir, ndvi, sza
        
def avhrr_mvc(hml):
    lat = np.flip(np.arange(-89.975, 90, 0.05)).astype(np.float32)
    lon = np.arange(-179.975, 180, 0.05).astype(np.float32)

    MAX_NDVI = np.full((3600, 7200), -2.0, dtype=np.float32)
    BEST_RED = np.full((3600, 7200), np.nan, dtype=np.float32)
    BEST_NIR = np.full((3600, 7200), np.nan, dtype=np.float32)
    BEST_SZA = np.full((3600, 7200), np.nan, dtype=np.float32)
    
    for f in hml:
        red, nir, ndvi, sza = get_avhrr_data(f)
        BEST_RED[ndvi > MAX_NDVI] = red[ndvi > MAX_NDVI]
        BEST_NIR[ndvi > MAX_NDVI] = nir[ndvi > MAX_NDVI]
        BEST_SZA[ndvi > MAX_NDVI] = sza[ndvi > MAX_NDVI]
        MAX_NDVI[ndvi > MAX_NDVI] = ndvi[ndvi > MAX_NDVI]
        
    RED_MVC = xr.DataArray(np.expand_dims(BEST_RED, axis=0), 
                                             coords={"time":[get_date(hml)], "lat":lat, "lon":lon})

    NIR_MVC = xr.DataArray(np.expand_dims(BEST_NIR, axis=0),
                                             coords={"time":[get_date(hml)], "lat":lat, "lon":lon})
    
    SZA_MVC = xr.DataArray(np.expand_dims(BEST_SZA, axis=0),
                                             coords={"time":[get_date(hml)], "lat":lat, "lon":lon})

    mvc_ds = xr.Dataset({"red_brdf_normalized_sza45":RED_MVC,
                "nir_brdf_normalized_sza45":NIR_MVC, "sza":SZA_MVC})
    
    
    mvc_ds.to_netcdf(os.path.join(MVC_DIR, get_filename(hml)))        
    mvc_ds.close()

    for f in hml:
        os.remove(f)
        
        
if __name__ == "__main__":  
    IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

    YY=np.arange(1982, 2023)[IDX]
    file_list_16days = []
    for year in np.arange(YY, YY+1):
        mr = np.arange(1, 13)
        for month in mr:
            days_in_month = calendar.monthrange(year, month)[1]
            for half_month_index in ["a", "b"]:
                if half_month_index == "a":
                    day_range = np.arange(1, 16)
                elif half_month_index == "b":
                    day_range = np.arange(16, days_in_month + 1)
                half_month_file_list = []
                for day in day_range:
                    doy = date(year, month, day).timetuple().tm_yday
                    fn = [f for f in AVHRR_files if "A"+str(year)+'{:0>3}'.format(doy) in f]
                    if len(fn) > 0:
                        half_month_file_list.append(fn[0])
                file_list_16days.append(half_month_file_list)

    for hml in file_list_16days:
        avhrr_mvc(hml)
