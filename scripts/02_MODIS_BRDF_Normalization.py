import sasktran as sk
# https://usask-arg.github.io/sasktran/brdf.html#brdf
import rioxarray as rxr
import os
import numpy as np
from datetime import datetime
from cysgp4 import PyDateTime
from tqdm import tqdm
from scipy.stats import linregress
import calendar
from datetime import date, timedelta
import xarray as xr
import sys


DATA_ROOT_DIR = "AVHRR/data/"
MCD43C1_DIR = os.path.join(DATA_ROOT_DIR, "MCD43C1.v061")
MCD43C1_files = [os.path.join(MCD43C1_DIR, f) for f in sorted(os.listdir(MCD43C1_DIR)) if ".hdf" in f]



BANDS_TO_USE = ["BRDF_Albedo_Parameter1_Band1",
                "BRDF_Albedo_Parameter1_Band2",
                "BRDF_Albedo_Parameter2_Band1",
                "BRDF_Albedo_Parameter2_Band2",
                "BRDF_Albedo_Parameter3_Band1",
                "BRDF_Albedo_Parameter3_Band2",
                "Local_Solar_Noon",
                "Percent_Snow",
                "BRDF_Quality"]
MVC_DIR = os.path.join(DATA_ROOT_DIR, "MODIS_MVC")
REF45_DIR = os.path.join(DATA_ROOT_DIR, "MODIS_45")


RED_NM = 645 # center of modis band 1 wavelength in nm
NIR_NM = 858.5 # center of modis band 1 wavelength in nm

lat = np.flip(np.arange(-89.975, 90, 0.05)).astype(np.float32)
lon = np.arange(-179.975, 180, 0.05).astype(np.float32)

BRDF_SCALE_FACTOR = 0.001
BRDF_FILL_VALUE = 32767
SZA_FILL_VALUE = 255
BRDF_QA_THRESHOLD = 3

"""
    ompute_reflectance_45(f_1, f_2, f_3, year, month, day, nm)
    
compute the BRDF reflectance 

"""
def compute_reflectance_45(f_1, f_2, f_3, year, month, day, nm):
    mjd = PyDateTime(datetime(year, month, day)).mjd
    reflectance_45 = np.full((3600, 7200), np.nan, dtype=np.float32)
    for i in range(f_1.shape[0]):
        for j in range(f_1.shape[1]):
            if np.invert(np.isnan(f_1[i, j]) | np.isnan(f_2[i, j]) | np.isnan(f_3[i, j])):
                brdf = sk.MODIS(f_1[i, j], f_2[i, j], f_3[i, j])
                reflectance_45[i, j] = brdf.reflectance(nm, lat[i], lon[j], mjd,
                                                     np.cos(45/180 * np.pi),np.cos(0), -1)
                brdf = None
    return reflectance_45 * np.pi

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
    return "MODIS_MVC_" + str(year_start) + '{:0>2}'.format(date_mvc.month) + hm_symbol + ".nc"


def construct_ref_45(fn, save_file=True):
    if os.path.isfile(os.path.join(REF45_DIR, fn.split("/")[-1].split(".")[1] + ".nc")):
        ds = xr.open_dataset(os.path.join(REF45_DIR, fn.split("/")[-1].split(".")[1] + ".nc"))
        print(ds)
        red_45 = ds["red_brdf_normalized_sza45"][0, :, :].values
        nir_45 = ds["nir_brdf_normalized_sza45"][0, :, :].values
        percent_snow = ds["percent_snow"][0, :, :].values
        ndvi_45 = (nir_45 - red_45)/(nir_45 + red_45)
        return red_45, nir_45, ndvi_45, percent_snow
    else:
        year = int(fn.split("/")[-1].split(".")[1][1:5])
        doy = int(fn.split("/")[-1].split(".")[1][5:8])
        date_obj = date(year, 1, 1) + timedelta(days=doy-1)
        month = date_obj.month
        day = date_obj.day

        c1_ds = rxr.open_rasterio(fn)
        c1_ds=c1_ds[BANDS_TO_USE]
        BRDF_Quality = c1_ds.BRDF_Quality.values[0, :, :]

        f_1_red = c1_ds.BRDF_Albedo_Parameter1_Band1.values[0, :, :].astype(np.float32)
        f_1_red[f_1_red==BRDF_FILL_VALUE] = np.nan
        f_1_red = f_1_red * BRDF_SCALE_FACTOR
        f_1_red[BRDF_Quality > BRDF_QA_THRESHOLD] = np.nan

        f_2_red = c1_ds.BRDF_Albedo_Parameter2_Band1.values[0, :, :].astype(np.float32)
        f_2_red[f_2_red==BRDF_FILL_VALUE] = np.nan
        f_2_red = f_2_red * BRDF_SCALE_FACTOR
        f_2_red[BRDF_Quality > BRDF_QA_THRESHOLD] = np.nan 

        f_3_red = c1_ds.BRDF_Albedo_Parameter3_Band1.values[0, :, :].astype(np.float32)
        f_3_red[f_3_red==BRDF_FILL_VALUE] = np.nan
        f_3_red = f_3_red * BRDF_SCALE_FACTOR
        f_3_red[BRDF_Quality > BRDF_QA_THRESHOLD] = np.nan 


        f_1_nir = c1_ds.BRDF_Albedo_Parameter1_Band2.values[0, :, :].astype(np.float32)
        f_1_nir[f_1_nir==BRDF_FILL_VALUE] = np.nan
        f_1_nir = f_1_nir * BRDF_SCALE_FACTOR
        f_1_nir[BRDF_Quality > BRDF_QA_THRESHOLD] = np.nan 

        f_2_nir = c1_ds.BRDF_Albedo_Parameter2_Band2.values[0, :, :].astype(np.float32)
        f_2_nir[f_2_nir==BRDF_FILL_VALUE] = np.nan
        f_2_nir = f_2_nir * BRDF_SCALE_FACTOR
        f_2_nir[BRDF_Quality > BRDF_QA_THRESHOLD] = np.nan 

        f_3_nir = c1_ds.BRDF_Albedo_Parameter3_Band2.values[0, :, :].astype(np.float32)
        f_3_nir[f_3_nir==BRDF_FILL_VALUE] = np.nan
        f_3_nir = f_3_nir * BRDF_SCALE_FACTOR
        f_3_nir[BRDF_Quality > BRDF_QA_THRESHOLD] = np.nan 
        
        percent_snow = c1_ds.Percent_Snow.values[0, :, :].astype(np.float32)


        solar_noon = c1_ds.Local_Solar_Noon.values[0, :, :].astype(np.float32)
        solar_noon[solar_noon==SZA_FILL_VALUE] = np.nan
        qa = c1_ds.BRDF_Albedo_Parameter1_Band1.values[0, :, :]
        BRDF_Quality = c1_ds.BRDF_Quality.values[0, :, :]

        red_45 = compute_reflectance_45(f_1_red, f_2_red, f_3_red, year, month, day, RED_NM)
        nir_45 = compute_reflectance_45(f_1_nir, f_2_nir, f_3_nir, year, month, day, NIR_NM)

        ndvi_45 = (nir_45 - red_45)/(nir_45 + red_45)
        INVALID_NDVI = (ndvi_45 > 1) | (ndvi_45 < -1)
        red_45[INVALID_NDVI] = np.nan
        nir_45[INVALID_NDVI] = np.nan
        ndvi_45[INVALID_NDVI] = np.nan
        
        c1_ds.close()


        if save_file:

            red_brdf_normalized_sza45 = xr.DataArray(np.expand_dims(red_45, axis=0),
                                                     coords={"time":[np.datetime64(date_obj)], "lat":lat, "lon":lon})

            nir_brdf_normalized_sza45 = xr.DataArray(np.expand_dims(nir_45, axis=0),
                                                     coords={"time":[np.datetime64(date_obj)], "lat":lat, "lon":lon})
            
            percent_da = xr.DataArray(np.expand_dims(percent_snow, axis=0),
                                                     coords={"time":[np.datetime64(date_obj)], "lat":lat, "lon":lon})

            ref_45_ds = xr.Dataset({"red_brdf_normalized_sza45":red_brdf_normalized_sza45,
                        "nir_brdf_normalized_sza45":nir_brdf_normalized_sza45,
                                   "percent_snow":percent_da})

            ref_45_ds.to_netcdf(os.path.join(REF45_DIR, fn.split("/")[-1].split(".")[1] + ".nc"))

            ref_45_ds.close()

        return red_45, nir_45, ndvi_45, percent_snow

            
def worker(hml):
    MAX_NDVI = np.full((3600, 7200), -2.0, dtype=np.float32)
    BEST_RED = np.full((3600, 7200), np.nan, dtype=np.float32)
    BEST_NIR = np.full((3600, 7200), np.nan, dtype=np.float32)
    BEST_PERCENT_SNOW = np.full((3600, 7200), np.nan, dtype=np.float32)

    for f in hml:
        red_45, nir_45, ndvi_45, percent_snow = construct_ref_45(f, save_file=True)
        BEST_RED[ndvi_45 > MAX_NDVI] = red_45[ndvi_45 > MAX_NDVI]
        BEST_NIR[ndvi_45 > MAX_NDVI] = nir_45[ndvi_45 > MAX_NDVI]
        BEST_PERCENT_SNOW[ndvi_45 > MAX_NDVI] = percent_snow[ndvi_45 > MAX_NDVI]
        MAX_NDVI[ndvi_45 > MAX_NDVI] = ndvi_45[ndvi_45 > MAX_NDVI]

    RED_MVC = xr.DataArray(np.expand_dims(BEST_RED, axis=0), 
                                             coords={"time":[get_date(hml)], "lat":lat, "lon":lon})

    NIR_MVC = xr.DataArray(np.expand_dims(BEST_NIR, axis=0),
                                             coords={"time":[get_date(hml)], "lat":lat, "lon":lon})
    
    PERCENT_SNOW_MVC = xr.DataArray(np.expand_dims(BEST_PERCENT_SNOW, axis=0),
                                             coords={"time":[get_date(hml)], "lat":lat, "lon":lon})

    mvc_ds = xr.Dataset({"red_brdf_normalized_sza45":RED_MVC,
                "nir_brdf_normalized_sza45":NIR_MVC,
                        "percent_snow":PERCENT_SNOW_MVC})
    mvc_ds.to_netcdf(os.path.join(MVC_DIR, get_filename(hml)))
    mvc_ds.close()

    for f in hml:
        os.remove(f)
        
def lcspp_worker(hml):
    for f in hml:
        red_45, nir_45, ndvi_45, percent_snow = construct_ref_45(f, save_file=True)
    for f in hml:
        os.remove(f)

if __name__ == "__main__":  
    YY = int(sys.argv[1])
    IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
    file_list_16days = []
    for year in np.arange(YY, YY+1):
        for month in np.arange(1, 13):
            days_in_month = calendar.monthrange(year, month)[1]
            for half_month_index in ["a", "b"]:
                if half_month_index == "a":
                    day_range = np.arange(1, 16)
                elif half_month_index == "b":
                    day_range = np.arange(16, days_in_month + 1)
                half_month_file_list = []
                for day in day_range:
                    doy = date(year, month, day).timetuple().tm_yday
                    fn = [f for f in MCD43C1_files if "A"+str(year)+'{:0>3}'.format(doy) in f]
                    if len(fn) > 0:
                        half_month_file_list.append(fn[0])
                file_list_16days.append(half_month_file_list)
    worker(file_list_16days[IDX])
