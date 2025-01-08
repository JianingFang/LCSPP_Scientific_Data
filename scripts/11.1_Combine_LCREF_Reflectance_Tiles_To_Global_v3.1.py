import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tqdm import tqdm
from datetime import date, datetime, timedelta
GAPFILLED = "AVHRR/data/AVHRR_GAPFILLED_v3.2/"
LCREF = "AVHRR/data/LCREF_v3.2/"

# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 60
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS
lat=np.flip(np.arange(-89.975, 90, 0.05))
lon=np.arange(-179.975, 180, 0.05)
gap_fill_list = sorted(os.listdir(GAPFILLED))


gap_fill_list = sorted(os.listdir(GAPFILLED))


IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1
yearmonth_list = []
for year in np.arange(1982, 2024):
    for month in np.arange(1, 13):
        for half_month_index in ["a", "b"]:
            yearmonth_list.append(str(year) + '{:0>2}'.format(month) + half_month_index)
yearmonth_list = np.array(yearmonth_list).reshape(-1, 24)[IDX]
time_arr = np.load("../../data/time_arr.npy").reshape(-1, 24)[IDX]
NAN_MASK = np.expand_dims(xr.open_dataset("../../data/processed/LCSPP_LCREF_MASK.nc").LCSPP_LCREF_MASK.values, axis=0)


for i in tqdm(range(len(yearmonth_list))):

    gap_fill_nir_parts = [os.path.join(GAPFILLED, f) for f in gap_fill_list if yearmonth_list[i] in f and "nir" in f and "gf" in f]
    gap_fill_red_parts = [os.path.join(GAPFILLED, f) for f in gap_fill_list if yearmonth_list[i] in f and "red" in f and "gf" in f]
    red_hf_parts = [os.path.join(GAPFILLED, f) for f in gap_fill_list if yearmonth_list[i] in f and "nir" in f and "hf" in f]
    nir_hf_parts =  [os.path.join(GAPFILLED, f) for f in gap_fill_list if yearmonth_list[i] in f and "red" in f and "hf" in f]

    nir_ma = np.full((3600, 7200), np.nan, dtype=np.float32)
    red_ma = np.full((3600, 7200), np.nan, dtype=np.float32)
    nir_hf_ma = np.full((3600, 7200), 100, dtype=np.uint8)
    red_hf_ma = np.full((3600, 7200), 100, dtype=np.uint8)

    for j in range(len(gap_fill_nir_parts)):
        nir_part = np.load(gap_fill_nir_parts[j])
        red_part = np.load(gap_fill_red_parts[j])
        nir_ma[j*SINGLE_HEIGHT:(j+1)*SINGLE_HEIGHT, :] = nir_part
        red_ma[j*SINGLE_HEIGHT:(j+1)*SINGLE_HEIGHT, :] = red_part

    for j in range(len(nir_hf_parts)):
        nir_hf_part = np.load(nir_hf_parts[j])
        red_hf_part = np.load(red_hf_parts[j])
        nir_hf_ma[j*SINGLE_HEIGHT:(j+1)*SINGLE_HEIGHT, :] = nir_hf_part
        red_hf_ma[j*SINGLE_HEIGHT:(j+1)*SINGLE_HEIGHT, :] = red_hf_part

    nir_hf_ma[nir_hf_ma==100] = 3
    red_hf_ma[red_hf_ma==100] = 3

    red_qa_da = xr.DataArray(np.expand_dims(red_hf_ma, axis=0), coords={"time":lcref_ds.time,
                                                            "lat":lcref_ds.lat,
                                                            "lon":lcref_ds.lon})

    red_qa_da.attrs={"long_name":"Gapfill Quality for Red Channel", "description":"0=observation, 1=good-quality gapfill by HANTS, 2=low-quality gapfill with mean seasonal cycle, 3=no data"}

    nir_qa_da = xr.DataArray(np.expand_dims(nir_hf_ma, axis=0), coords={"time":lcref_ds.time,
                                                            "lat":lcref_ds.lat,
                                                            "lon":lcref_ds.lon})

    nir_qa_da.attrs={"long_name":"Gapfill Quality for NIR Channel", "description":"0=observation, 1=good-quality gapfill by HANTS, 2=low-quality gapfill with mean seasonal cycle, 3=no data"}

    red_qa_da.values[NAN_MASK] = 3
    nir_qa_da.values[NAN_MASK] = 3

    # Apparently the lack of data during winter months
    # can cause some abberant values ~0.2% in the high latitude
    # regions forming distinctive bands, these issues can be avoided in later
    # analysis by setting SIF in high latidude region to 0, as well as excluding
    # time where the mean T is below 0 (not growing season). For now, we keep the pixels
    # as is.

    #nir_ma[nir_ma > 1] = np.nan
    #nir_ma[nir_ma < 0] = np.nan

    #red_ma[red_ma > 1] = np.nan
    #red_ma[red_ma < 0] = np.nan

    # we will keep this for now to be consistent with modis
    ndvi = (nir_ma-red_ma)/(nir_ma+red_ma)
    INVALID_NDVI = (ndvi > 1) | (ndvi < -1)
    nir_ma[INVALID_NDVI] = np.nan
    red_ma[INVALID_NDVI] = np.nan

    red_da = xr.DataArray(np.expand_dims(red_ma, axis=0), coords={"time":[time_arr[i],], "lat":lat, "lon":lon})
    nir_da = xr.DataArray(np.expand_dims(nir_ma, axis=0), coords={"time":[time_arr[i],], "lat":lat, "lon":lon})

    lcref_ds = xr.Dataset({"red":red_da, "nir":nir_da, "red_qa":red_qa_da, "nir_qa":nir_qa_da})
    lcref_ds.lat.attrs={"long_name":"latitude", "units":"degrees_north"}
    lcref_ds.lon.attrs={"long_name":"longitude", "units":"degrees_east"}
    lcref_ds.time.values[0]=lcref_ds.time.values.astype('datetime64[D]').astype('datetime64[ns]')[0]

    lcref_ds.attrs = {"title": "Long-term Continuous REFlectance from calibrated AVHRR record (LCREF-AVHRR)", 
                    "spatial_resolution": "0.050000 degrees per pixel",
                    "geospatial_lat_min": "-90",
                    "geospatial_lat_max": "90",
                    "geospatial_lon_min":"-180",
                    "geospatial_lon_max":"180",
                    "product_version": "v3.2",
                    "doi":"10.5281/zenodo.11905959",
                    "filename_notation": "a: day1-day15 of the month, b:day16-last day of the month",
                    "contacts": "Jianing Fang (jf3423@columbia.edu), Xu Lian (xl3179@columbia.edu)",
                    "date_source": "LTDR AVHRR V5 AVH09C1, MCD43C1 v061",
                    "created_date":date.today().strftime("%m/%d/%Y")}
    
    lcref_ds.to_netcdf(os.path.join(LCREF, "LCREF_AVHRR_v3.2_{}.nc".format(yearmonth_list[i])))

