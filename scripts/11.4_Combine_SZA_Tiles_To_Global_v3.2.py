import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tqdm import tqdm
from datetime import date, datetime, timedelta
GAPFILLED = "AVHRR/data/SZA_GAPFILLED/"
SZA = "AVHRR/data/SZA/"

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

for i in tqdm(range(len(yearmonth_list))):

    gap_fill_sza_parts = [os.path.join(GAPFILLED, f) for f in gap_fill_list if yearmonth_list[i] in f and "sza" in f]


    sza_ma = np.full((3600, 7200), np.nan, dtype=np.float32)


    for j in range(len(gap_fill_sza_parts)):
        sza_part = np.load(gap_fill_sza_parts[j])
        sza_ma[j*SINGLE_HEIGHT:(j+1)*SINGLE_HEIGHT, :] = sza_part

    sza_da = xr.DataArray(np.expand_dims(sza_ma, axis=0), coords={"time":[time_arr[i],], "lat":lat, "lon":lon})

    sza_ds = xr.Dataset({"sza":sza_da})
    sza_ds.lat.attrs={"long_name":"latitude", "units":"degrees_north"}
    sza_ds.lon.attrs={"long_name":"longitude", "units":"degrees_east"}

    sza_ds.attrs = {"title": "Gapfilled SZA", 
                    "spatial_resolution": "0.050000 degrees per pixel",
                    "geospatial_lat_min": "-90",
                    "geospatial_lat_max": "90",
                    "geospatial_lon_min":"-180",
                    "geospatial_lon_max":"180",
                    "product_version": "v_ltdr",
                    "filename_notation": "a: day1-day15 of the month, b:day16-last day of the month",
                    "contacts": "Jianing Fang (jf3423@columbia.edu), Xu Lian (xl3179@columbia.edu)",
                    "date_source": "LTDR AVHRR V5 AVH09C1",
                    "created_date":date.today().strftime("%m/%d/%Y")}

    sza_ds.to_netcdf(os.path.join(SZA, "SZA_{}.nc".format(yearmonth_list[i])))

