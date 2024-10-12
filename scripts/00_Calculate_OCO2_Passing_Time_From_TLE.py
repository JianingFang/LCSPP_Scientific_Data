import numpy as np
import os
import itertools
from torch import nn
from datetime import date, time, timedelta
import datetime
import skyfield
from skyfield.api import load, wgs84
import pandas as pd
from scipy import interpolate
import netCDF4 as nc


# TLE data obtained from https://www.space-track.org/

stations_url = '../../data/OCO2_TLE_SHORT.txt'

satellites = load.tle_file(stations_url)
ts = load.timescale()
print('Loaded', len(satellites), 'satellites')
epoches=np.array([sat.epoch.utc_datetime() for sat in satellites])
epoch_duration=np.diff(epoches)

observe_time=[]
lons=[]
lats=[]

for i, epoch in enumerate(epoches[:-1]):
    if epoch_duration[i].total_seconds() > 1:
        epoch_start = ts.from_datetime(epoch)
        times=pd.date_range(epoch, periods=int(epoch_duration[i].total_seconds()/10), freq='10S').tolist()
        t=ts.from_datetimes(times)
        satellite = satellites[i]
        geocentric = satellite.at(t)
        observe_time.append(t.utc_datetime())
        subpoint = wgs84.subpoint(geocentric)
        lons.append(subpoint.longitude.degrees)
        lats.append(subpoint.latitude.degrees)
        
lon_array=np.concatenate(lons, axis=0)
lat_array=np.concatenate(lats, axis=0)
time_array=np.concatenate(observe_time, axis=0)
date_array=np.array([t.date() for t in time_array])

np.save("../../data/processed/OCO2_Track.npy", np.array([lat_array, lon_array, time_array]).T)

date_of_interest=date(2014,9,6)
date_sel=date_array == date_of_interest
lon_sel=lon_array[date_sel]
lat_sel=lat_array[date_sel]
time_sel=time_array[date_sel]
lon_sel=lon_array[date_sel]
samples=np.array([lat_sel, lon_sel, time_sel]).T
noon_time=datetime.datetime.combine(date_of_interest, datetime.time(12,0,0), tzinfo=datetime.timezone.utc) + np.array([datetime.timedelta(hours=h) for h in (-lon_sel / 360 * 24)])
time_diff=np.array([t.total_seconds()/3600 for t in (time_sel-noon_time)]) - 1.6

low_range=(time_diff < -5.5) & (time_diff > -6.5)
low_range_idx=np.arange(0, lat_sel.shape[0])[low_range]
max_local_idx=np.argmax(lat_sel[low_range])
low_time=time_diff[low_range_idx[max_local_idx]]

high_range=(time_diff > 5.5) & (time_diff < 6.5)
high_range_idx=np.arange(0, lat_sel.shape[0])[high_range]
min_local_idx=np.argmin(lat_sel[high_range])
high_time=time_diff[high_range_idx[min_local_idx]]
valid_time = (time_diff > low_time) & (time_diff < high_time)
f=interpolate.interp1d(lat_sel[valid_time], time_diff[valid_time], kind="linear")

np.save("../../data/processed/latitude_time_diff_sep_6_2014.npy", np.array([lat_sel[valid_time], time_diff[valid_time]]))
latitude_time_diff=np.load("../../data/processed/latitude_time_diff_sep_6_2014.npy")
