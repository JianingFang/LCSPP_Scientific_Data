import sys
sys.path.insert(1, "AVHRR/script/util/")
from HANTS import HANTS

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tqdm import tqdm

GAPFILLED = "AVHRR/data/SZA_GAPFILLED/"
AVHRR_MVC_DIR = "AVHRR/data/AVHRR_MVC/"


# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 60
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS

IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1

rid = IDX
    
# Obtain the filenames for all the input datasets
AVHRR_MVC_files = []
yearmonth_list = []       


for year in np.arange(1982, 2014):
    for month in np.arange(1, 13):
        for half_month_index in ["a", "b"]:
            AVHRR_MVC_files.append(os.path.join(AVHRR_MVC_DIR, "AVHRR_MVC_" + str(year) + '{:0>2}'.format(month) + half_month_index + ".nc"))
            yearmonth_list.append(str(year) + '{:0>2}'.format(month) + half_month_index)


sza = np.full((len(AVHRR_MVC_files), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)

for i in range(len(AVHRR_MVC_files)):
    mvc_ds = xr.open_dataset(AVHRR_MVC_files[i])  
    sza[i, :, :] = mvc_ds["sza"].values[0, rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]

ni=sza.shape[0]
nb = ni
nf = 2 * ni / 24
fet = 0.1
dod = 1
delta = 0.1
HiLo = "none"

for i in range(sza.shape[1]):
    for j in range(sza.shape[2]):
        x = deepcopy(sza[:, i, j])
        if np.sum(np.isnan(x)) > 0 and np.sum(np.isfinite(x)) > 0:
            x_org = deepcopy(x)
            low = np.nanpercentile(x, 5)
            high = np.nanpercentile(x, 95)
        
            x[np.isnan(x)] = -1
            amp, phi, t  = HANTS(ni,nb,nf,x,np.arange(1, ni+1),HiLo,low,high,fet,dod,delta)
            t[t < low] = np.nan
            t[t > high] = np.nan
            t[t==0] = np.nan
            sza[x == -1, i, j] = np.squeeze(t[x==-1])
            msk = np.isnan(sza[:, i, j])
            x_org[x_org < low] = np.nan
            x_org[x_org > high] = np.nan
            
            tmp = np.nanmean(x_org.reshape(len(x_org)//24, -1), axis=0)
            tmptmp = np.tile(tmp, 3)
            
            for s in np.arange(24, 48):
                if np.isnan(tmp[s-24]):
                    for ss in np.arange(1,13):
                        meanv = np.nanmean(tmptmp[s-ss:s+ss+1])
                        if not np.isnan(meanv):
                            tmp[s-24] = meanv
            tmp = np.tile(tmp, len(x_org)//24)
            sza[msk, i, j] = tmp[msk]

for i in range(len(yearmonth_list)):
    np.save(os.path.join(GAPFILLED, "gf_{}_{}_{:0>2}.npy".format("sza", yearmonth_list[i], rid)), sza[i, :, :])