import sys
sys.path.insert(1, "AVHRR/script/util/")
from HANTS import HANTS

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from tqdm import tqdm

ML_CORRECTED = "AVHRR/data/AVHRR_ML_CORRECTED_v3.2/"
GAPFILLED = "AVHRR/data/AVHRR_GAPFILLED_v3.2/"

# Divide into memory if the entire dataset cannot fit into memory
TOTAL_REGIONS = 60
SINGLE_HEIGHT = 3000 // TOTAL_REGIONS

IDX = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1


if IDX > TOTAL_REGIONS-1:
    ref_var = "red"
    rid = IDX - TOTAL_REGIONS
else:
    ref_var = "nir"
    rid = IDX
    
# Obtain the filenames for all the input datasets
yearmonth_list = []

for year in np.arange(1982, 2024):
    for month in np.arange(1, 13):
        for half_month_index in ["a", "b"]:
            yearmonth_list.append(str(year) + '{:0>2}'.format(month) + half_month_index)

ref = np.full((len(yearmonth_list), SINGLE_HEIGHT, 7200), np.nan, dtype=np.float32)
hants_filled = np.zeros((len(yearmonth_list), SINGLE_HEIGHT, 7200), dtype=np.uint8)

for i in range(len(yearmonth_list)):
    if os.path.exists(os.path.join(ML_CORRECTED, "ml_{}_{}.npy".format(ref_var, yearmonth_list[i]))):
        ref_ar = np.load(os.path.join(ML_CORRECTED, "ml_{}_{}.npy".format(ref_var, yearmonth_list[i])))
        ref[i, :, :] = ref_ar[rid*SINGLE_HEIGHT:(rid+1)*SINGLE_HEIGHT, :]
        
# patch invalid period
valid_period = np.load("../../data/valid_period.npy")
invalid_idx = np.arange(len(valid_period))[np.invert(valid_period)]
ref[invalid_idx, :, :] = np.nanmean(np.concatenate([np.expand_dims(ref[invalid_idx - 24, :, :], axis=0), np.expand_dims(ref[invalid_idx + 24, :, :], axis=0)]), axis=0)


ni=ref.shape[0]
nb = ni
nf = 2 * ni / 24
fet = 0.1
dod = 1
delta = 0.1
HiLo = "none"

for i in range(ref.shape[1]):
    for j in range(ref.shape[2]):
        x = deepcopy(ref[:, i, j])
        if np.sum(np.isnan(x)) > 0 and np.sum(np.isfinite(x)) > 0:
            try:
                x_org = deepcopy(x)
                low = np.nanpercentile(x, 5)
                high = np.nanpercentile(x, 95)

                x[np.isnan(x)] = -1
                amp, phi, t  = HANTS(ni,nb,nf,x,np.arange(1, ni+1),HiLo,low,high,fet,dod,delta)
                t[t < low] = np.nan
                t[t > high] = np.nan
                t[t==0] = np.nan
                ref[x == -1, i, j] = np.squeeze(t[x==-1])
                invalid_idx = np.where(x == -1)[0]
                hants_filled[invalid_idx[np.invert(np.isnan(np.squeeze(t[x==-1])))], i, j] = 1
            except:
                pass

            msk = np.isnan(ref[:, i, j])
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
            ref[msk, i, j] = tmp[msk]
            msk_indices = np.where(msk)[0]
            hants_filled[msk_indices[np.invert(np.isnan(tmp[msk]))], i, j]=2
            hants_filled[np.where(np.isnan(ref[:, i, j]))[0], i, j]=3
        else:
            hants_filled[:, i, j]=3

for i in range(len(yearmonth_list)):
    np.save(os.path.join(GAPFILLED, "gf_{}_{}_{:0>2}.npy".format(ref_var, yearmonth_list[i], rid)), ref[i, :, :])
    np.save(os.path.join(GAPFILLED, "hf_{}_{}_{:0>2}.npy".format(ref_var, yearmonth_list[i], rid)), hants_filled[i, :, :])