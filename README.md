# A long-term reconstruction of a global photosynthesis proxy from surface reflectance over 1982-2023

## Overview
This repository contains the code and models used in the study titled **"A long-term reconstruction of spatially contiguous solar-induced fluorescence proxy from surface reflectance over 1982-2023"**. The goal of this project is to extend a SIF-informed global photosynthesis proxy to 1982, using a combination of Advanced Very High-Resolution Radiometer (AVHRR) and MODerate-resolution Imaging Spectroradiometer (MODIS) surface reflectance data. 

### Abstract of the manuscript:
Satellite-observed solar-induced chlorophyll fluorescence (SIF) is a powerful proxy for the photosynthetic characteristics of terrestrial ecosystems. Direct SIF observations are primarily limited to the recent decade, impeding their application in detecting long-term dynamics of ecosystem function. In this study, we leverage two surface reflectance bands available both from Advanced Very High-Resolution Radiometer (AVHRR, 1982-2023) and MODerate-resolution Imaging Spectroradiometer (MODIS, 2001-2023). Importantly, we calibrate and orbit-correct the AVHRR bands against their MODIS counterparts during their overlapping period.

Using the long-term bias-corrected reflectance data from AVHRR and MODIS, a neural network is trained to produce a Long-term Continuous SIF-informed Photosynthesis Proxy (LCSPP) by emulating Orbiting Carbon Observatory-2 SIF, mapping it globally over the 1982-2023 period. Compared with previous SIF-informed photosynthesis proxies, LCSPP has similar skill but can be advantageously extended to the AVHRR period. Further comparison with three widely used vegetation indices (NDVI, kNDVI, NIRv) shows a higher or comparable correlation of LCSPP with satellite SIF and site-level GPP estimates across vegetation types, ensuring a greater capacity for representing long-term photosynthetic activity. 

## Repository Structure

- **scripts/**: Contains the code for AVHRR reflectance calibration against MODIS.
- **notebooks/**: Contains Jupyter notebooks for training the reconstructed SIF model, analyzing the results, and generating figures for the manuscript.

## Data Access
The primary outputs of this study are long-term reconstructed solar-induced fluorescence proxies (LCSPP-AVHRR) from 1982-2022. The following versions of the data are available:

1. **LCSPP-AVHRR (v3.2)**:
   - 1982-2000: [Zenodo Link](https://doi.org/10.5281/zenodo.7916850)
   - 2001-2023: [Zenodo Link](https://doi.org/10.5281/zenodo.11906675)


2. **LCREF-AVHRR**: Long-term continuous reflectance record from AVHRR (1982-2023):
   - [Zenodo Link](https://doi.org/10.5281/zenodo.11905959)
   - Users can compute red/NIR-based vegetation indices such as NDVI, kNDVI, and NIRv from this dataset.

3. **LCSPP-MODIS**:
   - 2001-2023: [Zenodo Link](https://doi.org/10.5281/zenodo.11658088)

4. **LCREF-MODIS**:
   - 2001-2023: [Zenodo Link](https://doi.org/10.5281/zenodo.11657458)

### Data Format
- All datasets are provided at a 0.05° spatial resolution and biweekly temporal resolution in NetCDF format. Each month is split into two files:
  - File "a": Days 1-15
  - File "b": Days 16-end of the month
 
### Usage caveat:
We note that LCSPP is a SIF-informed photosynthesis proxy and should not be treated as SIF measurements. Although LCSPP has demonstrated skill in tracking the dynamics of GPP and PAR absorbed by canopy chlorophyll (APARchl), it is not suitable for estimating fluorescence quantum yield.

## Citation
If you use this repository or the accompanying datasets, please cite the associated manuscript:  
**"Fang, J., Lian, X., Ryu, Y., Jeong, S., Jiang, C., & Gentine, P. (2023). Reconstruction of a long-term spatially contiguous solar-induced fluorescence (LCSPP) over 1982-2022. arXiv preprint arXiv:2311.14987."**
