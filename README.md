# A long-term reconstruction of a global photosynthesis proxy from surface reflectance over 1982-2023

## Overview
This repository contains the code and models used in the study titled **"A long-term reconstruction of spatially contiguous solar-induced fluorescence proxy from surface reflectance over 1982-2022"**. The goal of this project is to extend a proxy of satellite-observed solar-induced chlorophyll fluorescence (SIF) back to 1982, using a combination of Advanced Very High-Resolution Radiometer (AVHRR) and MODerate-resolution Imaging Spectroradiometer (MODIS) surface reflectance data. 

### Abstract of the manuscript:
Satellite-observed solar-induced chlorophyll fluorescence (SIF) is a powerful proxy for the photosynthetic characteristics of terrestrial ecosystems. However, direct SIF observations are primarily limited to the recent decade, which limits their application in detecting long-term ecosystem dynamics. In this study, we leveraged two surface reflectance bands available from both AVHRR (1982-2022) and MODIS (2001-2022). We calibrated and orbit-corrected the AVHRR bands against their MODIS counterparts during the overlapping period (2001-2022).

Using long-term bias-corrected reflectance data, a neural network model was developed to reconstruct a proxy of the Orbiting Carbon Observatory-2 (OCO-2) SIF. This model allows us to map the global proxy, termed LCSIF, over the entire 1982-2022 period. Compared to other SIF-informed photosynthetic proxies, LCSIF shows similar skill but with the advantage of extension to the AVHRR period. LCSIF serves as a consistent indicator of photosynthetically active radiation absorbed by chlorophyll (APARchl). Additionally, it shows higher or comparable correlation with satellite SIF and site-level GPP estimates compared to vegetation indices like NDVI, kNDVI, and NIRv, across various vegetation types.

## Repository Structure

- **scripts/**: Contains the code for AVHRR reflectance calibration against MODIS.
- **notebooks/**: Contains Jupyter notebooks for training the reconstructed SIF model, analyzing the results, and generating figures for the manuscript.

## Data Access
The primary outputs of this study are long-term reconstructed solar-induced fluorescence proxies (LCSIF-AVHRR) from 1982-2022. The following versions of the data are available:

1. **LCSIF-AVHRR (v3.1)**:
   - 1982-2000: [Zenodo Link](https://doi.org/10.5281/zenodo.13922371)
   - 2001-2022: [Zenodo Link](https://doi.org/10.5281/zenodo.13922367)


2. **LCREF-AVHRR**: Long-term continuous reflectance record from AVHRR (1982-2022):
   - [Zenodo Link](https://doi.org/10.5281/zenodo.11905960)
   - Users can compute red/NIR-based vegetation indices such as NDVI, kNDVI, and NIRv from this dataset.

3. **LCSIF-MODIS**:
   - 2001-2022: [Zenodo Link](https://doi.org/10.5281/zenodo.13922379)

4. **LCREF-MODIS**:
   - 2001-2022: [Zenodo Link](https://doi.org/10.5281/zenodo.11657459)

### Data Format
- All datasets are provided at a 0.05° spatial resolution and biweekly temporal resolution in NetCDF format. Each month is split into two files:
  - File "a": Days 1-15
  - File "b": Days 16-end of the month
 
### Usage caveat:
We note that LCSIF is a reconstructed SIF proxy and should not be treated as SIF measurements. Although LCSIF has demonstrated skill in tracking the dynamics of GPP and PAR absorbed by canopy chlorophyll (APARchl), it is not suitable for estimating fluorescence quantum yield.

## Citation
If you use this repository or the accompanying datasets, please cite the associated manuscript:  
**"Fang, J., Lian, X., Ryu, Y., Jeong, S., Jiang, C., & Gentine, P. (2023). Reconstruction of a long-term spatially contiguous solar-induced fluorescence (LCSIF) over 1982-2022. arXiv preprint arXiv:2311.14987."**
