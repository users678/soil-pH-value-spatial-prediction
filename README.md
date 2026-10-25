# Soil pH value Spatial Prediction

## Overview

This repository contains the Python workflow for soil pH mapping using a stacking ensemble of Random Forest (RF) and Tabular Prior-data Fitted Network (TabPFN). Four stacking-based spatial modelling strategies are evaluated: projected coordinates, spatial lag features, regression kriging (RK), and geographically weighted regression (GWR). All models are evaluated using the same five outer kNNDM cross-validation folds linked to the farmland prediction domain.

## Requirements

- Python 3.11 and the libraries listed in `requirements.txt`.
- R with `CAST`, `sf`, and `jsonlite`; `Rscript` available on PATH.
- A local pretrained TabPFN-v2 regression checkpoint.

## Script Workflow

1. Prediction-domain adaptive cross-validation using kNNDM.
2. RF and TabPFN training, hyperparameter tuning, learned stacking, and nested OOF residual generation.
3. Spatial modelling using coordinates, spatial lag, RK, and GWR.
4. Model accuracy evaluation.
5. Observed-versus-predicted plots.
6. Farmland prediction mapping and empirical error-surface export.

## Usage

The soil sample data are not publicly available due to institutional restrictions. Input rasters, the farmland boundary, and model weights are not included in this repository.

Configure `DATA_PATH`, `RASTER_PATH`, `FARM_PATH`, and `TABPFN_MODEL_PATH` in `Code.py`, or set the corresponding environment variables. The sample covariate columns must match the raster bands in order. The covariate raster must use EPSG:32649 at 30 m resolution.

Install the required Python packages and run:

```bash
python -m pip install -r requirements.txt
python tougao.py
```

The workflow exports four pH prediction GeoTIFFs and four IDW-interpolated absolute outer-validation error GeoTIFFs. The error surfaces are descriptive indicators, not calibrated prediction intervals. Results are saved in `tougao_outputs`.
