import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import contextlib
import hashlib
import json
import shutil
import subprocess
import warnings
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'tougao_outputs' / 'matplotlib_cache'))
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.compose import ColumnTransformer
from scipy.stats import pearsonr, wasserstein_distance, ks_2samp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from pyproj import Transformer
import rasterio
import fiona
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely.geometry import shape, box
from shapely.validation import make_valid
from libpysal.weights import KNN
from esda.moran import Moran
from tabpfn import TabPFNRegressor

PROJECT_DIR = Path(__file__).resolve().parent
SUBMISSION_OUTPUT_DIR = PROJECT_DIR / 'tougao_outputs'
FROZEN_FOLDS_DIR = PROJECT_DIR / 'tougao_support' / 'frozen_knndm'

# 1. Prediction-Domain Adaptive Cross-Validation
# 1.1 Read Samples and Define the Farmland Prediction Domain
DATA_PATH = os.environ.get('DATA_PATH', str(PROJECT_DIR / 'data' / 'samples.csv'))
RASTER_PATH = os.environ.get('RASTER_PATH', str(PROJECT_DIR / 'data' / 'covariates.tif'))
FARM_PATH = os.environ.get('FARM_PATH', str(PROJECT_DIR / 'data' / 'farmland.geojson'))
OUTPUT_DIR = SUBMISSION_OUTPUT_DIR / 'validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_FOLDS = 5
RANDOM_SEED = 42
RSCRIPT_PATH = os.environ.get('RSCRIPT_PATH') or shutil.which('Rscript') or 'Rscript'

full_data = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
if full_data.columns.duplicated().any():
    raise ValueError('Input column aliases produced duplicate column names.')
required_cols = ['FID', 'Lon', 'Lat', 'pH']
if not set(required_cols).issubset(full_data.columns) or not full_data['FID'].is_unique:
    raise ValueError('Required sample columns are missing or FID is duplicated.')
if not np.isfinite(full_data[['Lon', 'Lat', 'pH']].to_numpy(dtype=float)).all():
    raise ValueError('Invalid coordinates or pH. Do not impute coordinates or the response.')
excluded_cols = required_cols + ['Spatial_Cluster', 'CV_Fold', 'New_FID', 'source']
feature_cols = [col for col in full_data.columns if col not in excluded_cols]
full_data[feature_cols] = full_data[feature_cols].apply(pd.to_numeric, errors='raise')
predictor_values = full_data[feature_cols].to_numpy(dtype=float)
missing_markers = (predictor_values == -9999) | (np.abs(predictor_values) > 1e15)
full_data[feature_cols] = np.where(missing_markers, np.nan, predictor_values)
X_all = full_data[feature_cols].to_numpy(dtype=float)
y_all = full_data['pH'].to_numpy(dtype=float)
if np.isinf(X_all).any() or np.isnan(X_all).all(axis=0).any():
    raise ValueError('Infinite predictors or a completely missing predictor.')

transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32649', always_xy=True)
coords_all_proj = np.column_stack(transformer.transform(full_data['Lon'].values, full_data['Lat'].values))
sample_points = full_data[['FID', 'Lon', 'Lat']].copy()
sample_points.insert(0, 'row_index', np.arange(len(full_data)))
sample_points['x'], sample_points['y'] = coords_all_proj.T
sample_points.to_csv(OUTPUT_DIR / 'sample_points.csv', index=False)

# 1.2 Project Farmland Polygons and Extract Farmland Pixel Centres
def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        yield geometry
    else:
        for part in getattr(geometry, 'geoms', []):
            yield from polygon_parts(part)

with rasterio.open(RASTER_PATH) as src, fiona.open(FARM_PATH) as farm:
    if src.crs is None or src.crs.to_epsg() != 32649 or not farm.crs:
        raise ValueError('Expected EPSG:32649 covariate grid and a known farmland CRS.')
    if not np.allclose(src.res, (30, 30)):
        raise ValueError('Expected the original 30 m covariate grid.')
    geometries = []
    repaired_n = 0
    for feature in farm:
        if feature['geometry'] is None:
            continue
        geom = shape(transform_geom(farm.crs_wkt, src.crs, feature['geometry']))
        if not geom.is_valid:
            geom = make_valid(geom)
            repaired_n += 1
        geometries.extend(polygon_parts(geom))
    if not geometries or not all(box(*src.bounds).covers(g) for g in geometries):
        raise ValueError('Empty farmland geometry or farmland extends outside the covariate grid.')
    farmland_mask = rasterize(((g, 1) for g in geometries), out_shape=src.shape,
                             transform=src.transform, fill=0, all_touched=False, dtype='uint8')
    rows, cols = np.where(farmland_mask == 1)
    if not len(rows):
        raise ValueError('No farmland prediction pixels.')
    pixel_x, pixel_y = rasterio.transform.xy(src.transform, rows, cols, offset='center')
    prediction_coords = np.column_stack([pixel_x, pixel_y])
    prediction_points = pd.DataFrame(prediction_coords, columns=['x', 'y'])
    prediction_points.to_csv(OUTPUT_DIR / 'prediction_points.csv', index=False)
    grid_metadata = {'crs': str(src.crs), 'resolution_m': list(src.res),
                     'height': src.height, 'width': src.width, 'transform': list(src.transform)}

# 1.3 Calculate the Target Nearest-Neighbour Distance Distribution (A)
target_distances = cKDTree(coords_all_proj).query(prediction_coords)[0]
print(f'Prediction domain: {len(prediction_coords):,} original farmland pixels; {len(full_data)} samples.', flush=True)

# 1.4 Run CAST kNNDM Using Farmland Pixel Centres as Prediction Locations
r_code = r'''
suppressPackageStartupMessages(library(CAST)) 
suppressPackageStartupMessages(library(sf)) 
suppressPackageStartupMessages(library(jsonlite))  
args <- commandArgs(trailingOnly=TRUE)  
out <- args[1]; k <- as.integer(args[2]); seed <- as.integer(args[3]) 
samples <- read.csv(file.path(out, 'sample_points.csv'))  
target <- read.csv(file.path(out, 'prediction_points.csv'))  
sample_sf <- st_as_sf(samples, coords=c('x','y'), crs=32649, remove=FALSE) 
target_sf <- st_as_sf(target, coords=c('x','y'), crs=32649) 
set.seed(seed)  
folds <- CAST::knndm(tpoints=sample_sf, predpoints=target_sf,  
                     space='geographical', k=k, clustering='hierarchical',  
                     linkf='ward.D2', maxp=0.5) 
saveRDS(folds, file.path(out, 'cast_knndm.rds'))  
fold_id <- integer(nrow(samples)) 
for (i in seq_along(folds$indx_test)) {  
  tr <- folds$indx_train[[i]]; va <- folds$indx_test[[i]]  
  stopifnot(length(intersect(tr,va))==0,  
            setequal(c(tr,va),seq_len(nrow(samples))), all(fold_id[va]==0))  
  fold_id[va] <- i 
}  
stopifnot(all(fold_id>0), length(unique(fold_id))==k)  
samples$fold_id <- fold_id  
write.csv(samples, file.path(out,'knndm_folds.csv'), row.names=FALSE)  
write_json(list(CAST_version=as.character(packageVersion('CAST')),  
                W=folds$W, q=folds$q, k=k, seed=seed,  
                space=folds$space, clustering='hierarchical',linkf='ward.D2',maxp=0.5),  
           file.path(out,'cast_metadata.json'),auto_unbox=TRUE,pretty=TRUE) 
capture.output(print(folds),file=file.path(out,'cast_summary.txt')) 
capture.output(sessionInfo(),file=file.path(out,'R_session_info.txt'))  
print(folds)  
'''
r_script = OUTPUT_DIR / 'generate_knndm.R'
r_script.write_text(r_code, encoding='utf-8')

if not (OUTPUT_DIR / 'knndm_folds.csv').exists() and FROZEN_FOLDS_DIR.is_dir():
    frozen_required = ['knndm_folds.csv', 'cast_metadata.json', 'distance_distributions.npz']
    if not all((FROZEN_FOLDS_DIR / name).is_file() for name in frozen_required):
        raise ValueError('Incomplete bundled CAST fold records.')
    for name in frozen_required + ['cast_knndm.rds', 'cast_summary.txt', 'R_session_info.txt']:
        source_record = FROZEN_FOLDS_DIR / name
        if source_record.is_file():
            shutil.copy2(source_record, OUTPUT_DIR / name)

if (OUTPUT_DIR / 'knndm_folds.csv').exists():
    frozen = pd.read_csv(OUTPUT_DIR / 'knndm_folds.csv')
    frozen_meta = json.loads((OUTPUT_DIR / 'cast_metadata.json').read_text())
    previous_A = np.load(OUTPUT_DIR / 'distance_distributions.npz')['target_A']
    if (not np.array_equal(frozen.FID.to_numpy(), full_data.FID.to_numpy())
            or not np.array_equal(frozen.row_index.to_numpy(), np.arange(len(full_data)))
            or not np.allclose(frozen[['x','y']].to_numpy(), coords_all_proj, rtol=0, atol=1e-6)
            or frozen_meta['k'] != N_FOLDS or frozen_meta['seed'] != RANDOM_SEED
            or not np.array_equal(previous_A, target_distances)):
        raise ValueError('Frozen kNNDM data/domain mismatch; do not silently replace the comparison folds.')
    print('Reusing the verified, frozen official CAST farmland kNNDM five folds.', flush=True)
else:
    print('Calling CAST::knndm with the farmland prediction points; optimizing fold configuration...', flush=True)
    subprocess.run([RSCRIPT_PATH, str(r_script), str(OUTPUT_DIR), str(N_FOLDS), str(RANDOM_SEED)], check=True)

# 1.5 Retrieve and Verify the Five Outer Training and Validation Splits
fold_table = pd.read_csv(OUTPUT_DIR / 'knndm_folds.csv')
if not np.array_equal(fold_table.FID.to_numpy(), full_data.FID.to_numpy()) or not np.array_equal(fold_table.row_index, np.arange(len(full_data))):
    raise ValueError('kNNDM sample identity/order mismatch.')
fold_ids = fold_table.fold_id.to_numpy(dtype=int)
cv_splits = []
validation_counts = np.zeros(len(full_data), dtype=int)
split_records = []
for fold_id in range(1, N_FOLDS+1):
    train_idx = np.flatnonzero(fold_ids != fold_id)
    test_idx = np.flatnonzero(fold_ids == fold_id)
    if not len(train_idx) or not len(test_idx) or np.intersect1d(train_idx, test_idx).size:
        raise ValueError('Empty fold or training/test overlap.')
    cv_splits.append((train_idx, test_idx))
    validation_counts[test_idx] += 1
    split_records.append({'fold_id': fold_id, 'train_row_index': train_idx.tolist(), 'test_row_index': test_idx.tolist()})
if not np.all(validation_counts == 1):
    raise ValueError('Every sample must be tested once in outer CV.')
(OUTPUT_DIR / 'cv_splits.json').write_text(json.dumps(split_records, indent=2))
pd.DataFrame([{'fold_id': record['fold_id'], 'train_n': len(record['train_row_index']),
               'test_n': len(record['test_row_index'])} for record in split_records]).to_csv(OUTPUT_DIR / 'fold_sizes.csv', index=False)
full_data_with_folds = full_data.copy()
full_data_with_folds['CV_Fold'] = fold_ids
full_data_with_folds.to_csv(OUTPUT_DIR / 'samples_with_folds.csv', index=False)

# 1.6 Independently Verify Distance Matching and Export Diagnostics
validation_distances = np.full(len(full_data), np.nan)
for train_idx, test_idx in cv_splits:
    validation_distances[test_idx] = cKDTree(coords_all_proj[train_idx]).query(coords_all_proj[test_idx])[0]
if not np.isfinite(validation_distances).all():
    raise ValueError('Incomplete validation distance distribution.')
W = float(wasserstein_distance(target_distances, validation_distances))
cast_meta = json.loads((OUTPUT_DIR / 'cast_metadata.json').read_text())
if not np.isclose(W, cast_meta['W'], rtol=1e-6, atol=1e-3):
    raise ValueError('Independent Python W does not agree with CAST W.')
sample_points['fold_id'] = fold_ids
sample_points['validation_to_training_NND_m'] = validation_distances
sample_points.to_csv(OUTPUT_DIR / 'sample_validation_distances.csv', index=False)
np.savez_compressed(OUTPUT_DIR / 'distance_distributions.npz', target_A=target_distances, validation_B=validation_distances)
summary_rows = []
for label, values in [('Farmland prediction A', target_distances), ('kNNDM validation B', validation_distances)]:
    summary_rows.append({'distribution': label, 'n': len(values), 'min_m': float(values.min()),
                         'mean_m': float(values.mean()), 'median_m': float(np.median(values)),
                         'p90_m': float(np.quantile(values,.9)), 'p95_m': float(np.quantile(values,.95)), 'max_m': float(values.max())})
pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / 'distance_summary.csv', index=False)
diagnostics = {'Wasserstein_m': W, 'CAST_W_m': cast_meta['W'],
               'KS_D': float(ks_2samp(target_distances, validation_distances).statistic),
               'target_fraction_below_min_B': float(np.mean(target_distances < validation_distances.min())),
               'target_fraction_above_max_B': float(np.mean(target_distances > validation_distances.max())),
               'data_path': Path(DATA_PATH).name, 'data_sha256': hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest(),
               'farm_path': Path(FARM_PATH).name, 'raster_path': Path(RASTER_PATH).name, 'grid': grid_metadata,
               'prediction_pixels': len(prediction_coords), 'geometry_repairs_in_memory': repaired_n,
               'domain': 'unbuffered farmland; pixel-centre inclusion; all farmland pixels regardless of covariate missingness',
               'samples_moved_or_removed': False, 'target_subsampled': False,
               'environmental_coverage': 'Not established by geographical kNNDM; requires separate verified covariate diagnosis',
               'model_evaluation': 'Outer-fold evaluation follows in Sections 2–4'}
(OUTPUT_DIR / 'matching_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
fig, ax = plt.subplots(figsize=(8,5))
for label, values in [('Farmland pixels to all samples (A)', target_distances), ('Outer test to training samples (B)', validation_distances)]:
    ordered = np.sort(values)
    ax.plot(ordered, np.arange(1,len(ordered)+1)/len(ordered), label=label)
ax.set(xlabel='Nearest-neighbour distance (m)', ylabel='Empirical cumulative probability', title=f'Farmland kNNDM matching: W = {W:.2f} m')
ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'distance_matching_ECDF.png', dpi=220); plt.close(fig)
fig, ax = plt.subplots(figsize=(8,6))
stride = max(1,len(prediction_coords)//30000)
ax.scatter(prediction_coords[::stride,0], prediction_coords[::stride,1], s=.3, c='lightgrey', label='Farmland pixels (plot subset)')
for fold_id in range(1,N_FOLDS+1):
    group = fold_ids == fold_id
    ax.scatter(coords_all_proj[group,0], coords_all_proj[group,1], s=22, label=f'Fold {fold_id} (n={group.sum()})')
ax.set(xlabel='Easting (m)', ylabel='Northing (m)', title='Farmland prediction domain and kNNDM folds')
ax.set_aspect('equal'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / 'farmland_knndm_folds.png', dpi=220); plt.close(fig)
print(pd.read_csv(OUTPUT_DIR / 'fold_sizes.csv').to_string(index=False), flush=True)
print(f'kNNDM matching verified: Python W={W:.4f} m; CAST W={cast_meta["W"]:.4f} m.', flush=True)
print(f'Section 1 completed. Five zero-based (train_idx, test_idx) pairs are in cv_splits; outputs: {OUTPUT_DIR}', flush=True)


# 2. Base Learners, Hyperparameter Tuning and Nested Stacking
# 2.0 Nested Cross-Validation Settings
MODEL_OUTPUT_DIR = SUBMISSION_OUTPUT_DIR / 'models'
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABPFN_MODEL_PATH = os.environ.get('TABPFN_MODEL_PATH', str(PROJECT_DIR / 'weights' / 'tabpfn-v2-regressor.ckpt'))
INNER_FOLDS = 3
STACKING_OOF_FOLDS = 5
META_SELECTION_FOLDS = 3
RESIDUAL_OOF_FOLDS = 5
META_ALPHAS = [.001, .01, .1, 1., 3., 10., 30., 100., 1000.]
TABPFN_ESTIMATORS = 4
RF_ESTIMATORS = 400
BASE_NAMES = ['RF', 'TabPFN']

# 2.1 RF and TabPFN Specifications and Candidate Hyperparameters
def base_model_specs(n_samples, with_coordinates=False):
    if with_coordinates == 'spatial_lag':
        plain = base_model_specs(n_samples, with_coordinates=False)
        wrapped = {}
        for name, (model, grid) in plain.items():
            if grid is not None:
                grids = grid if isinstance(grid, list) else [grid]
                grids = [{f'estimator__{key}': values for key, values in item.items()} for item in grids]
                grid = grids if isinstance(grid, list) else grids[0]
            wrapped[name] = (TrainingNeighbourRegressor(model, n_neighbors=8), grid)
        return wrapped
    if n_samples < INNER_FOLDS:
        raise ValueError('Too few training observations for inner tuning.')

    def preprocessing():
        steps = [SimpleImputer(strategy='mean', keep_empty_features=True)]
        if with_coordinates:
            steps.append(ColumnTransformer([
                ('environment', 'passthrough', slice(0, -2)),
                ('coordinates', StandardScaler(), [-2, -1])]))
        return steps

    rf_model = make_pipeline(*preprocessing(), RandomForestRegressor(
        n_estimators=RF_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=2))
    tab_model = make_pipeline(*preprocessing(), TabPFNRegressor(
        model_path=TABPFN_MODEL_PATH, device='cpu', n_estimators=TABPFN_ESTIMATORS,
        random_state=RANDOM_SEED, n_preprocessing_jobs=1, fit_mode='fit_preprocessors'))
    return {
        'RF': (rf_model, {'randomforestregressor__max_depth': [6, None],
                         'randomforestregressor__min_samples_leaf': [2, 5],
                         'randomforestregressor__max_features': [.5, 1.]}),
        'TabPFN': (tab_model, None)}

# 2.1.1 Select Hyperparameters within Each Training Subset
def tune_base_models(x_train, y_train, with_coordinates=False):
    fitted = {}; records = {}
    specs = base_model_specs(len(y_train), with_coordinates=with_coordinates)
    for name in BASE_NAMES:
        estimator, grid = specs[name]
        if grid is None:
            fitted[name] = estimator.fit(x_train, y_train)
            records[name] = {'procedure': 'prespecified pretrained checkpoint',
                             'inference_ensemble_size': TABPFN_ESTIMATORS}
        else:
            search = GridSearchCV(estimator, grid,
                                  cv=KFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                                  scoring='neg_mean_squared_error', refit=True,
                                  n_jobs=1, error_score='raise')
            search.fit(x_train, y_train)
            fitted[name] = search.best_estimator_
            records[name] = {'best_params': search.best_params_,
                             'inner_selection_RMSE': float(np.sqrt(-search.best_score_)),
                             'candidate_count': len(search.cv_results_['params'])}
    return fitted, records

# 2.2 Meta-Learning Stacking Using Base-Model OOF Predictions
class LearnedStackingModel:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def predict(self, x):
        z = np.column_stack([self.base_models[name].predict(x) for name in BASE_NAMES])
        return self.meta_model.predict(z)

def meta_candidates():
    candidates = []
    for positive in [False, True]:
        for alpha in META_ALPHAS:
            candidates.append({'family': 'Ridge', 'alpha': alpha, 'positive': positive})
        for alpha in [.001, .01, .1, 1., 10.]:
            for ratio in [.1, .5, .9]:
                candidates.append({'family': 'ElasticNet', 'alpha': alpha,
                                   'positive': positive, 'l1_ratio': ratio})
    return candidates

def make_meta_estimator(candidate):
    if candidate['family'] == 'Ridge':
        estimator = Ridge(alpha=candidate['alpha'], positive=candidate['positive'],
                          solver='lbfgs' if candidate['positive'] else 'svd',
                          tol=1e-8, max_iter=10000)
    elif candidate['family'] == 'ElasticNet':
        estimator = ElasticNet(alpha=candidate['alpha'], l1_ratio=candidate['l1_ratio'],
                               positive=candidate['positive'], max_iter=100000, tol=1e-7,
                               selection='cyclic')
    else:
        raise ValueError('Unknown prespecified meta family.')
    return make_pipeline(StandardScaler(), estimator)

def build_base_oof(x_train, y_train, with_coordinates=False):
    if len(y_train) < STACKING_OOF_FOLDS:
        raise ValueError('Too few samples for the prespecified stacking OOF folds.')
    z = np.full((len(y_train), len(BASE_NAMES)), np.nan)
    ids = np.zeros(len(y_train), dtype=int)
    audit = []
    for inner_id, (itr, iva) in enumerate(KFold(STACKING_OOF_FOLDS, shuffle=True,
                                               random_state=RANDOM_SEED).split(x_train), 1):
        models, records = tune_base_models(x_train[itr], y_train[itr],
                                          with_coordinates=with_coordinates)
        z[iva] = np.column_stack([models[name].predict(x_train[iva]) for name in BASE_NAMES])
        ids[iva] = inner_id
        audit.append({'stacking_fold': inner_id, 'train_local_rows': itr.tolist(),
                      'validation_local_rows': iva.tolist(), 'tuning': records})
    if not np.isfinite(z).all():
        raise ValueError('Incomplete base OOF predictions.')
    return z, ids, audit

def fit_stacking_process(x_train, y_train, with_coordinates=False):
    candidates = meta_candidates()
    losses = np.zeros(len(candidates))
    selection_audit = []
    validation_n = 0
    for selection_id, (itr, iva) in enumerate(KFold(META_SELECTION_FOLDS, shuffle=True,
                                                   random_state=RANDOM_SEED+2).split(x_train), 1):
        z_train, _, base_audit = build_base_oof(x_train[itr], y_train[itr], with_coordinates)
        bases, full_audit = tune_base_models(x_train[itr], y_train[itr], with_coordinates)
        z_validation = np.column_stack([bases[name].predict(x_train[iva]) for name in BASE_NAMES])
        split_sse = []
        for i, candidate in enumerate(candidates):
            meta = make_meta_estimator(candidate).fit(z_train, y_train[itr])
            prediction = meta.predict(z_validation)
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite meta-candidate validation predictions.')
            sse = float(np.sum((prediction-y_train[iva])**2))
            losses[i] += sse
            split_sse.append(sse)
        validation_n += len(iva)
        selection_audit.append({'selection_fold': selection_id,
            'train_local_rows': itr.tolist(), 'validation_local_rows': iva.tolist(),
            'base_oof_tuning_within_selection_training': base_audit,
            'full_base_tuning_within_selection_training': full_audit,
            'candidate_validation_SSE': split_sse})
        print(f'  Whole-stack meta selection: {selection_id}/{META_SELECTION_FOLDS}', flush=True)
    selected_index = int(np.argmin(losses))
    selected = candidates[selected_index]
    z_oof, base_oof_fold, local_tuning_audit = build_base_oof(x_train, y_train, with_coordinates)
    meta_model = make_meta_estimator(selected).fit(z_oof, y_train)
    base_models, full_tuning = tune_base_models(x_train, y_train, with_coordinates)
    model = LearnedStackingModel(base_models, meta_model)
    scale = meta_model.named_steps['standardscaler']
    linear = meta_model.steps[-1][1]
    original_coefficients = linear.coef_ / scale.scale_
    original_intercept = linear.intercept_ - np.dot(original_coefficients, scale.mean_)
    audit = {'with_coordinates': with_coordinates,
             'base_tuning_full_training': full_tuning, 'base_oof_tuning': local_tuning_audit,
             'meta_family': selected['family'], 'meta_parameters': selected,
             'meta_alpha': selected['alpha'], 'meta_positive': selected['positive'],
             'meta_selection_folds': selection_audit,
             'meta_candidate_results': [dict(candidate, internal_RMSE=float(np.sqrt(loss/validation_n)))
                                        for candidate, loss in zip(candidates, losses)],
             'meta_coefficients_original_prediction_scale': dict(zip(BASE_NAMES, original_coefficients.tolist())),
             'meta_intercept_original_prediction_scale': float(original_intercept),
             'meta_selection_score_note': 'Training-only whole-stack selection, not external map accuracy',
             'stacking_OOF_folds': STACKING_OOF_FOLDS,
             'OOF_training_size_note': 'Five-fold OOF reduces but does not eliminate the training-size difference'}
    return model, z_oof, base_oof_fold, audit

def show_stacking_coefficients(audit, outer_fold, variant):
    coefficients = audit['meta_coefficients_original_prediction_scale']
    intercept = audit['meta_intercept_original_prediction_scale']
    terms = ' '.join(f'{coefficients[name]:+.4f} × {name}' for name in BASE_NAMES)
    print(f'  Fold {outer_fold} {variant}: pH = {intercept:.4f} {terms}', flush=True)
    print(f'  Meta {audit["meta_family"]}: {audit["meta_parameters"]}; coefficients are not percentages.', flush=True)
    return {'outer_fold': outer_fold, 'variant': variant, 'meta_family': audit['meta_family'],
            'meta_alpha': audit['meta_alpha'], 'meta_positive': audit['meta_positive'],
            'meta_l1_ratio': audit['meta_parameters'].get('l1_ratio'),
            'intercept': intercept, **{f'coefficient_{name}': coefficients[name] for name in BASE_NAMES}}

def stacking_oof_residuals(x_train, y_train):
    prediction = np.full(len(y_train), np.nan)
    residual_fold = np.zeros(len(y_train), dtype=int)
    audit = []
    for inner_id, (itr, iva) in enumerate(KFold(RESIDUAL_OOF_FOLDS, shuffle=True,
                                               random_state=RANDOM_SEED+1).split(x_train), 1):
        model, _, _, record = fit_stacking_process(x_train[itr], y_train[itr])
        prediction[iva] = model.predict(x_train[iva])
        residual_fold[iva] = inner_id
        audit.append({'residual_fold': inner_id, 'train_local_rows': itr.tolist(),
                      'validation_local_rows': iva.tolist(), 'stacking': record})
        print(f'  Entire-process OOF residuals: {inner_id}/{RESIDUAL_OOF_FOLDS}', flush=True)
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete entire-stack OOF residual predictions.')
    return prediction, y_train-prediction, residual_fold, audit

# 2.3 Spatial Diagnostics of Training Residuals
def diagnose_training_residuals(coords_train, residuals):
    records = []
    for k_now in [4, 6, 8, 10, 12]:
        if k_now >= len(residuals):
            continue
        weights = KNN.from_array(coords_train, k=k_now)
        weights.transform = 'R'
        np.random.seed(RANDOM_SEED+k_now)
        result = Moran(residuals, weights, permutations=999)
        records.append({'k': k_now, 'Moran_I': float(result.I), 'permutation_p': float(result.p_sim)})
    return records

# 2.4 Fit Models within Each Outer Training Fold
def fit_outer_base_fold(x_train, y_train, x_test):
    ensemble_model, base_oof, base_oof_ids, stacking_audit = fit_stacking_process(x_train, y_train)
    base_predictions = {name: ensemble_model.base_models[name].predict(x_test) for name in BASE_NAMES}
    base_predictions['Stacking'] = ensemble_model.predict(x_test)
    y_ensemble_train_oof, residuals_train, residual_ids, residual_audit = stacking_oof_residuals(x_train, y_train)
    return {'predictions': base_predictions, 'base_oof': base_oof, 'base_oof_ids': base_oof_ids,
            'stacking_audit': stacking_audit, 'training_stack_oof': y_ensemble_train_oof,
            'residuals_train': residuals_train, 'residual_ids': residual_ids,
            'residual_audit': residual_audit}

# 3. Spatial Modelling Strategies
SPATIAL_NAMES = ['RF_coordinates', 'TabPFN_coordinates', 'Stacking_coordinates',
                 'RF_spatial_lag', 'TabPFN_spatial_lag', 'Stacking_spatial_lag',
                 'Stacking_RK', 'Stacking_GWR']
GWR_MIN_NEIGHBOURS = 60

# 3.1 Projected Coordinates as Predictors
def fit_coordinate_stacking(x_train, y_train, coords_train, x_test, coords_test):
    train_augmented = np.column_stack([x_train, coords_train])
    test_augmented = np.column_stack([x_test, coords_test])
    model, _, _, audit = fit_stacking_process(train_augmented, y_train, with_coordinates=True)
    prediction = model.predict(test_augmented)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite coordinate-stacking prediction.')
    audit['base_validation_predictions'] = {name: model.base_models[name].predict(test_augmented).tolist()
                                            for name in ['RF', 'TabPFN']}
    audit['coordinate_CRS'] = 'EPSG:32649'
    audit['coordinate_scaling'] = 'Fit within every training/tuning/OOF subset'
    audit['predictor_names'] = feature_cols + ['Easting_m', 'Northing_m']
    return prediction, audit


# 3.2 Spatial Lag Features from Training Neighbours
class TrainingNeighbourRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, n_neighbors=8):
        self.estimator = estimator
        self.n_neighbors = n_neighbors

    def _feature(self, query, exclude_self=False):
        distance = cdist(query, self.donor_coordinates_)
        if exclude_self:
            np.fill_diagonal(distance, np.inf)
        k = min(self.n_neighbors, len(self.donor_y_)-(1 if exclude_self else 0))
        if k < 1:
            raise ValueError('Too few donor samples for neighbour pH.')
        indices = np.argsort(distance, axis=1, kind='stable')[:, :k]
        d = np.take_along_axis(distance, indices, axis=1)
        values = self.donor_y_[indices]
        zero = d <= 1e-8
        weights = 1 / np.maximum(d, 1e-8)
        has_zero = zero.any(axis=1)
        weights[has_zero] = zero[has_zero].astype(float)
        return (weights*values).sum(axis=1)/weights.sum(axis=1)

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        self.donor_coordinates_ = x[:, -2:].copy()
        self.donor_y_ = np.asarray(y, dtype=float).copy()
        self.n_features_in_ = x.shape[1]
        lag = self._feature(self.donor_coordinates_, exclude_self=True)
        self.estimator_ = clone(self.estimator).fit(np.column_stack([x[:, :-2], lag]), y)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        lag = self._feature(x[:, -2:])
        return self.estimator_.predict(np.column_stack([x[:, :-2], lag]))


def fit_neighbour_stacking(x_train, y_train, coords_train, x_test, coords_test):
    train = np.column_stack([x_train, coords_train])
    test = np.column_stack([x_test, coords_test])
    model, _, _, audit = fit_stacking_process(train, y_train, with_coordinates='spatial_lag')
    predictions = {f'{name}_spatial_lag': model.base_models[name].predict(test)
                   for name in ['RF', 'TabPFN']}
    predictions['Stacking_spatial_lag'] = model.predict(test)
    audit['neighbour_feature'] = {'neighbours': 8, 'weights': 'inverse distance; exact-location donors averaged',
        'training_rows': 'Leave self out within each fit',
        'validation_rows': 'Donors exclusively from the current training subset',
        'coordinate_CRS': 'EPSG:32649', 'features': 'Environmental predictors plus neighbour pH; coordinates only locate donors'}
    return predictions, audit


# 3.3 Regression Kriging of Nested OOF Stacking Residuals
def predict_rk_correction(coords_train, residuals_train, coords_test):
    model = OrdinaryKriging(coords_train[:, 0], coords_train[:, 1], residuals_train,
                            variogram_model='exponential', exact_values=True, verbose=False)
    mean, variance = model.execute('points', coords_test[:, 0], coords_test[:, 1])
    correction = np.ma.asarray(mean).filled(np.nan).ravel()
    variance = np.ma.asarray(variance).filled(np.nan).ravel()
    if not np.isfinite(correction).all() or not np.isfinite(variance).all() or (variance < -1e-6).any():
        raise ValueError('Invalid residual kriging prediction or variance.')
    audit = {'variogram_model': 'exponential',
             'variogram_parameters_partial_sill_range_nugget': model.variogram_model_parameters.tolist(),
             'SD_scope': 'Conditional residual-kriging SD, not total predictive uncertainty'}
    return correction, np.sqrt(np.maximum(variance, 0)), audit


# 3.4 GWR of Local Residual Trends
def fit_gwr_residual_model(coords_train, residuals_train):
    if len(residuals_train) <= GWR_MIN_NEIGHBOURS + 2:
        raise ValueError('Too few training samples for the prespecified GWR bandwidth range.')
    scaler_coords = StandardScaler().fit(coords_train)
    gwr_X_train = scaler_coords.transform(coords_train)
    bw = Sel_BW(coords_train, residuals_train[:, None], gwr_X_train,
                fixed=False, kernel='bisquare', n_jobs=1).search(
                    criterion='AICc', bw_min=GWR_MIN_NEIGHBOURS,
                    bw_max=len(residuals_train)-1)
    model = GWR(coords_train, residuals_train[:, None], gwr_X_train,
                bw=bw, fixed=False, kernel='bisquare', n_jobs=1)
    results = model.fit()
    audit = {'bandwidth_neighbours': float(bw), 'bandwidth_selection': 'Training AICc',
             'minimum_neighbours': GWR_MIN_NEIGHBOURS, 'kernel': 'adaptive bisquare',
             'regressors': ['intercept', 'scaled Easting', 'scaled Northing'],
             'interpretation': 'Local residual trend; no environmental-driver interpretation'}
    return model, results, scaler_coords, audit


def predict_fitted_gwr(bundle, coords_test):
    model, results, scaler_coords, _ = bundle
    count = len(coords_test)
    correction = np.empty(count, dtype=float)
    batch_n = len(model.y)
    scale, training_residual = results.scale, results.resid_response
    for start in range(0, count, batch_n):
        query = coords_test[start:start+batch_n]
        correction[start:start+len(query)] = model.predict(
            query, scaler_coords.transform(query), scale, training_residual).predictions.ravel()
    model.points = None
    model.P = None
    if not np.isfinite(correction).all():
        raise ValueError('Nonfinite GWR correction.')
    return correction


def predict_gwr_correction(coords_train, residuals_train, coords_test):
    bundle = fit_gwr_residual_model(coords_train, residuals_train)
    correction = predict_fitted_gwr(bundle, coords_test)
    return correction, {**bundle[3], 'correction_min': float(correction.min()),
                        'correction_max': float(correction.max())}


def fit_spatial_comparisons(x_train, y_train, coords_train, x_test, coords_test,
                            base_prediction, residuals_train):
    n = len(x_test)
    predictions = {name: np.full(n, np.nan) for name in SPATIAL_NAMES}
    rk_sd = np.full(n, np.nan)
    audit = {}
    try:
        predictions['Stacking_coordinates'], record = fit_coordinate_stacking(
            x_train, y_train, coords_train, x_test, coords_test)
        for name, values in record['base_validation_predictions'].items():
            predictions[f'{name}_coordinates'] = np.asarray(values)
            audit[f'{name}_coordinates'] = {'status': 'ok', 'source': 'Coordinate-stack final base fit'}
        audit['Stacking_coordinates'] = {'status': 'ok', **record}
    except Exception as exc:
        audit['Stacking_coordinates'] = {'status': 'failed', 'error': repr(exc)}
    try:
        lag_predictions, lag_record = fit_neighbour_stacking(
            x_train, y_train, coords_train, x_test, coords_test)
        predictions.update(lag_predictions)
        for name in lag_predictions:
            audit[name] = {'status': 'ok', **lag_record}
    except Exception as exc:
        for name in ['RF_spatial_lag', 'TabPFN_spatial_lag', 'Stacking_spatial_lag']:
            audit[name] = {'status': 'failed', 'error': repr(exc)}
    try:
        correction, rk_sd, record = predict_rk_correction(coords_train, residuals_train, coords_test)
        predictions['Stacking_RK'] = base_prediction + correction
        audit['Stacking_RK'] = {'status': 'ok', **record}
    except Exception as exc:
        audit['Stacking_RK'] = {'status': 'failed', 'error': repr(exc)}
    try:
        correction, record = predict_gwr_correction(coords_train, residuals_train, coords_test)
        predictions['Stacking_GWR'] = base_prediction + correction
        audit['Stacking_GWR'] = {'status': 'ok', **record}
    except Exception as exc:
        audit['Stacking_GWR'] = {'status': 'failed', 'error': repr(exc)}
    return predictions, rk_sd, audit


# 3.5 Evaluate All Models Using the Shared Outer kNNDM Folds
if not Path(TABPFN_MODEL_PATH).is_file():
    raise FileNotFoundError(f'Local TabPFN checkpoint not found: {TABPFN_MODEL_PATH}')
EVALUATED_NAMES = BASE_NAMES + ['Stacking'] + SPATIAL_NAMES
SKIP_OUTER_CV = os.environ.get('SKIP_OUTER_CV', '0') == '1'
if SKIP_OUTER_CV:
    saved_config = json.loads((MODEL_OUTPUT_DIR / 'model_configuration.json').read_text())
    if (saved_config['source_data_sha256'] != hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest()
            or saved_config['base_learners'] != BASE_NAMES):
        raise ValueError('Completed CV inputs/models do not match current mapping inputs.')
    outer_predictions = pd.read_csv(MODEL_OUTPUT_DIR / 'outer_OOF_predictions.csv')
    if (not np.array_equal(outer_predictions.FID, full_data.FID)
            or not np.array_equal(outer_predictions.CV_Fold, fold_ids)
            or not np.allclose(outer_predictions.pH, y_all)
            or not np.isfinite(outer_predictions[EVALUATED_NAMES].to_numpy(float)).all()):
        raise ValueError('Incomplete or mismatched saved outer predictions.')
    print('Reusing completed outer CV predictions; final models will still be refitted on all samples.', flush=True)
else:
    EVALUATED_NAMES = BASE_NAMES + ['Stacking'] + SPATIAL_NAMES
    outer_predictions = full_data[['FID', 'Lon', 'Lat', 'pH']].copy()
    outer_predictions['CV_Fold'] = fold_ids
    for model_name in EVALUATED_NAMES:
        outer_predictions[model_name] = np.nan
    outer_predictions['RK_conditional_residual_SD'] = np.nan
    fold_model_audit = []
    fold_metric_rows = []
    stacking_coefficient_rows = []
    meta_candidate_rows = []
    for outer_fold, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        print(f'Outer kNNDM fold {outer_fold}/{N_FOLDS}: train={len(train_idx)}, test={len(test_idx)}', flush=True)
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        coords_train_proj, coords_test_proj = coords_all_proj[train_idx], coords_all_proj[test_idx]
        trained = fit_outer_base_fold(X_train, y_train, X_test)
        predictions = trained['predictions']
        residuals_train = trained['residuals_train']
        stacking_coefficient_rows.append(show_stacking_coefficients(trained['stacking_audit'],outer_fold,'Stacking'))
        meta_candidate_rows.extend(dict(row, outer_fold=outer_fold, variant='Stacking',
            selected=row['family']==trained['stacking_audit']['meta_family'] and
                     all(row.get(key)==value for key,value in trained['stacking_audit']['meta_parameters'].items()))
            for row in trained['stacking_audit']['meta_candidate_results'])
        base_comparison = []
        for name in BASE_NAMES+['Stacking']:
            prediction = predictions[name]
            base_comparison.append({'model': name, 'R2': float(r2_score(y_test,prediction)),
                                    'RMSE': float(np.sqrt(mean_squared_error(y_test,prediction))),
                                    'MAE': float(mean_absolute_error(y_test,prediction))})
        print('  Outer held-out accuracy (not tuning scores):\n'+pd.DataFrame(base_comparison).to_string(index=False),flush=True)
        residual_table = pd.DataFrame({'row_index': train_idx, 'FID': full_data.FID.to_numpy()[train_idx],
                                       'outer_fold': outer_fold, 'residual_inner_fold': trained['residual_ids'],
                                       'observed_pH': y_train, 'OOF_stacking_prediction': trained['training_stack_oof'],
                                       'residual': residuals_train,
                                       'x': coords_train_proj[:,0], 'y': coords_train_proj[:,1]})
        residual_table.to_csv(MODEL_OUTPUT_DIR / f'fold_{outer_fold}_training_residuals.csv', index=False)
        pd.DataFrame(trained['base_oof'], columns=BASE_NAMES).assign(row_index=train_idx,
                   stacking_inner_fold=trained['base_oof_ids']).to_csv(
                       MODEL_OUTPUT_DIR / f'fold_{outer_fold}_meta_training_OOF.csv', index=False)
        spatial_audit = {}
        print('  Comparing coordinate stacking, residual kriging and GWR...', flush=True)
        spatial_predictions, rk_sd, spatial_audit = fit_spatial_comparisons(
            X_train, y_train, coords_train_proj, X_test, coords_test_proj,
            predictions['Stacking'], residuals_train)
        predictions.update(spatial_predictions)
        outer_predictions.loc[test_idx, 'RK_conditional_residual_SD'] = rk_sd
        lag_audit = spatial_audit['Stacking_spatial_lag']
        if lag_audit['status'] == 'ok':
            stacking_coefficient_rows.append(show_stacking_coefficients(lag_audit, outer_fold, 'Stacking_spatial_lag'))
            meta_candidate_rows.extend(dict(row, outer_fold=outer_fold, variant='Stacking_spatial_lag',
                selected=all(row.get(key)==value for key,value in lag_audit['meta_parameters'].items()))
                for row in lag_audit['meta_candidate_results'])
        coordinate_audit = spatial_audit['Stacking_coordinates']
        if coordinate_audit['status']=='ok':
            stacking_coefficient_rows.append(show_stacking_coefficients(coordinate_audit,outer_fold,'Stacking_coordinates'))
            meta_candidate_rows.extend(dict(row, outer_fold=outer_fold, variant='Stacking_coordinates',
                selected=all(row.get(key)==value for key,value in coordinate_audit['meta_parameters'].items()))
                for row in coordinate_audit['meta_candidate_results'])
        fold_model_audit.append({'outer_fold': outer_fold, 'train_rows': train_idx.tolist(),
                                 'test_rows': test_idx.tolist(), 'stacking': trained['stacking_audit'],
                                 'residual_process': trained['residual_audit'], 'spatial_methods': spatial_audit,
                                 'Moran_diagnostics': diagnose_training_residuals(coords_train_proj, residuals_train)})
        for name in EVALUATED_NAMES:
            prediction = predictions[name]
            outer_predictions.loc[test_idx, name] = prediction
            complete = bool(np.isfinite(prediction).all())
            fold_metric_rows.append({'outer_fold': outer_fold, 'model': name, 'complete': complete,
                                    'n_predictions': int(np.isfinite(prediction).sum()),
                                    **({'R2': float(r2_score(y_test, prediction)),
                                        'RMSE': float(np.sqrt(mean_squared_error(y_test, prediction))),
                                        'MAE': float(mean_absolute_error(y_test, prediction))} if complete else {})})
        outer_predictions.to_csv(MODEL_OUTPUT_DIR / 'outer_OOF_predictions.csv', index=False)
        stacking_display = outer_predictions[['FID','Lon','Lat','pH','CV_Fold']+BASE_NAMES+['Stacking']].copy()
        stacking_display['Stacking_error_pred_minus_observed'] = stacking_display.Stacking-stacking_display.pH
        if 'Stacking_coordinates' in outer_predictions:
            stacking_display['Stacking_coordinates'] = outer_predictions.Stacking_coordinates
        stacking_display.to_csv(MODEL_OUTPUT_DIR / 'stacking_predictions.csv',index=False)
        pd.DataFrame(stacking_coefficient_rows).to_csv(MODEL_OUTPUT_DIR / 'stacking_coefficients.csv',index=False)
        pd.DataFrame(meta_candidate_rows).to_csv(MODEL_OUTPUT_DIR / 'meta_candidate_scores.csv',index=False)
        pd.DataFrame(fold_metric_rows).to_csv(MODEL_OUTPUT_DIR / 'per_fold_metrics.csv', index=False)
        (MODEL_OUTPUT_DIR / 'nested_training_audit.json').write_text(json.dumps(fold_model_audit, indent=2))


    (MODEL_OUTPUT_DIR / 'model_configuration.json').write_text(json.dumps({
        'outer_validation': 'farmland CAST kNNDM five-fold', 'training_internal_folds': INNER_FOLDS,
        'internal_design': 'random KFold', 'base_learners': BASE_NAMES,
        'meta_learner': 'Scaled Ridge or ElasticNet, whole-stack training-only selection',
        'meta_candidates': meta_candidates(), 'stacking_OOF_folds': STACKING_OOF_FOLDS,
        'meta_selection_folds': META_SELECTION_FOLDS, 'residual_OOF_folds': RESIDUAL_OOF_FOLDS,
        'exploratory_revision_after_inspecting_previous_outer_results': True,
        'outer_folds_frozen': True,
        'RF_n_estimators': RF_ESTIMATORS, 'TabPFN_n_estimators': TABPFN_ESTIMATORS,
        'TabPFN_checkpoint': Path(TABPFN_MODEL_PATH).name,
        'checkpoint_sha256': hashlib.sha256(Path(TABPFN_MODEL_PATH).read_bytes()).hexdigest(),
        'TabPFN_fine_tuned': False, 'seed': RANDOM_SEED, 'evaluated_methods': EVALUATED_NAMES,
        'coordinate_model': 'All bases receive environment + EPSG:32649 Easting/Northing; train-fit coordinate scaling',
        'RK': 'Exponential variogram fitted on training entire-stack OOF residuals',
        'GWR': 'Adaptive bisquare local residual regression on scaled coordinates; training AICc bandwidth',
        'no_test_percentile_clipping': True, 'no_silent_base_fallback': True,
        'spatial_lag_method': 'Eight inverse-distance training pH neighbours; fit-local donor sets, training leave-self-out',
        'scope': 'Cross-validation, spatial comparisons and optional farmland mapping',
        'source_data_sha256': hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest()
    }, indent=2))

# 4. Performance Evaluation Using Pooled Outer-Fold Predictions
def accuracy_metrics(observed, predicted):
    complete = bool(np.isfinite(predicted).all())
    result = {'complete': complete, 'n_predictions': int(np.isfinite(predicted).sum())}
    if not complete:
        return {**result, 'R2': np.nan, 'RMSE': np.nan, 'MAE': np.nan,
                'Bias_pred_minus_observed': np.nan, 'Pearson_r': np.nan}
    correlation = float(pearsonr(observed, predicted)[0]) if np.std(observed)>0 and np.std(predicted)>0 else np.nan
    return {**result, 'R2': float(r2_score(observed, predicted)),
            'RMSE': float(np.sqrt(mean_squared_error(observed, predicted))),
            'MAE': float(mean_absolute_error(observed, predicted)),
            'Bias_pred_minus_observed': float(np.mean(predicted-observed)), 'Pearson_r': correlation}

def summarize_outer_accuracy(oof_table, names, output_dir):
    output_dir = Path(output_dir)
    if not oof_table.FID.is_unique or not np.isfinite(oof_table.pH).all():
        raise ValueError('Invalid OOF sample identities or responses.')
    observed = oof_table.pH.to_numpy(float)
    pooled = []; per_fold = []
    for name in names:
        pooled.append({'model': name, 'n_samples': len(observed),
                       **accuracy_metrics(observed, oof_table[name].to_numpy(float))})
        for fold in sorted(oof_table.CV_Fold.unique()):
            subset = oof_table[oof_table.CV_Fold==fold]
            per_fold.append({'outer_fold': int(fold), 'model': name, 'n_samples': len(subset),
                             **accuracy_metrics(subset.pH.to_numpy(float), subset[name].to_numpy(float))})
    pooled = pd.DataFrame(pooled); per_fold = pd.DataFrame(per_fold)
    pooled.to_csv(output_dir / 'pooled_metrics.csv', index=False)
    per_fold.to_csv(output_dir / 'per_fold_metrics.csv', index=False)
    paired = []; gains = []
    for name in [m for m in names if m.startswith('Stacking_')]:
        method = oof_table[name].to_numpy(float)
        baseline = oof_table.Stacking.to_numpy(float)
        for i in range(len(observed)):
            paired.append({'FID': oof_table.FID.iloc[i], 'CV_Fold': int(oof_table.CV_Fold.iloc[i]),
                           'method': name, 'observed_pH': observed[i],
                           'baseline_prediction': baseline[i], 'method_prediction': method[i],
                           'delta_absolute_error': abs(method[i]-observed[i])-abs(baseline[i]-observed[i]),
                           'delta_squared_error': (method[i]-observed[i])**2-(baseline[i]-observed[i])**2})
        for fold in sorted(oof_table.CV_Fold.unique()):
            base = per_fold[(per_fold.outer_fold==fold)&(per_fold.model=='Stacking')].iloc[0]
            row = per_fold[(per_fold.outer_fold==fold)&(per_fold.model==name)].iloc[0]
            gains.append({'outer_fold': int(fold), 'method': name, 'complete': bool(row.complete),
                          'delta_RMSE_vs_stacking': row.RMSE-base.RMSE,
                          'delta_MAE_vs_stacking': row.MAE-base.MAE})
    paired = pd.DataFrame(paired); gains = pd.DataFrame(gains)
    if not paired.empty:
        paired.to_csv(output_dir / 'paired_errors_vs_stacking.csv', index=False)
        gains.to_csv(output_dir / 'spatial_gain_by_fold.csv', index=False)
    return pooled, per_fold, gains

accuracy_summary, fold_accuracy, spatial_gains = summarize_outer_accuracy(
    outer_predictions, EVALUATED_NAMES, MODEL_OUTPUT_DIR)
print('\nPOOLED OUTER-FOLD ACCURACY (all samples)\n'+accuracy_summary.to_string(index=False), flush=True)

# 5. Observed versus Outer-Fold Predicted Values
def plot_outer_evaluation(oof_table, summary, names, output_dir):
    mpl.rcParams.update({'font.family':'serif',
                         'font.serif':['Times New Roman', 'DejaVu Serif'],
                         'axes.unicode_minus':False})
    output_dir = Path(output_dir)
    observed = oof_table.pH.to_numpy(float)
    preferred = ['RF', 'TabPFN', 'Stacking',
                 'Stacking_coordinates',
                 'Stacking_spatial_lag', 'Stacking_RK', 'Stacking_GWR']
    scatter_names = [name for name in preferred if name in names]
    if not scatter_names:
        scatter_names = list(names)
    labels = {'RF': 'RF', 'TabPFN': 'TabPFN', 'Stacking': 'Stacking',
              'Stacking_coordinates': 'Stacking + coordinates',
              'Stacking_spatial_lag': 'Stacking + spatial lag',
              'Stacking_RK': 'Stacking + RK', 'Stacking_GWR': 'Stacking + GWR'}
    finite = oof_table[scatter_names].to_numpy(float)
    finite = finite[np.isfinite(finite)]
    lo = min(observed.min(),finite.min()) if len(finite) else observed.min()
    hi = max(observed.max(),finite.max()) if len(finite) else observed.max()
    lo = np.floor((lo-.05)*2)/2
    hi = np.ceil((hi+.05)*2)/2
    tick_step = .5 if hi-lo <= 4 else 1.
    ticks = np.arange(lo, hi+tick_step/2, tick_step)
    base_row = [name for name in scatter_names if name in ['RF', 'TabPFN', 'Stacking']]
    spatial_row = [name for name in scatter_names if name not in base_row]
    panel_rows = [row for row in [base_row, spatial_row] if row]
    panel_mm = 65.
    gap_mm = 18.
    max_columns = max(map(len, panel_rows))
    width_mm = 36.+max_columns*panel_mm+(max_columns-1)*gap_mm
    height_mm = 190. if len(panel_rows)==2 else 97.
    row_bottoms_mm = [112.,19.] if len(panel_rows)==2 else [19.]
    with mpl.rc_context({'font.family':'serif',
                         'font.serif':['Times New Roman', 'DejaVu Serif'],
                         'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'}):
        fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4), facecolor='white')
        panel_index = 0
        for row_names, bottom_mm in zip(panel_rows, row_bottoms_mm):
            row_width_mm = len(row_names)*panel_mm+(len(row_names)-1)*gap_mm
            row_left_mm = (width_mm-row_width_mm)/2
            left_positions = row_left_mm+np.arange(len(row_names))*(panel_mm+gap_mm)
            for name, left_mm in zip(row_names, left_positions):
                ax = fig.add_axes([left_mm/width_mm, bottom_mm/height_mm,
                                   panel_mm/width_mm, panel_mm/height_mm])
                prediction = oof_table[name].to_numpy(float)
                valid = np.isfinite(prediction)
                ax.scatter(observed[valid], prediction[valid], color='steelblue',
                           s=24, alpha=.7, edgecolor='white', linewidths=.4, zorder=2)
                ax.plot([lo,hi], [lo,hi], 'k--', lw=1.2, zorder=1)
                ax.set(xlim=(lo,hi), ylim=(lo,hi), xticks=ticks, yticks=ticks)
                ax.set_aspect('equal')
                ax.grid(alpha=.3)
                ax.tick_params(direction='out', length=3., width=.8,
                               labelsize=10., pad=3.)
                for spine in ax.spines.values():
                    spine.set_linewidth(.8)
                    spine.set_color('black')
                title = f'({chr(ord("a")+panel_index)}) '+labels.get(name, name.replace('_',' + '))
                ax.set_title(title, fontsize=14., pad=10.)
                ax.set_xlabel('OBSERVED SOIL pH', fontsize=12.)
                ax.set_ylabel('PREDICTED SOIL pH', fontsize=12.)
                row = summary[summary.model==name].iloc[0]
                text = (f'R² = {row.R2:.3f}\nRMSE = {row.RMSE:.3f}' if row.complete
                        else 'Incomplete method\nNo pooled accuracy')
                ax.text(.05, .95, text, transform=ax.transAxes, va='top',
                        fontsize=11., linespacing=1.25,
                        bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.8, 'pad':2.})
                panel_index += 1
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(output_dir / f'observed_vs_predicted.{extension}',
                        dpi=600, facecolor='white')
        plt.close(fig)

plot_outer_evaluation(outer_predictions, accuracy_summary, EVALUATED_NAMES, MODEL_OUTPUT_DIR)

# 6. Farmland pH Mapping, Empirical Error Surfaces and GeoTIFF Export
# 6.1 Validate Band Order and Refit Final Models Using All Samples
RUN_MAPPING = os.environ.get('RUN_MAPPING', '1') == '1'
RASTER_BAND_ORDER_CONFIRMED = True
MAP_BATCH_SIZE = 512
MAP_OUTPUT_DIR = MODEL_OUTPUT_DIR / 'farmland_maps'
MAP_NAMES = ['pH_Stacking_coordinates','pH_Stacking_spatial_lag',
             'pH_Stacking_RK','pH_Stacking_GWR']
MAP_FIGURE_DIR = MODEL_OUTPUT_DIR / 'gis_spatial_maps'
ERROR_IDW_NEIGHBOURS = 8
ERROR_IDW_POWER = 2.0


def validate_mapping_grid(src, mask, features, band_order_confirmed):
    if src.crs is None or src.crs.to_epsg()!=32649 or src.count!=len(features) or src.shape!=mask.shape:
        raise ValueError('Raster CRS/band count/grid does not match the farmland domain.')
    try:
        descriptions = list(src.descriptions)
    except UnicodeDecodeError:
        if not band_order_confirmed:
            raise ValueError('Invalid raster description encoding; verify band order first.')
        descriptions = [None] * src.count
        warnings.warn('Unreadable GIS band labels; using verified feature order.')
    if not band_order_confirmed and descriptions!=features:
        raise ValueError('Raster band order must be verified before mapping.')
    if band_order_confirmed:
        for i, label in enumerate(descriptions):
            if label and label != f'Band_{i+1}' and label != features[i]:
                raise ValueError('Semantic raster descriptions conflict with the confirmed predictor order.')


def fit_final_mapping_models(x_all, y_all, coords_all):
    base,_,_,base_audit = fit_stacking_process(x_all,y_all)
    _,residual,residual_ids,residual_audit = stacking_oof_residuals(x_all,y_all)
    coordinate_model,_,_,coordinate_audit = fit_stacking_process(
        np.column_stack([x_all,coords_all]),y_all,with_coordinates=True)
    print('Fitting all-sample stacking with training-only neighbour pH...', flush=True)
    lag_model,_,_,lag_audit = fit_stacking_process(
        np.column_stack([x_all,coords_all]),y_all,with_coordinates='spatial_lag')
    kriging = OrdinaryKriging(coords_all[:,0],coords_all[:,1],residual,
                             variogram_model='exponential',exact_values=True,verbose=False)
    gwr = fit_gwr_residual_model(coords_all,residual)
    return {'stacking':base,'coordinates':coordinate_model,'spatial_lag':lag_model,
            'kriging':kriging,'gwr':gwr}, {
        'base':base_audit,'coordinates':coordinate_audit,'residual_process':residual_audit,
        'spatial_lag':lag_audit,
        'GWR':gwr[3],'kriging_variogram':kriging.variogram_model_parameters.tolist()}, residual,residual_ids


# 6.2 Batch Prediction of the Four Spatial Configurations
def predict_mapping_batch(models, predictors, coords):
    base_single = [models['stacking'].base_models[name].predict(predictors) for name in ['RF','TabPFN']]
    coordinate_input = np.column_stack([predictors,coords])
    coordinate_single = [models['coordinates'].base_models[name].predict(coordinate_input) for name in ['RF','TabPFN']]
    base = (models['stacking'].meta_model.predict(np.column_stack(base_single))
            if hasattr(models['stacking'], 'meta_model') else models['stacking'].predict(predictors))
    coordinate = (models['coordinates'].meta_model.predict(np.column_stack(coordinate_single))
                  if hasattr(models['coordinates'], 'meta_model') else models['coordinates'].predict(coordinate_input))
    lag_prediction = models['spatial_lag'].predict(coordinate_input)
    correction,var = models['kriging'].execute('points',coords[:,0],coords[:,1])
    correction = np.ma.asarray(correction).filled(np.nan).ravel()
    var = np.ma.asarray(var).filled(np.nan).ravel()
    gwr_correction = predict_fitted_gwr(models['gwr'],coords)
    if not np.isfinite(var).all() or (var < -1e-6).any():
        raise ValueError('Invalid residual-kriging mapping variance.')
    predictions = np.column_stack([coordinate,lag_prediction,
                                   base+correction,base+gwr_correction])
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite mapping prediction; no silent fallback or clipping.')
    return predictions


# 6.3 Export Farmland pH GeoTIFFs and Record Coverage
def export_farmland_maps(models, raster_path, mask, features, output_dir,
                         batch_size=512, band_order_confirmed=False):
    if batch_size<1:
        raise ValueError('Mapping batch size must be positive.')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    status_file = output_dir / 'map_status.json'
    status_file.write_text(json.dumps({'status':'running'},indent=2))
    checkpoint_dir = output_dir / 'prediction_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    model_cache = output_dir / 'final_models.pkl'
    checkpoint_enabled = model_cache.exists()
    model_fingerprint = hashlib.sha256(model_cache.read_bytes()).hexdigest() if checkpoint_enabled else ''
    import ast
    prediction_nodes = [node for node in ast.parse(Path(__file__).read_text(encoding='utf-8')).body
                        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in
                        ['predict_mapping_batch','predict_fitted_gwr','TrainingNeighbourRegressor']]
    prediction_code = ast.dump(ast.Module(body=prediction_nodes,type_ignores=[]),include_attributes=False)
    raster_stat = Path(raster_path).stat()
    run_fingerprint = hashlib.sha256((model_fingerprint + str(raster_stat.st_size)
        + str(raster_stat.st_mtime_ns) + str(features) + str(MAP_NAMES) + prediction_code).encode()).hexdigest()
    resumed_batches = 0
    completed_batches = 0
    predicted_n = 0
    imputed_n = 0; missing_per_band = np.zeros(len(features),dtype=int)
    grid_min = np.full(len(features),np.inf); grid_max = np.full(len(features),-np.inf)
    try:
        with rasterio.open(raster_path) as src, contextlib.ExitStack() as stack:
            validate_mapping_grid(src,mask,features,band_order_confirmed)
            profile = src.profile.copy()
            profile.update(driver='GTiff',count=1,dtype='float32',nodata=-9999.,compress='lzw')
            destinations = [stack.enter_context(rasterio.open(output_dir / f'{name}.partial.tif','w',**profile))
                            for name in MAP_NAMES]
            for row in range(0,src.height,256):
                for col in range(0,src.width,256):
                    win = Window(col,row,min(256,src.width-col),min(256,src.height-row))
                    rs,cs = win.toslices(); farm = mask[rs,cs].astype(bool)
                    arrays = [np.full(farm.shape,-9999.,dtype='float32') for _ in MAP_NAMES]
                    if farm.any():
                        values = src.read(window=win,masked=True).astype('float64').filled(np.nan)
                        values[~np.isfinite(values)|(np.abs(values)>1e15)|(values == -9999)] = np.nan
                        values = np.moveaxis(values,0,-1)
                        farmland_values = values[farm]
                        missing_per_band += np.isnan(farmland_values).sum(axis=0)
                        for j in range(len(features)):
                            finite = farmland_values[:,j][np.isfinite(farmland_values[:,j])]
                            if len(finite):
                                grid_min[j] = min(grid_min[j],finite.min())
                                grid_max[j] = max(grid_max[j],finite.max())
                        complete_covariates = farm & np.isfinite(values).all(axis=2)
                        imputed_n += int(np.count_nonzero(farm & ~complete_covariates))
                        valid = farm
                        rr,cc = np.where(valid)
                        gx,gy = rasterio.transform.xy(src.transform,rr+row,cc+col,offset='center')
                        coords = np.column_stack([gx,gy]); predictors = values[valid]
                        predicted = np.empty((len(coords),len(MAP_NAMES)))
                        for start in range(0,len(coords),batch_size):
                            stop = min(start+batch_size,len(coords))
                            input_fingerprint = hashlib.sha256(
                                run_fingerprint.encode() + predictors[start:stop].tobytes()
                                + coords[start:stop].tobytes()).hexdigest()
                            checkpoint = checkpoint_dir / f'{input_fingerprint}.npz'
                            if checkpoint_enabled and checkpoint.exists():
                                with np.load(checkpoint) as saved:
                                    batch_prediction = saved['prediction']
                                resumed_batches += 1
                            else:
                                batch_prediction = predict_mapping_batch(models,predictors[start:stop],coords[start:stop])
                                if checkpoint_enabled:
                                    temporary = checkpoint.with_suffix('.partial.npz')
                                    np.savez_compressed(temporary, prediction=batch_prediction)
                                    temporary.replace(checkpoint)
                            if batch_prediction.shape != (stop-start,len(MAP_NAMES)) or not np.isfinite(batch_prediction).all():
                                raise ValueError('Invalid prediction/checkpoint batch.')
                            predicted[start:stop] = batch_prediction
                            completed_batches += 1
                            if completed_batches % 10 == 0:
                                progress = {'status':'running','predicted_pixels':predicted_n+stop,
                                    'total_farmland_pixels':int(np.count_nonzero(mask)),
                                    'completed_batches':completed_batches,'resumed_batches':resumed_batches,
                                    'checkpointed':checkpoint_enabled}
                                status_file.write_text(json.dumps(progress,indent=2))
                                print(f'  Mapping progress: {predicted_n+stop:,}/{int(np.count_nonzero(mask)):,} pixels; cached batches={resumed_batches}',flush=True)
                        for j in range(len(MAP_NAMES)):
                            arrays[j][valid] = predicted[:,j]
                        predicted_n += len(coords)
                    for dst,array in zip(destinations,arrays):
                        dst.write(array,1,window=win)
                print(f'  Mapping rows {row+1}–{min(row+256,src.height)}/{src.height}; predicted farmland pixels={predicted_n:,}',flush=True)
        if not predicted_n:
            raise ValueError('No farmland pixels predicted; no completed map.')
        for name in MAP_NAMES:
            (output_dir / f'{name}.partial.tif').replace(output_dir / f'{name}.tif')
        pd.DataFrame({'band':np.arange(1,len(features)+1),'feature':features,
                      'farmland_missing_pixels':missing_per_band,
                      'farmland_min':np.where(np.isfinite(grid_min),grid_min,np.nan),
                      'farmland_max':np.where(np.isfinite(grid_max),grid_max,np.nan)}).to_csv(
                          output_dir / 'mapping_covariate_audit.csv',index=False)
        coverage = {'status':'complete','farmland_pixels':int(np.count_nonzero(mask)),
                    'predicted_farmland_pixels':predicted_n,
                    'unpredicted_missing_covariate_pixels':int(np.count_nonzero(mask))-predicted_n,
                    'covers_all_farmland':predicted_n==int(np.count_nonzero(mask)),
                    'predicted_pixels_with_imputed_inputs':imputed_n,
                    'predicted_pixels_with_complete_inputs':predicted_n-imputed_n,
                    'completed_batches':completed_batches,'resumed_batches':resumed_batches,
                    'mask':'configured farmland domain, pixel-centre inclusion',
                    'missing_policy':'All farmland predicted; missing inputs handled by training-fitted model pipeline imputation, no grid-mean fitting',
                    'prediction_maps':[f'{name}.tif' for name in MAP_NAMES],
                    'covariate_coverage_records':['map_status.json','mapping_covariate_audit.csv'],
                    'band_order':'user-confirmed CSV feature order' if band_order_confirmed else 'raster band descriptions',
                    'no_clipping':True,'GWR_prediction':'actual fitted GWR, same as validation'}
        status_file.write_text(json.dumps(coverage,indent=2))
        return coverage
    except Exception as exc:
        status_file.write_text(json.dumps({'status':'failed','error':repr(exc),
                                         'partial_outputs_are_not_completed_maps':True},indent=2))
        raise


# 6.4 Interpolate Absolute Outer-Fold Errors Using IDW
def idw_error_estimates(source_xy, source_errors, query_xy, neighbours=8, power=2.):
    source_xy = np.asarray(source_xy, dtype=float)
    source_errors = np.asarray(source_errors, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)
    if (source_xy.ndim!=2 or source_xy.shape[1]!=2 or source_errors.ndim!=2
            or len(source_errors)!=len(source_xy) or not len(source_xy)
            or query_xy.ndim!=2 or query_xy.shape[1]!=2
            or not all(np.isfinite(a).all() for a in [source_xy,source_errors,query_xy])
            or (source_errors<0).any() or neighbours<1 or int(neighbours)!=neighbours
            or not np.isfinite(power) or power<=0):
        raise ValueError('Invalid absolute-error IDW inputs or settings.')
    unique_xy, inverse, counts = np.unique(source_xy, axis=0, return_inverse=True, return_counts=True)
    errors = np.zeros((len(unique_xy),source_errors.shape[1]))
    np.add.at(errors, inverse, source_errors)
    errors /= counts[:,None]
    k = min(int(neighbours),len(unique_xy))
    distance, indices = cKDTree(unique_xy).query(query_xy,k=k)
    distance = np.asarray(distance).reshape(len(query_xy),k)
    indices = np.asarray(indices).reshape(len(query_xy),k)
    zero = distance<=1e-8
    weights = np.maximum(distance,1e-8)**(-power)
    exact = zero.any(axis=1)
    weights[exact] = zero[exact].astype(float)
    weights /= weights.sum(axis=1,keepdims=True)
    return np.sum(weights[:,:,None]*errors[indices],axis=1)


def export_outer_error_surfaces(oof_table, raster_path, mask, output_dir,
                                neighbours=8, power=2.):
    models = ['Stacking_coordinates','Stacking_spatial_lag','Stacking_RK','Stacking_GWR']
    required = ['FID','Lon','Lat','pH','CV_Fold']+models
    if (not set(required).issubset(oof_table.columns) or not oof_table.FID.is_unique
            or oof_table.CV_Fold.isna().any() or oof_table.CV_Fold.nunique()<2
            or not np.isfinite(oof_table[['Lon','Lat','pH']+models].to_numpy(float)).all()):
        raise ValueError('Four complete, identified outer-fold predictions are required for error maps.')
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    points = oof_table[required].copy()
    xy = np.column_stack(Transformer.from_crs(4326,32649,always_xy=True).transform(
        points.Lon.to_numpy(float),points.Lat.to_numpy(float)))
    absolute = np.abs(points[models].to_numpy(float)-points.pH.to_numpy(float)[:,None])
    point_errors = points[['FID','Lon','Lat','pH','CV_Fold']].copy()
    for j,name in enumerate(models):
        point_errors[f'prediction_{name}'] = points[name]
        point_errors[f'absolute_outer_error_{name}'] = absolute[:,j]
    point_errors.to_csv(output_dir/'outer_absolute_errors_at_samples.csv',index=False)
    files = [output_dir/f'IDW_absolute_outer_error_{name}.tif' for name in models]
    with rasterio.open(raster_path) as src:
        if src.crs is None or src.crs.to_epsg()!=32649 or src.shape!=mask.shape:
            raise ValueError('Error maps require the same EPSG:32649 farmland grid.')
        profile = src.profile.copy()
        profile.update(driver='GTiff',count=1,dtype='float32',nodata=-9999.,compress='lzw')
        rr,cc = np.where(np.asarray(mask,dtype=bool))
        if not len(rr):
            raise ValueError('Empty farmland error-map domain.')
        arrays = [np.full(src.shape,-9999.,dtype='float32') for _ in models]
        for start in range(0,len(rr),20000):
            stop = min(start+20000,len(rr))
            x,y = rasterio.transform.xy(src.transform,rr[start:stop],cc[start:stop],offset='center')
            values = idw_error_estimates(xy,absolute,np.column_stack([x,y]),neighbours,power)
            for j in range(len(models)):
                arrays[j][rr[start:stop],cc[start:stop]] = values[:,j]
        for name,path,array in zip(models,files,arrays):
            temporary = path.with_suffix('.partial.tif')
            with rasterio.open(temporary,'w',**profile) as dst:
                dst.write(array,1)
                dst.set_band_description(1,f'IDW-interpolated absolute outer-CV error: {name}')
                dst.update_tags(model=name,source='Outer held-out predictions; observed minus predicted, absolute',
                    interpretation='Descriptive empirical error indicator; not calibrated prediction uncertainty',
                    idw_neighbours=str(neighbours),idw_power=str(power))
            temporary.replace(path)
    metadata = {'status':'complete','models':models,'sample_n':len(points),
        'farmland_pixels':len(rr),'crs':'EPSG:32649','idw_neighbours':int(neighbours),'idw_power':float(power),
        'source_table_sha256':hashlib.sha256(points.to_csv(index=False).encode()).hexdigest(),
        'duplicate_coordinates':'Mean absolute errors at coincident observations',
        'interpretation':'Spatial interpolation of observed outer cross-validation absolute errors',
        'limitations':['Not the unknown true error at each pixel','Not calibrated predictive uncertainty',
                       'Assumes nearby validation errors are informative about nearby locations',
                       'CV models use fewer training observations than the final all-sample models'],
        'rasters':[str(p) for p in files]}
    (output_dir/'empirical_error_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Four IDW outer-error surfaces exported: {len(rr):,} farmland pixels.',flush=True)
    return files


# 6.5 Collect Four pH Maps and Four Empirical Error Maps for GIS
def prepare_spatial_map_figures(oof_table, raster_path, mask, prediction_dir, output_dir,
                                neighbours=8, power=2.):
    import shutil
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    models = ['Stacking_coordinates','Stacking_spatial_lag','Stacking_RK','Stacking_GWR']
    error_paths = export_outer_error_surfaces(oof_table,raster_path,mask,output_dir,neighbours,power)
    prediction_paths = [Path(prediction_dir)/f'pH_{name}.tif' for name in models]
    missing = [str(path) for path in prediction_paths if not path.is_file()]
    gis_predictions = []
    for path in prediction_paths:
        if not path.is_file():
            continue
        with rasterio.open(path) as source, rasterio.open(raster_path) as reference:
            if (source.crs,source.transform,source.shape)!=(reference.crs,reference.transform,reference.shape):
                raise ValueError(f'Prediction raster does not match the GIS reference grid: {path}')
            if source.count!=1 or source.nodata!=-9999.:
                raise ValueError(f'Expected a single-band prediction raster with NoData=-9999: {path}')
            if not np.array_equal(source.read_masks(1)>0,np.asarray(mask,dtype=bool)):
                raise ValueError(f'Prediction raster farmland footprint does not match: {path}')
        destination = output_dir/path.name
        if path.resolve()!=destination.resolve():
            shutil.copy2(path,destination)
        gis_predictions.append(destination)
    status = {'status':'pending_prediction_maps' if missing else 'complete',
              'error_maps_complete':True,'prediction_maps_complete':not missing,
              'output_format':'Single-band numeric GeoTIFF','crs':'EPSG:32649',
              'nodata':-9999.,'gis_raster_count':len(error_paths)+len(gis_predictions),
              'gis_rasters':[str(p) for p in gis_predictions+error_paths],
              'missing_prediction_maps':missing}
    if missing:
        print('GIS export awaits prediction raster: '+', '.join(missing),flush=True)
    (output_dir/'spatial_figure_status.json').write_text(json.dumps(status,indent=2))
    return status


def final_mapping_signature(x_all,y_all,coords_all):
    import ast
    import importlib.metadata
    wanted = {'base_model_specs','tune_base_models','LearnedStackingModel','meta_candidates',
              'make_meta_estimator','build_base_oof','fit_stacking_process','stacking_oof_residuals',
              'TrainingNeighbourRegressor','fit_gwr_residual_model','fit_final_mapping_models'}
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
    if {n.name for n in nodes}!=wanted:
        raise ValueError('Incomplete final training-code fingerprint.')
    settings = {'schema':2,'base_names':BASE_NAMES,'feature_names':feature_cols,'seed':RANDOM_SEED,
                'inner_folds':INNER_FOLDS,'base_oof_folds':STACKING_OOF_FOLDS,
                'meta_folds':META_SELECTION_FOLDS,'residual_folds':RESIDUAL_OOF_FOLDS,
                'meta_alphas':META_ALPHAS,'rf_trees':RF_ESTIMATORS,'tab_ensemble':TABPFN_ESTIMATORS,
                'gwr_min':GWR_MIN_NEIGHBOURS,
                'checkpoint_sha256':hashlib.sha256(Path(TABPFN_MODEL_PATH).read_bytes()).hexdigest(),
                'versions':{name:importlib.metadata.version(name) for name in
                            ['numpy','scikit-learn','tabpfn','torch','PyKrige','mgwr']}}
    digest = hashlib.sha256(json.dumps(settings,sort_keys=True).encode())
    digest.update(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode())
    for values in [x_all,y_all,coords_all]:
        digest.update(np.ascontiguousarray(values,dtype='float64').tobytes())
    return digest.hexdigest()


# 6.6 Execute Final Mapping and Save Completion Status
if RUN_MAPPING:
    if not accuracy_summary.complete.all():
        raise ValueError('At least one outer method failed. Resolve it before final mapping.')
    with rasterio.open(RASTER_PATH) as mapping_grid:
        validate_mapping_grid(mapping_grid,farmland_mask,feature_cols,RASTER_BAND_ORDER_CONFIRMED)
    print('Refitting all samples and training-only OOF residuals for farmland mapping...',flush=True)
    MAP_OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    import cloudpickle
    cache_file = MAP_OUTPUT_DIR / 'final_models.pkl'
    cache_metadata = MAP_OUTPUT_DIR / 'final_models_signature.json'
    final_signature = final_mapping_signature(X_all,y_all,coords_all_proj)
    if (cache_file.exists() and cache_metadata.exists()
            and json.loads(cache_metadata.read_text()).get('schema') == 2
            and json.loads(cache_metadata.read_text())['signature'] == final_signature):
        final_models,final_audit,final_residuals,final_residual_folds = cloudpickle.loads(cache_file.read_bytes())
        print('Reusing matching all-sample final models.', flush=True)
    else:
        final_models,final_audit,final_residuals,final_residual_folds = fit_final_mapping_models(
            X_all,y_all,coords_all_proj)
        cache_file.write_bytes(cloudpickle.dumps((final_models,final_audit,final_residuals,final_residual_folds)))
        cache_metadata.write_text(json.dumps({'schema':2,'signature':final_signature},indent=2))
    pd.DataFrame({'FID':full_data.FID,'residual_inner_fold':final_residual_folds,
                  'observed_pH':y_all,'OOF_stacking_prediction':y_all-final_residuals,
                  'residual':final_residuals}).to_csv(MAP_OUTPUT_DIR / 'all_sample_training_residuals.csv',index=False)
    (MAP_OUTPUT_DIR / 'final_model_training_audit.json').write_text(json.dumps(final_audit,indent=2))
    coverage = export_farmland_maps(final_models,RASTER_PATH,farmland_mask,feature_cols,
                                    MAP_OUTPUT_DIR,MAP_BATCH_SIZE,RASTER_BAND_ORDER_CONFIRMED)
    print('Farmland map coverage:\n'+json.dumps(coverage,indent=2),flush=True)
else:
    print('Farmland mapping is configured but not executed. Set RUN_MAPPING=1 to refit and map farmland.',flush=True)

spatial_figure_status = None
spatial_figure_status = prepare_spatial_map_figures(
    outer_predictions,RASTER_PATH,farmland_mask,MAP_OUTPUT_DIR,MAP_FIGURE_DIR,
    ERROR_IDW_NEIGHBOURS,ERROR_IDW_POWER)
print('Spatial GeoTIFF outputs:\n'+json.dumps(spatial_figure_status,indent=2),flush=True)

(MODEL_OUTPUT_DIR / 'completion.json').write_text(json.dumps({
    'status':'complete','workflow':'full_spatial_evaluation',
    'mapping_executed':RUN_MAPPING,'outer_predictions_n':len(outer_predictions),
    'all_methods_complete':bool(accuracy_summary.complete.all()),
    'eight_spatial_panels_complete':bool(spatial_figure_status and spatial_figure_status['status']=='complete')},indent=2))
