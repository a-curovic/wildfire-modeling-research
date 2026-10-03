# California wildfire occurrence modelling

This is the updated version of my California wildfire research project. The goal is to estimate the probability that at least one wildfire is recorded in a California EPA Level III ecoregion on a given day.

The project started as a seasonal LightGBM study and was later reworked into a stricter development pipeline. The main update was not adding a more complicated model. It was separating feature selection, preprocessing, tuning, calibration, and final testing so that each stage answered one question without repeatedly looking at the test period.

## Current workflow

The scripts are numbered in the order I used them for the updated methodology:

1. `01_clean_weather.py` - parses the raw NOAA Global Hourly fields and creates the weather variables used later.
2. `02_aggregate_weather.py` - processes the station-year weather files and aggregates hourly observations to station-day summaries.
3. `03_build_dataset.py` - joins weather and wildfire records to 13 EPA Level III ecoregions and adds vegetation, population, road, and transmission information before writing the daily modelling dataset.
4. `04_data_audit.py` - checks panel integrity, missingness, prevalence, temporal drift, and strong feature redundancy.
5. `05_feature_ablation.py` - runs feature-family ablations with a fixed LightGBM model, confirms compact finalists across chronological folds, and compares feature sets with paired temporal block bootstrap.
6. `06_preprocessing_screen.py` - compares imbalance and missing-data strategies with fold-local preprocessing. Ecoregion is treated as categorical and synthetic oversampling uses SMOTENC rather than ordinary SMOTE.
7. `07_finalist_bootstrap.py` - compares the main preprocessing finalists using paired out-of-fold predictions and temporal block bootstrap.
8. `08_final_model_development.py` - tunes the frozen finalists, compares them on 2021-2022, evaluates calibration, selects an F2 threshold on 2022, and saves the frozen pre-test specification.
9. `09_final_test.py` - loads the frozen model and evaluates it once on the reserved 2023-2025 period. It does not retune, recalibrate, change features, or change the threshold.

`wildfire_config.py` holds the shared paths and the frozen 36-feature set. `wildfire_transformer.py` contains the small custom scaler used by the final pipelines.

## Data and split

The project uses weather observations from 2013-2025 together with wildfire, vegetation, ecoregion, road, transmission-line, and population data. The raw weather archive consisted of 2,043 station-year CSV files. The prediction unit in the final dataset is one ecoregion-day, with 13 Level III ecoregions represented for each date.

The main chronological split is:

- Training: 2013-2020
- Validation/development: 2021-2022
- Final out-of-time test: 2023-2025

The test period was inspected descriptively during the data audit, so I describe it as an out-of-time test rather than a perfectly blinded holdout. The final model-selection scripts do not use 2023-2025.

## Modelling decisions

Average Precision is the main ranking metric because wildfire occurrence is strongly imbalanced. Brier score and log loss are used to judge probability quality. Precision, recall, F1, and F2 are reported after a decision threshold is chosen. F2 is used for threshold selection because missing a fire is treated as more costly than producing an extra positive prediction.

The updated feature-selection process produced a compact 36-feature specification. Infrastructure variables were useful to investigate, but the primary model does not keep the unstable road, population, or transmission proxies. The preprocessing screen keeps native LightGBM missing-value handling as the simple control and compares it with selected imputation and SMOTENC pipelines.

## Running the project

The raw datasets are not included in this repository. See [`data/README.md`](data/README.md) for the expected inputs. Once the files are in place, run the numbered scripts in order.

The expensive screening and tuning stages can take several hours on a laptop. Generated models, predictions, caches, and intermediate datasets are intentionally ignored by Git.

Before running the full pipeline, I recommend running the small synthetic smoke test:

```bash
python smoke_test.py
```

`01_clean_weather.py` contains helper functions used by step 02 and does not need to be run separately. The full workflow is:

```bash
python 02_aggregate_weather.py
python 03_build_dataset.py
python 04_data_audit.py
python 05_feature_ablation.py
python 06_preprocessing_screen.py
python 07_finalist_bootstrap.py
python 08_final_model_development.py
python 09_final_test.py
```

For `06_preprocessing_screen.py`, set `STAGE` to `sampler_screen` first and then to `imputer_screen` after selecting the imbalance methods to carry forward.

## Methodology report

[`docs/California_Wildfire_Methodology_Report_2026-08-04.pdf`](docs/California_Wildfire_Methodology_Report_2026-08-04.pdf) documents the main methodological redevelopment through 4 August 2026. At that point final tuning, calibration, and the frozen test evaluation were still pending, so the later scripts in this repository continue beyond the report's cutoff date.

## Earlier version

The original seasonal LightGBM study was presented at CAC'26 and accepted for publication with Springer Nature. I have not kept the old Spring/Summer/Fall/Winter training scripts in this repository because the code here is meant to represent the updated methodology rather than every exploratory version of the project.

## Main limitations

- The target combines different wildfire ignition mechanisms rather than modelling cause-specific events.
- Human ignition opportunity is not represented directly in the final primary feature set.
- The same 13 ecoregions occur across all time periods, so this evaluates temporal transfer rather than transfer to unseen regions.
- The raw source data are too large to keep in the repository, so full reproduction requires downloading the original sources.

## Author

Alen Curovic
