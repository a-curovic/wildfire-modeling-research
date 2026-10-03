# Data files

The raw data are not committed because the complete project data are large and come from several external sources.

The scripts expect this general layout:

```text
data/
  raw/
    ncei_global_hourly_CA_2013_2025/
    ca_eco_l3.shp (+ accompanying shapefile files)
    mapdataall_cleaned.csv
    TransmissionLine_CEC.shp (+ accompanying files)
    Counties.shp (+ accompanying files)
    CaCounty2010-2020.xlsx
    CaCounty2020-2025.xlsx
  interim/
    road_density_ecoregion_year_2013_2025.csv
    weather_daily_agg.csv
    vegetation_tifs/
    vegetation_stats_parts/
  processed/
    fullDataSet3.csv
```

Vegetation is local-first. If `data/interim/vegetation_stats_parts/` already contains the previously generated monthly ecoregion statistics, `03_build_dataset.py` reuses them and does not need Rasterio or AWS. Otherwise place NDVI/EVI GeoTIFFs in `data/raw/vegetation_tifs/` (or `data/interim/vegetation_tifs/`). S3 is only an optional fallback when `WILDFIRE_VEGETATION_S3_BUCKET` is explicitly set.

If you already have the final `fullDataSet3.csv`, place it in `data/processed/` and start the modelling workflow at `04_data_audit.py`; rebuilding all raw geospatial inputs is not required just to reproduce the later model-development stages.
