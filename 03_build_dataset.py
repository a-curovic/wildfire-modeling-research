"""Build the ecoregion-day dataset used by the updated wildfire methodology.

This combines daily weather, fire incidents, ecoregion geometry, infrastructure,
population, and monthly vegetation features. Raw source files are intentionally
kept outside the public repository.
"""

import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
import boto3
import re
import time
import rasterio
#from rasterstats import zonal_stats
#from rasterio.io import MemoryFile
import os
import gc
import json
#from shapely.ops import unary_union
from rasterio.features import rasterize
from wildfire_config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR

gpd.options.io_engine = "fiona"
def assert_unique_keys(data: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicates = data.loc[data.duplicated(keys, keep=False),keys]
    
    if not duplicates.empty:
        raise ValueError(f"{name} contains duplicate merge keys:\n"
                         f"{duplicates.head(20).to_string(index=False)}")


eco = gpd.read_file(RAW_DIR / "ca_eco_l3.shp").to_crs("EPSG:4326")
eco = eco[["US_L3CODE","US_L3NAME","geometry"]].dropna(subset=['geometry']).copy()
eco["US_L3CODE"] = (eco["US_L3CODE"].astype(str).str.strip())
eco["US_L3NAME"] = (eco["US_L3NAME"].astype(str).str.strip())

eco['geometry'] = shapely.make_valid(eco['geometry'])
eco = eco.dissolve(by=["US_L3CODE","US_L3NAME"], as_index=False)

assert len(eco) == 13, (f"Expected exactly 13 dissolved ecoregions, found {len(eco)}")

eco_metric = eco.to_crs("EPSG:3310")
eco["area_km2"] = eco_metric.area / 1e6

print(eco)
start_date = "2013-01-01"
end_date = "2025-08-24"
calendar = pd.DataFrame({
    "date": pd.date_range(start_date,end_date, freq="D").date
})


print("[eco] area stats (km²):", eco["area_km2"].describe()[["min","mean","max"]].to_dict())

ecoregions = eco[["US_L3CODE","US_L3NAME"]].drop_duplicates().sort_values("US_L3CODE").reset_index(drop=True) 
print("Number of Unique ecoregions:", len(ecoregions))
print(ecoregions.to_string(index=False)) 

assert len(ecoregions) == 13, (f"Expected 13 ecoregions, found {len(ecoregions)}")

group_keys = ["US_L3CODE","US_L3NAME","date"]

base = calendar.merge(ecoregions, how="cross")

assert not base.duplicated(group_keys).any()
assert base.groupby("date").size().eq(13).all()

expected_base_rows = len(calendar) * 13

assert len(base) == expected_base_rows, (f"Expected {expected_base_rows} rows, found {len(base)}")

weather = pd.read_csv(INTERIM_DIR / "weather_daily_agg.csv")
weather['date'] = pd.to_datetime(weather['date'],errors="coerce").dt.date
weather = weather.dropna(subset=["date","LATITUDE","LONGITUDE"]).copy()

weather_geo = gpd.GeoDataFrame(weather, geometry=gpd.points_from_xy(weather['LONGITUDE'],weather['LATITUDE']),
                               crs = "EPSG: 4326")

weather_join = gpd.sjoin(weather_geo,eco[['US_L3CODE','US_L3NAME','geometry']], how='inner',
                         predicate='within')
for junk in ("index_right",'index_left'):
    if junk in weather_join.columns:
        weather_join.drop(columns=junk,inplace=True)

print(f"[join] after sjoin rows: {len(weather_join)}; columns: {len(weather_join.columns)}")
print(weather_join.columns)

numeric_cols = (weather_join
                .select_dtypes(include=[np.number])
                .columns.difference(["LATITUDE","LONGITUDE",*group_keys]))


weather_agg = (weather_join.groupby(group_keys,dropna=False)[numeric_cols].mean().reset_index())

calendar_dates = set(calendar["date"])
weather_dates = set(weather_agg["date"])

dates_without_any_weather = sorted(
    calendar_dates - weather_dates
)

if dates_without_any_weather:
    raise ValueError(
        "Dates with no weather data for any ecoregion:\n"
        f"{dates_without_any_weather[:20]}"
    )
fire_csv = pd.read_csv(RAW_DIR / "mapdataall_cleaned.csv")
fire_csv = fire_csv.rename(columns={'incident_longitude':'longitude','incident_latitude':'latitude', 'incident_dateonly_created':'date'})

fire_csv['date'] = pd.to_datetime(fire_csv['date'], errors="coerce").dt.date
fire_csv = fire_csv.dropna(subset=['date','latitude','longitude'])

fires_gdf = gpd.GeoDataFrame(fire_csv, geometry=gpd.points_from_xy(fire_csv['longitude'],fire_csv['latitude']), crs="EPSG:4326")
fires_join = gpd.sjoin(fires_gdf, eco[["US_L3CODE","US_L3NAME","geometry"]], how="inner", predicate="within")

for junk in ("index_right", "index_left"):
    if junk in fires_join.columns:
        fires_join.drop(columns=junk, inplace=True)

fires_daily = (fires_join.groupby(group_keys,dropna=False).size().reset_index(name="fire_count"))

fires_daily["fire_occurred"] = (fires_daily['fire_count'] > 0).astype(int)


df = base.merge(weather_agg, on=group_keys, how="left", validate="1:1").merge(fires_daily, on=group_keys, how="left", validate="1:1")

#base = weather_agg[["US_L3CODE","US_L3NAME","date"]].drop_duplicates()
print("\nbase YEARS")
print(pd.to_datetime(base["date"]).dt.year.value_counts().sort_index())
df["fire_count"] = df["fire_count"].fillna(0).astype(int)
df["fire_occurred"] = df["fire_occurred"].fillna(0).astype(int)
df["year"] = pd.to_datetime(df["date"]).dt.year.astype(int)

###
eco_area = eco_metric[["US_L3CODE","US_L3NAME","geometry"]].copy()
eco_area["area_km2"] = eco_metric.area / 1e6

t=time.perf_counter()

road_density_all = pd.read_csv(INTERIM_DIR / "road_density_ecoregion_year_2013_2025.csv", dtype={"US_L3CODE": str})
road_density_all["US_L3CODE"] = road_density_all["US_L3CODE"].str.strip()
road_keys = ["US_L3CODE","year"]

assert_unique_keys(road_density_all, road_keys, "road_density_all")

df = df.merge(road_density_all[["US_L3CODE","year","road_km_per_100km2"]],
                         on=road_keys,
                         how="left",
                         validate="m:1")


#---transmission lines---

trans_lines = gpd.read_file(RAW_DIR / "TransmissionLine_CEC.shp").to_crs(4326)
trans_lines_m = trans_lines.to_crs("EPSG:3310")
print("before trans overlay", time.perf_counter()-t); t=time.perf_counter()
trans_in_eco = gpd.overlay(trans_lines_m,eco_metric[["US_L3CODE","US_L3NAME","geometry"]], how="intersection")
print("after trans overlay", time.perf_counter()-t); t=time.perf_counter()
trans_in_eco['trans_km'] = trans_in_eco.length / 1000.0

trans_km_by_eco = (trans_in_eco.groupby(["US_L3CODE","US_L3NAME"])['trans_km'].sum().reset_index())
trans_km_by_eco = trans_km_by_eco.merge(eco_area[["US_L3CODE","area_km2"]],on="US_L3CODE", how="left")
trans_km_by_eco['trans_km_per_100km2'] = (trans_km_by_eco['trans_km']/trans_km_by_eco['area_km2']) *100.0

trans_density = trans_km_by_eco[["US_L3CODE","trans_km_per_100km2"]]
assert_unique_keys(trans_density,["US_L3CODE"], "trans_density")
df = df.merge(trans_density, on="US_L3CODE", how="left", validate="m:1")

#population 
def load_population(path, countyColumn = "Counties"):
    pop = pd.read_excel(path)

    pop.columns = [str(c).strip() for c in pop.columns]

    pop = pop.rename(columns={countyColumn: "CountyName"})

    pop["CountyName"] = pop["CountyName"].astype(str).str.strip()

    yearColumns = [c for c in pop.columns if str(c).isdigit()]

    popLong = pop[["CountyName"] + yearColumns].melt(
        id_vars = "CountyName",
        var_name="year",
        value_name="Population"
    )

    popLong['year'] = popLong['year'].astype(int)
    popLong["Population"] = pd.to_numeric(popLong['Population'],errors="coerce")

    return popLong

pop2010_2020 = load_population(RAW_DIR / "CaCounty2010-2020.xlsx")
pop2021_2025 = load_population(RAW_DIR / "CaCounty2020-2025.xlsx")


fullPopulation = pd.concat([pop2010_2020,pop2021_2025],ignore_index=True)
fullPopulation = fullPopulation[fullPopulation["year"].between(2013,2025)].copy()

print("[fullPopulation] shape:", fullPopulation.shape)
print(fullPopulation.head())

counties = gpd.read_file(RAW_DIR / "Counties.shp").to_crs("EPSG:3310")
counties["CountyName"] = counties["CountyName"].astype(str).str.strip()

counties['county_area_km2'] = counties.area / 1e6
print("Counties geometry loaded:", counties.shape)

print("before population overlay", time.perf_counter()-t); t=time.perf_counter()

inter = gpd.overlay(
    counties[["CountyName","geometry"]],
    eco_area[["US_L3CODE","US_L3NAME","geometry"]],
    how="intersection"
)
print("After Population overlay", time.perf_counter()-t); t=time.perf_counter()

inter["overlap_km2"] = inter.area / 1e6

inter["weight"] = (inter["overlap_km2"] / inter.groupby("CountyName")["overlap_km2"].transform("sum"))

print(inter[["CountyName","US_L3CODE","US_L3NAME","weight"]].head())

interPop = inter.merge(fullPopulation,on="CountyName", how="left")
interPop["popWeight"] = interPop["Population"] * interPop["weight"]

population_l3_year = (interPop.groupby(["US_L3CODE","US_L3NAME","year"],as_index=False)["popWeight"].sum().rename(columns={"popWeight":"population"}))

population_l3_year = population_l3_year.merge(eco_area[["US_L3CODE","area_km2"]].drop_duplicates(),
                                              on="US_L3CODE",
                                              how="left")

population_l3_year["population_density"] = ( population_l3_year["population"] / population_l3_year["area_km2"])

print(f"[population] computed density for {len(population_l3_year)} ecoregion-year rows")
print(population_l3_year.head())

df["year"] = df["year"].astype(int)
print("before last merger overlay", time.perf_counter()-t); t=time.perf_counter()

population_keys = ["US_L3CODE","year"]
assert_unique_keys(population_l3_year, population_keys, "population_l3_year")

df = df.merge(population_l3_year[["US_L3CODE", "year", "population_density"]],
    on=population_keys,
    how="left",
    validate="m:1")


### NVDI and EVI merge

s3 = boto3.client("s3")
bucket = os.getenv("WILDFIRE_VEGETATION_S3_BUCKET")
prefix = os.getenv("WILDFIRE_VEGETATION_S3_PREFIX", "California vegetation indices/")
if not bucket:
    raise RuntimeError(
        "Set WILDFIRE_VEGETATION_S3_BUCKET before downloading vegetation rasters."
    )



def list_tifs(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not(key.lower().endswith(".tif") or key.lower().endswith(".tiff")):
                continue
            
            if "_ndvi_" in key.lower() or "_evi_" in key.lower():
                keys.append(key)

    return sorted(keys)

keys = list_tifs(bucket, prefix)
print("Total tif files found:", len(keys))
print("First 5 keys:", keys[:5])


veg_dir = INTERIM_DIR / "vegetation_tifs"
veg_dir.mkdir(exist_ok=True)

def local_path_for_key(key):
    return veg_dir / key.split("/")[-1]

def download_vegetation_files_once(bucket, keys, s3):
    for i, key in enumerate(keys, start=1):
        local_path = local_path_for_key(key)

        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"[download {i}/{len(keys)}] already exists: {local_path.name}", flush=True)
            continue

        print(f"[download {i}/{len(keys)}] downloading: {local_path.name}", flush=True)
        s3.download_file(bucket, key, str(local_path))
        

download_vegetation_files_once(bucket, keys, s3)


def build_zone_raster(example_tif_path, eco_gdf):
    with rasterio.open(example_tif_path) as src:
        eco_proj = eco_gdf.to_crs(src.crs).copy().reset_index(drop=True)

        shapes = []

        for zone_id, row in eco_proj.iterrows():
            geom = shapely.make_valid(row.geometry)

            if geom is None or geom.is_empty:
                continue

            if geom.geom_type not in ("Polygon", "MultiPolygon"):
                continue

            geom_json = json.loads(shapely.to_geojson(geom))
            shapes.append((geom_json, zone_id + 1))

        zone_raster = rasterize(
            shapes=shapes,
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            dtype="int16"
        )

        raster_info = {
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
        }

    return zone_raster, eco_proj, raster_info

def parse_file_info(key):
    fname = key.split("/")[-1].lower()

    # Matches:
    # MOD13A3.061__1_km_monthly_EVI_20130101T000000_aid0001.tif
    # MOD13A3.061__1_km_monthly_NDVI_20150301T000000_aid0001.tif
    m = re.search(r"_(ndvi|evi)_(\d{8})t\d{6}", fname)
    if not m:
        raise ValueError(f"Could not parse filename: {fname}")

    index_type = m.group(1)   # ndvi or evi
    yyyymmdd = m.group(2)     # e.g. 20130101

    year = int(yyyymmdd[:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])

    return index_type, year, month, day

def zonal_features_from_local_tif_fast(tif_path, eco_proj, zone_raster, expected_info):
    with rasterio.open(tif_path) as src:
        # Safety check: all rasters should match the grid used to build zone_raster
        if src.shape != expected_info["shape"]:
            raise ValueError(f"Raster shape changed for {tif_path}: {src.shape}")

        if src.transform != expected_info["transform"]:
            raise ValueError(f"Raster transform changed for {tif_path}")
        
        if src.crs != expected_info["crs"]:
            raise ValueError(f"Raster CRS changed for {tif_path}")

        fill_value = src.nodata if src.nodata is not None else -3000
        band = src.read(1)

    valid = np.ones(band.shape, dtype=bool)

    if fill_value is not None:
        valid &= band != fill_value

    valid &= np.isfinite(band)

    rows = []

    for zone_id, row in eco_proj.iterrows():
        zone_value = zone_id + 1
        zone_mask = (zone_raster == zone_value) & valid
        vals = band[zone_mask]

        if vals.size == 0:
            mean_val = std_val = min_val = max_val = np.nan
        else:
            vals = vals.astype("float64")
            mean_val = np.mean(vals) * 0.0001
            std_val = np.std(vals) * 0.0001
            min_val = np.min(vals) * 0.0001
            max_val = np.max(vals) * 0.0001

        rows.append({
            "US_L3CODE": row["US_L3CODE"],
            "US_L3NAME": row["US_L3NAME"],
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
        })

    return pd.DataFrame(rows)
# Make sure df has year and month for daily->monthly merge
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date"].dt.month.astype(int)

first_local_path = local_path_for_key(keys[0])
zone_raster, eco_proj_for_veg, raster_info = build_zone_raster(first_local_path, eco)


veg_stats_dir = INTERIM_DIR / "vegetation_stats_parts"
veg_stats_dir.mkdir(exist_ok=True)

frames = []

for i, key in enumerate(keys, start=1):
    try:
        print(f"\n[{i}/{len(keys)}] START {key}", flush=True)

        index_type, year, month, day = parse_file_info(key)

        if index_type == "ndvi" and (year > 2024 or (year == 2024 and month > 7)):
            print(f"[{i}/{len(keys)}] SKIP NDVI after 2024-07", flush=True)
            continue

        start = time.perf_counter()

        local_path = local_path_for_key(key)
        
        part_path = veg_stats_dir / f"{local_path.stem}_stats.csv"

        if part_path.exists() and part_path.stat().st_size > 0:
            print(f"[{i}/{len(keys)}] SKIP already processed {part_path.name}", flush=True)
            continue

        z = zonal_features_from_local_tif_fast(
            local_path,
            eco_proj_for_veg,
            zone_raster,
            raster_info
        )

        print(
            f"[{i}/{len(keys)}] DONE in {time.perf_counter() - start:.2f}s",
            flush=True
        )

        z["year"] = year
        z["month"] = month

        z = z.rename(columns={
            "mean": f"{index_type}_mean",
            "std": f"{index_type}_std",
            "min": f"{index_type}_min",
            "max": f"{index_type}_max",
        })

        z.to_csv(part_path, index=False)
        frames.append(z)

        del z
        gc.collect()

    except Exception as e:
        print(f"\n[{i}/{len(keys)}] ERROR on {key}", flush=True)
        print("Exception type:", type(e).__name__, flush=True)
        print("Exception:", repr(e), flush=True)
        raise

part_files = sorted(veg_stats_dir.glob("*_stats.csv"))
veg_long = pd.concat([pd.read_csv(p) for p in part_files], ignore_index=True)

veg_long["US_L3CODE"] = veg_long["US_L3CODE"].astype(str).str.strip()
veg_long["US_L3NAME"] = veg_long["US_L3NAME"].astype(str).str.strip()
veg_long["year"] = veg_long["year"].astype(int)
veg_long["month"] = veg_long["month"].astype(int)

ndvi_df = veg_long[
    ["US_L3CODE", "US_L3NAME", "year", "month",
     "ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max"]
].dropna(how="all", subset=["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max"]).drop_duplicates()

evi_df = veg_long[
    ["US_L3CODE", "US_L3NAME", "year", "month",
     "evi_mean", "evi_std", "evi_min", "evi_max"]
].dropna(how="all", subset=["evi_mean", "evi_std", "evi_min", "evi_max"]).drop_duplicates()

vegetation_keys = ["US_L3CODE","US_L3NAME","year","month"]

assert_unique_keys(ndvi_df, vegetation_keys, "ndvi_df")
assert_unique_keys(evi_df, vegetation_keys, "evi_df")

veg_monthly = ndvi_df.merge(
    evi_df,
    on = vegetation_keys,
    how ="outer",
    validate="1:m")

assert_unique_keys(veg_monthly, vegetation_keys, "veg_monthly")

df["US_L3CODE"] = df["US_L3CODE"].astype(str).str.strip()
df["US_L3NAME"] = df["US_L3NAME"].astype(str).str.strip()
df["year"] = df["year"].astype(int)
df["month"] = df["month"].astype(int)

df = df.merge(
    veg_monthly,
    on=vegetation_keys,
    how="left",
    validate="m:1")

print("Vegetation monthly shape:", veg_monthly.shape)
print("Final dataset shape:", df.shape)
print("NDVI coverage:", df.loc[df["ndvi_mean"].notna(), "date"].min(),
      df.loc[df["ndvi_mean"].notna(), "date"].max())
print("EVI coverage:", df.loc[df["evi_mean"].notna(), "date"].min(),
      df.loc[df["evi_mean"].notna(), "date"].max())
###


####
#Full check
print("shape",df.shape)                             # rows, columns
print("columns",df.columns.tolist())                    # column names
print("sample",df.sample(5, random_state=0) )          # spot-check a few rows
print("info",df.info())                              # dtypes + non-null counts
print("numeric summary",df.describe().T)                        # numeric summary
print("text/categorical summary",df.describe(include='object').T )       # text/categorical summary
print("top missing rates",df.isna().mean().sort_values(ascending=False).head(20) )  # top missing rates
print("dup eco-day row",df.duplicated(["US_L3CODE","US_L3NAME","date"]).sum() )   # dup eco-day rows
print("date span",df["date"].min(), df["date"].max())     # date span
print("fire occured",df["fire_occurred"].value_counts(dropna=False, normalize=True) if "fire_occurred" in df else None)

print("Final dataset", df.shape)
print("final dataset columns", df.columns)
print("final dataset missing values", df.isna().sum())

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

df['season'] = df['date'].dt.month.apply(get_season)


def add_weather_fire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived meteorological and human-exposure features.

    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year ####

    # Saturation vapor pressure using temperature mean, in kPa.
    df["sat_vp"] = 0.61094 * np.exp(
        (17.625 * df["temp_c_mean"]) / (df["temp_c_mean"] + 243.04)
    )

    # Actual vapor pressure using dew point mean, in kPa.
    df["A_sat_vp"] = 0.61094 * np.exp(
        (17.625 * df["dew_point_c_mean"]) / (df["dew_point_c_mean"] + 243.04)
    )

    # Relative humidity as a percentage.
    df["RH_pct"] = ((df["A_sat_vp"] / df["sat_vp"]) * 100).clip(lower=0, upper=100)

    # Vapor pressure deficit in kPa.
    df["VPD_kpa"] = (df["sat_vp"] - df["A_sat_vp"]).clip(lower=0)

    # Hot-Dry-Windy proxy.
    df["HDWI_proxy"] = df["VPD_kpa"] * df["wind_speed_ms_max"]

    # Diurnal temperature range.
    df["DTR"] = df["temp_c_max"] - df["temp_c_min"]

    # Utility risk proxy: wind acting on transmission-line density.
    df["transmission_risk"] = df["trans_km_per_100km2"] * df["wind_speed_ms_max"]

    # Road access normalized by population density.
    df["roads_per_pop_density"] = (
        df["road_km_per_100km2"] / (df["population_density"] + 1e-8)
    )

    return df.sort_values(["date", "US_L3CODE"]).reset_index(drop=True)

df = add_weather_fire_features(df)

expected_n_ecoregions = len(ecoregions)

# Exactly one row per ecoregion-date
duplicate_count = df.duplicated(group_keys).sum()

assert duplicate_count == 0, (
    f"Found {duplicate_count} duplicate ecoregion-date rows"
)

# Exactly 13 rows per date
rows_per_date = df.groupby("date").size()

bad_date_counts = rows_per_date[
    rows_per_date != expected_n_ecoregions
]

assert bad_date_counts.empty, (
    "Some dates do not contain all ecoregions:\n"
    f"{bad_date_counts.head(20)}"
)

# Exactly 13 unique codes per date
codes_per_date = (
    df
    .groupby("date")["US_L3CODE"]
    .nunique()
)

assert codes_per_date.eq(expected_n_ecoregions).all()

# Expected total number of rows
expected_rows = (
    df["date"].nunique()
    * expected_n_ecoregions
)

assert len(df) == expected_rows, (
    f"Expected {expected_rows} rows, found {len(df)}"
)

print("Panel integrity passed.")
print("Dates:", df["date"].nunique())
print("Ecoregions:", expected_n_ecoregions)
print("Rows:", len(df))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DIR / "fullDataSet3.csv", index=False)


