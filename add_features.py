# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openap import prop
from openap.kinematic import WRAP

import airportsdata
from geopy.distance import great_circle

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin

import lightgbm as lgb
from sklearn.model_selection import train_test_split

# %%
df = pd.read_csv("../processed_ch_sub_final.csv")
df["submission"] = np.where(df["dataset"] == "challenge", 0, 1)
# %%
airports = airportsdata.load("icao")


def get_lat_lon(icao_code):
    airport_info = airports.get(icao_code)
    if airport_info:
        return airport_info["lat"], airport_info["lon"]
    else:
        return None, None


def calculate_great_circle(row):
    if pd.notnull(row["adep_lat"]) and pd.notnull(row["ades_lat"]):
        adep_coords = (row["adep_lat"], row["adep_lon"])
        ades_coords = (row["ades_lat"], row["ades_lon"])
        return great_circle(adep_coords, ades_coords).kilometers
    return None


df_airport = pd.read_csv("airports.csv")
df_airport

df["adep_lat"], df["adep_lon"] = zip(*df["adep"].apply(get_lat_lon))
df["ades_lat"], df["ades_lon"] = zip(*df["ades"].apply(get_lat_lon))

# Merge df2 with df_airport for ADEP (departure airport)
df = df.merge(
    df_airport[["gps_code", "latitude_deg", "longitude_deg"]],
    left_on="adep",
    right_on="gps_code",
    how="left",
).rename(columns={"latitude_deg": "new_adep_lat", "longitude_deg": "new_adep_lon"})

# # Fill NaN values in 'adep_lat' with the new values from 'new_adep_lat'
df["adep_lat"] = df["adep_lat"].fillna(df["new_adep_lat"])
df["adep_lon"] = df["adep_lat"].fillna(df["new_adep_lon"])

# # Dro the temporary 'new_adep_lat' and 'new_adep_lon' columns if they are not needed
df.drop(["new_adep_lat", "new_adep_lon", "gps_code"], axis=1, inplace=True)

# # Merge df2 with df_airport for ADEP (departure airport)
df = df.merge(
    df_airport[["gps_code", "latitude_deg", "longitude_deg"]],
    left_on="ades",
    right_on="gps_code",
    how="left",
).rename(columns={"latitude_deg": "new_ades_lat", "longitude_deg": "new_ades_lon"})

# # Fill NaN values in 'adep_lat' with the new values from 'new_adep_lat'
df["ades_lat"] = df["ades_lat"].fillna(df["new_ades_lat"])
df["ades_lon"] = df["ades_lon"].fillna(df["new_ades_lon"])

# # Drop the temporary 'new_adep_lat' and 'new_adep_lon' columns if they are not needed
df.drop(["new_ades_lat", "new_ades_lon", "gps_code"], axis=1, inplace=True)

df.loc[df["adep"] == "OKBK", "adep_lat"] = 29.226600646972656
df.loc[df["adep"] == "OKBK", "adep_lon"] = 47.96889877319336

df["great_circle_distance_km"] = df.apply(calculate_great_circle, axis=1)

df.to_csv("final_basic_features.csv", index=False)

# %%
# import climb features
df_climb = pd.read_csv("../final_sub_dataframe_with_climb_feats.csv")
new_columns = [col for col in df_climb.columns if col not in df.columns]

# Add only the new columns to df
df = df.join(df_climb[new_columns])

# df.to_csv("1s_final_sub_dataframe_with_climb_feat.csv", index = False)
df
# %%
# Add OpenAP features
df.drop(columns=["adep_lon", "adep_lat", "ades_lon", "ades_lat"], inplace=True)

# Create a dictionary to store cd0 values for each unique aircraft type
cd0_dict = {}
ceiling_dict = {}
cruise_dict = {}
cruise_mach_dict = {}

# Fetch cd0 for each unique aircraft type
for ac in df["aircraft_type"].unique():
    try:
        aircraft = prop.aircraft(ac)
        cd0_dict[ac] = aircraft["drag"]["cd0"]
        ceiling_dict[ac] = aircraft["ceiling"] * 3.28084
        cruise_dict[ac] = aircraft["cruise"]["height"] * 3.28084
        cruise_mach_dict[ac] = aircraft["cruise"]["mach"]
    except (KeyError, ValueError):
        # Handle cases where 'drag' or 'cd0' is missing
        cd0_dict[ac] = np.nan
        ceiling_dict[ac] = np.nan
        cruise_dict[ac] = np.nan
        cruise_mach_dict[ac] = np.nan

# Function to calculate drag term for each aircraft type


def air_density(altitude, temperature_kelvin):
    # Constants
    P0 = 101325  # Sea level standard atmospheric pressure (Pa)
    T0 = 288.15  # Sea level standard temperature (K)
    L = 0.0065  # Temperature lapse rate (K/m)
    R = 287.05  # Specific gas constant for dry air (J/(kg·K))
    g = 9.80665  # Gravity (m/s^2)
    M = 0.0289644  # Molar mass of Earth's air (kg/mol)

    # Calculate pressure at the given altitude (using barometric formula)
    pressure = P0 * (1 - (L * altitude) / T0) ** ((g * M) / (R * L))

    # Calculate air density using the ideal gas law
    density = pressure / (R * temperature_kelvin)

    return density


def calculate_drag(row):
    # Retrieve cd0 value from the dictionary
    cd0 = cd0_dict.get(row["aircraft_type"], np.nan)

    # Calculate drag terms, handle missing cd0 with NaN
    drag_term1 = cd0 * (row["avg_tas_mps_0_5000"] ** 2) / 2 if pd.notna(cd0) else np.nan
    drag_term2 = (
        cd0 * (row["avg_tas_mps_5000_10000"] ** 2) / 2 if pd.notna(cd0) else np.nan
    )
    drag_term3 = (
        cd0 * (row["avg_tas_mps_10000_15000"] ** 2) / 2 if pd.notna(cd0) else np.nan
    )
    drag_term4 = (
        cd0 * (row["avg_tas_mps_15000_20000"] ** 2) / 2 if pd.notna(cd0) else np.nan
    )

    return drag_term1, drag_term2, drag_term3, drag_term4


def calculate_ceiling_diff(row):
    max_ceiling = ceiling_dict.get(row["aircraft_type"], np.nan)
    cruise = cruise_dict.get(row["aircraft_type"], np.nan)

    ceiling_diff = max_ceiling - row["max_cruise_altitude"]
    cruise_diff = cruise - row["mean_cruise_altitude"]
    return ceiling_diff, cruise_diff


def get_average_mach(row):
    # Extract the mean cruise temperature and TAS from the row
    temperature_kelvin = row["mean_cruise_temperature"]
    tas = row["mean_cruise_tas"]  # Assume this is in meters per second (m/s)

    # Calculate the speed of sound using the formula
    gamma = 1.4  # Adiabatic index for air
    R = 287.05  # Specific gas constant for air (J/(kg·K))
    speed_of_sound = np.sqrt(gamma * R * temperature_kelvin)

    # Calculate Mach number
    mach_number = tas / speed_of_sound

    mach_nominal = cruise_mach_dict.get(row["aircraft_type"], np.nan)
    mach_diff = mach_number - mach_nominal

    return mach_number, mach_diff


# Apply the function to each row of the DataFrame and unpack the results into two new columns
df[["drag_term1", "drag_term2", "drag_term3", "drag_term4"]] = df.apply(
    calculate_drag, axis=1, result_type="expand"
)
df[["ceiling_diff", "cruise_diff"]] = df.apply(
    calculate_ceiling_diff, axis=1, result_type="expand"
)
df[["get_average_mach", "mach_diff"]] = df.apply(
    get_average_mach, axis=1, result_type="expand"
)

# df.to_csv("1s_final_sub_dataframe_with_climb_feat.csv", index = False)
# %%
# df = pd.read_csv("final_sub_only_fuel_spent.csv")

cruise_range_dict = {}
climb_range_dict = {}

climb_range_openap_dict = {}
cruise_range_openap_dict = {}

for ac in df["aircraft_type"].unique():
    try:
        climb_range_openap_dict[ac] = WRAP(ac=ac).climb_range()
        cruise_range_openap_dict[ac] = WRAP(ac=ac).cruise_range()
    except:
        climb_range_openap_dict[ac] = {"default": np.nan}
        cruise_range_openap_dict[ac] = {"default": np.nan}

for ac in df["aircraft_type"].unique():
    flown_distance = df[df["aircraft_type"] == ac]["flown_distance"]
    climb_range = df[df["aircraft_type"] == ac]["climb_range"]

    default_cruise_range = (
        cruise_range_openap_dict[ac]["default"]
        if not np.isnan(cruise_range_openap_dict[ac]["default"])
        else np.nan
    )
    default_climb_range = (
        climb_range_openap_dict[ac]["default"]
        if not np.isnan(climb_range_openap_dict[ac]["default"])
        else np.nan
    )

    df.loc[df["aircraft_type"] == ac, "climb_range_normalized"] = (
        climb_range / default_climb_range
    )
    df.loc[df["aircraft_type"] == ac, "flown_distance_normalized"] = (
        flown_distance - climb_range
    ) / default_cruise_range

# %%
for ac in df["aircraft_type"].unique():
    if (
        df.loc[df["aircraft_type"] == ac, "flown_distance_normalized"].isna().sum()
        > 1000
    ):
        median_flown_distance = df.loc[
            df["aircraft_type"] == ac, "flown_distance"
        ].median()
        df.loc[df["aircraft_type"] == ac, "flown_distance_normalized"] = (
            df.loc[df["aircraft_type"] == ac, "flown_distance"] / median_flown_distance
        )

    if df.loc[df["aircraft_type"] == ac, "climb_range_normalized"].isna().sum() > 1000:
        median_climb_range = df.loc[df["aircraft_type"] == ac, "climb_range"].median()
        df.loc[df["aircraft_type"] == ac, "climb_range_normalized"] = (
            df.loc[df["aircraft_type"] == ac, "climb_range"] / median_climb_range
        )

    df.loc[df["aircraft_type"] == ac, "max_flown_dist"] = df.loc[
        df["aircraft_type"] == ac, "climb_range"
    ].max()
    df.loc[df["aircraft_type"] == ac, "min_flown_dist"] = df.loc[
        df["aircraft_type"] == ac, "climb_range"
    ].min()

    df.loc[df["aircraft_type"] == ac, "max_climb_range"] = df.loc[
        df["aircraft_type"] == ac, "climb_range"
    ].max()
    df.loc[df["aircraft_type"] == ac, "min_climb_range"] = df.loc[
        df["aircraft_type"] == ac, "climb_range"
    ].min()


# %%
for col in df.columns:
    if col != "tow":
        nb_nan = df[col].isna().sum()
        if nb_nan > 400000:
            df = df.drop(columns=col)
        # if('mass_loss' in col):
        #     df = df.drop(columns=col)
        if "thrust" in col:
            df = df.drop(columns=col)
        if ("fuelflow" in col) and ("cruise" not in col):
            df = df.drop(columns=col)

df.to_csv("final_sub_normalized.csv", index=False)
# %%
## Imputation

df = pd.read_csv("final_sub_normalized.csv")

# Step 1: Calculate the number of NaN values per column
nan_counts = df.isna().sum()

# Step 2: Filter the features that have more than 10,000 NaNs
feature_list = nan_counts[nan_counts > 0].index.tolist()

# Step 3: Create a dictionary to store the top 10 correlated features for each column
top_10_correlated = {}

# Filter the numerical columns
numerical_df = df.select_dtypes(include=["float64", "int64"])

# Calculate the correlation matrix for the numerical columns
correlation_matrix = numerical_df.corr()

# Step 4: For each feature in feature_list, find the top 10 most correlated features
for feature in feature_list:
    # Calculate correlation for the feature
    corr = correlation_matrix[feature]

    # Find top 10 correlated features, excluding the feature itself
    top_10_corr = corr.drop(feature).abs().sort_values(ascending=False).head(50)

    # Store the top 10 features for this particular feature
    top_10_correlated[feature] = top_10_corr.index.tolist()


# %%
# Placeholder for LightGBM model predictions
def impute_with_lightgbm(df, feature_list, top_10_correlated):
    counter = 0
    for feature in feature_list:
        if feature != "tow":
            counter += 1
            print(f"Imputing {feature}, {counter}/{len(feature_list)}")
            # Step 1: Get the top 10 correlated features for the current feature
            correlated_features = top_10_correlated[feature]

            # Step 2: Create a dataset with only the correlated features as predictors
            X = df[correlated_features]
            y = df[feature]

            # Step 3: Split into training set where the feature is not NaN
            X_train = X[~y.isna()]
            y_train = y[~y.isna()]

            # Step 4: Train a LightGBM model
            train_data = lgb.Dataset(X_train, label=y_train)

            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "max_depth": 5,
                "verbose": -1,
            }

            # Train the model
            model = lgb.train(params, train_data, num_boost_round=100)

            # Step 5: Identify the missing values in the target feature
            missing_values_index = df[feature].isna()

            # Step 6: Predict the missing values using the trained model
            df.loc[missing_values_index, feature] = model.predict(
                df.loc[missing_values_index, correlated_features]
            )

            print(f"Filled missing values for feature: {feature}")

    return df


# Call the function to impute missing values using LightGBM models
df = impute_with_lightgbm(df, feature_list, top_10_correlated)
# %%

df.to_csv("final_sub_imputed.csv", index=False)
