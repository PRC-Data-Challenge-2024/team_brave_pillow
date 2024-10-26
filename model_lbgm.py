# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import lightgbm as lgbm
from lightgbm import LGBMRegressor as lgbmreg


# %%
#### -----------Functions here----------- ####
def classify_time(hour, nb_of_partition=24):
    range_of_hour = 24 / nb_of_partition

    return hour // range_of_hour


# %%
dff = pd.read_csv("processed_ch_sub_final.csv")
# print(df_challenge_processed.columns)
dff["adepdespair"] = dff["adep"] + dff["ades"]
dff["countrypair"] = dff["country_code_adep"] + dff["country_code_ades"]

#### -----------Features Processing----------- ###
print("#### -----------Features Processing----------- ###")

# Convert 'actual_offblock_time' to datetime
dff["actual_offblock_time"] = pd.to_datetime(dff["actual_offblock_time"])
dff["arrival_time"] = pd.to_datetime(dff["arrival_time"])

# Extract the hour
dff["hour_offblock"] = dff["actual_offblock_time"].dt.hour
dff["hour_arrival"] = dff["arrival_time"].dt.hour

dff["time_of_day_offblock"] = dff["hour_offblock"].apply(classify_time)
dff["time_of_day_arrival"] = dff["hour_arrival"].apply(classify_time)

# Extract the day
dff["date"] = pd.to_datetime(dff["date"], format="%Y-%m-%d")
dff["day_of_year"] = dff["date"].dt.dayofyear

# Convert float64 to float32, might boost the training
dff = dff.astype(
    {
        col: "float32"
        for col in dff.drop(columns="flight_id").select_dtypes(include="float").columns
    }
)

dff = dff.drop(columns=["hour_offblock", "hour_arrival", "date"])
dff["adepdespair"] = pd.factorize(dff["adepdespair"])[0]
dff["airline"] = pd.factorize(dff["airline"])[0]
dff["aircraft_type"] = pd.factorize(dff["aircraft_type"])[0]
dff["countrypair"] = pd.factorize(dff["countrypair"])[0]
dff["adep"] = pd.factorize(dff["adep"])[0]
dff["ades"] = pd.factorize(dff["ades"])[0]
dff["country_code_adep"] = pd.factorize(dff["country_code_adep"])[0]
dff["country_code_ades"] = pd.factorize(dff["country_code_ades"])[0]
dff["wtc"] = pd.factorize(dff["wtc"])[0]

dff = dff.drop(
    columns=[
        "ac",
        "callsign",
        "name_adep",
        "name_ades",
        "actual_offblock_time",
        "arrival_time",
    ]
)
ct_features = [
    "adep",
    "country_code_adep",
    "ades",
    "country_code_ades",
    "aircraft_type",
    "airline",
    "adepdespair",
    "countrypair",
]
dff[ct_features] = dff[ct_features].astype("category")
dff.to_csv("dff_lgb.csv", index=False)
# %%
############### Train-Test check #################
dff = pd.read_csv("dff_lgb.csv")
# Drop extra features
dff = dff.drop(
    columns=[
        "mean_descent_thrust",  ####-----bad
        "max_climb_vertacc",  ####-----bad
        "mean_climb_mass_loss",  ####-----bad
        "mean_initclimb_mass_loss",  ####-----bad
        "mean_neg_descent_vertacc",  ####-----bad
        "mean_pos_initclimb_vertacc",  ####-----bad
        "min_descent_vertacc",  ####-----bad
        "min_rod",  ####-----bad
        "mean_rod",  ####-----bad
        "median_rod",  ####-----bad
        "dur_descent",  ####-----bad
        "mean_cruise_humidity",  ####-----bad
        "mean_cruise_temperature",  ####-----bad
        "max_initclimb_vertacc",  ####-----bad
        "dur_cruise2",
        "mean_descent_tas",  ####-----bad
        "time_of_day_offblock",
        "max_descent_tas",  ####-----bad
        "dur_cruise",
        "taxiout_time",
    ]
)
ct_features = [
    "adep",
    "country_code_adep",
    "ades",
    "wtc",
    "country_code_ades",
    "aircraft_type",
    "airline",
    "adepdespair",
    "countrypair",
]
dff[ct_features] = dff[ct_features].astype("category")
# %%     --------takes about 1.5 minutes
# initialize data
X = dff.query("dataset=='challenge'").drop(columns=["tow", "dataset"])
y = dff.query("dataset=='challenge'")["tow"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)
X_train = X_train.drop(columns=["flight_id"])
column_indices = [X_train.columns.get_loc(col) for col in ct_features]

model = lgbmreg(
    reg_alpha=0.17,
    reg_lambda=0.03,
    num_leaves=215,
    learning_rate=0.08,
    n_estimators=800,
    colsample_bytree=0.5,
    importance_type="gain",
    force_col_wise=True,
)

model.fit(
    X_train,
    y_train,
    categorical_feature=ct_features,
)
preds = model.predict(
    X_test.drop(columns=["flight_id"]), categorical_feature=ct_features
)
print("RMSE:", rmse(y_test, preds))


# %% FINAL SUBMISSION SET --------takes about 1.5 minutes
######### Training on full challenge dataset and predicting FINAL set #########

X = dff.query("dataset=='challenge'").drop(columns=["tow", "dataset"])
y = dff.query("dataset=='challenge'")["tow"]
X_fin = dff.query("dataset=='final'").drop(columns=["tow", "dataset"])

X = X.drop(columns=["flight_id"])
column_indices = [X.columns.get_loc(col) for col in ct_features]

model = lgbmreg(
    reg_alpha=0.17,
    reg_lambda=0.03,
    random_state=33,
    num_leaves=215,
    learning_rate=0.08,
    n_estimators=800,
    colsample_bytree=0.5,
    importance_type="gain",
)

model.fit(
    X,
    y,
    categorical_feature=ct_features,
)

preds = model.predict(
    X_fin.drop(columns=["flight_id"]), categorical_feature=ct_features
)
# %% merge with flight_ids
X_fin = X_fin.assign(tow=preds)
X_fin = X_fin[["flight_id", "tow"]].astype(int)

# %% reorder them for submission
df_fin = pd.read_csv("data/final_submission_set.csv")[["flight_id"]]
df_fin = df_fin.merge(X_fin)
df_fin = df_fin[["flight_id", "tow"]].astype(int)
df_fin.to_csv("submissions/final_v0.csv", index=False)
# %%
