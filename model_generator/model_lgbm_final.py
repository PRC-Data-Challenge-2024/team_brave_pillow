# %%
## Checking the overall performance of the model

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import os

from sklearn.model_selection import train_test_split


import lightgbm as lgb
from lightgbm import LGBMRegressor as lgbreg

from typing import Dict

import numpy as np

import re

model_name = input("Model name(no_space): ")

#### -----------Functions here----------- ####
def classify_time(hour, nb_of_partition=24):
    range_of_hour = 24 / nb_of_partition

    return hour // range_of_hour

# Define aircraft families as a constant
AIRCRAFT_FAMILIES: Dict[str, str] = {
    r'B73[0-9]': 'B73x',
    r'B78[0-9]': 'B78x',
    r'B77[0-9]': 'B77x',
    r'A31[0-9]': 'A31x',
}

# Compile patterns once for better performance
COMPILED_PATTERNS = {re.compile(pattern): replacement 
                    for pattern, replacement in AIRCRAFT_FAMILIES.items()}

def combine_aircraft_type(df: pd.DataFrame, 
                        column_name: str = 'aircraft_type',
                        threshold: float = 0.01) -> pd.DataFrame:
    """
    Process aircraft types in a dataframe by combining similar types and handling rare ones.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Name of the column containing aircraft types
        threshold (float): Threshold below which aircraft types are considered rare
        
    Returns:
        pd.DataFrame: DataFrame with processed aircraft types in new column
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    def standardize_type(aircraft_type):
        """Standardize individual aircraft type."""
        if not isinstance(aircraft_type, str):
            return 'Unknown'
            
        aircraft_type = aircraft_type.strip().upper()
        if not aircraft_type:
            return 'Unknown'
            
        # Check against compiled patterns
        for pattern, replacement in COMPILED_PATTERNS.items():
            if pattern.match(aircraft_type):
                return replacement
                
        return aircraft_type
    
    # First pass: Standardize aircraft types
    df['aircraft_type'] = df[column_name].apply(standardize_type)
    
    # Calculate frequencies and handle rare types
    type_counts = df['aircraft_type'].value_counts(normalize=True)
    df['aircraft_type'] = df['aircraft_type'].apply(
        lambda x: 'Other' if type_counts.get(x, 0) < threshold else x
    )
    
    return df

def calculate_sample_weight(X):
    # Initialize all weights to 1
    sample_weight = np.ones(len(X))

    # Assign weight of 6 if aircraft_type is in [0, 6, 7, 15]
    # sample_weight[X['aircraft_type'].isin([0, 6, 7, 15])] = 6.0
    # sample_weight[X['aircraft_type'].isin([2, 4, 5])] = 10

    return sample_weight

## Start processing
dff = pd.read_csv("prc_atow_tbp/final_sub_linreg60.csv")
# dff = pd.read_csv("processed_ch_sub_final.csv")

dff = dff.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# print(df_challenge_processed.columns)
dff["adepdespair"] = dff["adep"] + dff["ades"]
dff["countrypair"] = dff["country_code_adep"] + dff["country_code_ades"]

#### -----------Features Processing----------- ###
print("#### -----------Features Processing----------- ###")
aircraft_type_counts = dff['aircraft_type'].value_counts(normalize=True)
dff = combine_aircraft_type(dff, threshold=0.01)

print(f"Len aircraft type: {len(aircraft_type_counts)} -> {len(dff['aircraft_type'].unique())}")

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

# Drop unnecessary column
dff = dff.drop(columns=["hour_offblock", "hour_arrival"])
dff = dff.drop(columns=["date"])
dff["adepdespair"] = pd.factorize(dff["adepdespair"])[0]
dff["airline"] = pd.factorize(dff["airline"])[0]
dff["aircraft_type"] = pd.factorize(dff["aircraft_type"])[0]
dff["countrypair"] = pd.factorize(dff["countrypair"])[0]
dff["adep"] = pd.factorize(dff["adep"])[0]
dff["ades"] = pd.factorize(dff["ades"])[0]
dff["country_code_adep"] = pd.factorize(dff["country_code_adep"])[0]
dff["country_code_ades"] = pd.factorize(dff["country_code_ades"])[0]

dff['spec_thrust1'] = dff['drag_term1'].copy() + dff['avg_path_angle_0_5000'].copy()
dff['spec_thrust2'] = dff['drag_term2'].copy() + dff['avg_path_angle_5000_10000'].copy()
dff['spec_thrust3'] = dff['drag_term3'].copy() + dff['avg_path_angle_10000_15000'].copy()
dff['spec_thrust4'] = dff['drag_term4'].copy() + dff['avg_path_angle_15000_20000'].copy()

dff = dff.drop(
    columns=[
        "wtc",
        "ac",
        "callsign",
        "name_adep",
        "name_ades",
        "actual_offblock_time",
        "arrival_time",
        "dataset"
    ]
)

nb_top = 80
top_features = list(pd.read_csv('feature_importance_linreg_2376.csv')['feature'].iloc[:nb_top])
# top_features = list(pd.read_csv('feature_importance_climbcruise_2489.csv')['feature'])
# top_features = list(pd.read_csv('feature_importance_super_base_3019.csv')['feature'])

for important_feature in ['tow', 'flight_id', 'submission',
                        'adep', 'ades', 'aircraft_type',
                        ]:
    if important_feature not in top_features:
        top_features.append(important_feature)

ct_features = [
    "adep",
    "ades",
    "aircraft_type",
    "airline",
    "adepdespair",
    "countrypair",
]

for important_feature in ct_features:
    if important_feature not in top_features:
        top_features.append(important_feature)

dff[ct_features] = dff[ct_features].astype("category")

dff = dff[top_features]

# dff = dff[dff['aircraft_type'].isin([0, 6, 7, 15])]

dff.to_csv('processed_df_all.csv', index = False)

object_columns = dff.select_dtypes(include=['object']).columns
print("Object: ", object_columns)

############### Train-Test check #################
# initialize data
X = dff.query("submission==0").drop(columns=["tow", "submission"])
y = dff.query("submission==0")["tow"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

column_indices = [X_train.columns.get_loc(col) for col in ct_features]

####
# best 26/10/2024
# model = lgbreg(
#     reg_alpha=2.5305,
#     reg_lambda=5.7005,
#     random_state=42,
#     num_leaves=419,
#     learning_rate=0.0616,
#     n_estimators=500,  # assuming you want to keep this
#     colsample_bytree=0.5857,
#     importance_type="gain",
#     bagging_fraction=0.4835,
#     feature_fraction=0.5857,
#     max_depth=19,
#     min_child_weight=0.4990,
#     min_data_in_leaf=36,
#     min_split_gain=0.7749,
# )

model = lgbreg(
    reg_alpha=0.1020,
    reg_lambda=2.59,
    random_state=42,
    num_leaves=499,
    learning_rate=0.0511,
    n_estimators=680,  
    importance_type="gain",
    bagging_fraction=0.2618,
    feature_fraction=0.5022,
    max_depth=45,
    min_child_weight=6.7588,
    min_data_in_leaf=36,
    min_split_gain=0.8109,
)

print("Start fitting. Vamos!")

best_pred = {}

sample_weight = calculate_sample_weight(X_train)

model.fit(
    X_train.drop(columns=["flight_id"]),
    y_train,
    categorical_feature=ct_features,
    sample_weight=sample_weight  # Add sample weight here
)
# make the prediction using the resulting model
preds = model.predict(
    X_test.drop(columns=["flight_id"]), categorical_feature=ct_features
)

X_test = X_test.assign(tow=y_test, tow_pred=preds)

train_preds = model.predict(
    X_train.drop(columns=["flight_id"]), categorical_feature=ct_features
)

rmse = int(root_mean_squared_error(y_test, preds))
rmse_train = int(root_mean_squared_error(train_preds, y_train))

print(f"RMSE Train: {rmse_train}, RMSE Test: {rmse}")

X_train['test'] = 0
X_test['test'] = 1

# Recombine the train and test sets (optional)
X_combined = pd.concat([X_train.assign(tow=y_train, tow_pred = train_preds), X_test], axis=0)

# Save the combined dataframe with 'tow_true' and 'tow_pred'
X_combined.to_csv(f'{model_name}_{rmse}_with_tow_pred.csv', index=False)

######### Training on full challenge dataset and predictiong submission #########

ct_features = [
    "adep",
    "ades",
    "aircraft_type",
    "airline",
    "adepdespair",
    "countrypair",
]
dff[ct_features] = dff[ct_features].astype("category")
# initialize data
X = dff.query("submission==0").drop(columns=["tow", "submission"])
y = dff.query("submission==0")["tow"]
X_sub = dff.query("submission==1").drop(columns=["tow", "submission"])
# initialize Pool

column_indices = [X.drop(columns=["flight_id"]).columns.get_loc(col) for col in ct_features]

# train the model
sample_weight = calculate_sample_weight(X)

model.fit(
    X.drop(columns=["flight_id"]),
    y,
    categorical_feature=ct_features,
    sample_weight=sample_weight  # Add sample weight here
)

# After training, get the predictions for X
tow_prediction = model.predict(X.drop(columns=["flight_id"]))

# Create a DataFrame to store the results (assuming you have flight_id in X)
# If flight_id is not part of X, make sure to get it from the original dataset
results_df = pd.DataFrame({
    'flight_id': X['flight_id'],  # Adjust if flight_id is not in X
    'tow_prediction': tow_prediction,
    'tow': y
})

results_df.to_csv(f'train_pred_{model_name}_{rmse}.csv', index=False)

# make the prediction using the resulting model
preds = model.predict(
    X_sub.drop(columns=["flight_id"]), categorical_feature=ct_features
)

X_sub = X_sub.assign(tow=preds)
X_sub = X_sub[["flight_id", "tow"]].astype(int)

# %%
df_sub = pd.read_csv("final_submission_set.csv").drop(columns=["tow"])
df_sub = df_sub.merge(X_sub)
df_sub = df_sub[["flight_id", "tow"]].astype(int)
df_sub.to_csv(f"{model_name}_{rmse}.csv", index=False)

# %%
print(len(X.columns), len(model.feature_importances_))

# After training and making predictions, add this:
dfeat = pd.DataFrame({
    "feature": X.drop(columns=["flight_id"]).columns,
    "importance": model.feature_importances_
})

# Sort and display the feature importances
dfeat = dfeat.sort_values(by="importance", ascending=False)
print(dfeat)

# Optionally, save the feature importance to a CSV file
save_feat_name = f"feature_importance_{model_name}_{rmse}.csv"
dfeat.to_csv(f"feature_importance_{model_name}_{rmse}.csv", index=False)
print(f"Feature importance saved to {save_feat_name}")
# %%
