# %%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import lightgbm as lgb
from lightgbm import LGBMRegressor as lgbreg

from typing import Dict

from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pickle

import re

model_name = input("Model name(no_space): ")

# Define the directory path
dir_path = f"./{model_name}"

# Check if the directory exists
if not os.path.exists(dir_path):
    # Create the directory
    os.makedirs(dir_path)
    print(f"Directory '{model_name}' created successfully.")
else:
    print(f"Directory '{model_name}' already exists.")

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

def calculate_sample_weight(X, aircraft_interest):
    # Initialize all weights to 1
    sample_weight = np.ones(len(X))

    for ac in X['aircraft_type'].unique():
        ac_pop = len(X[X['aircraft_type'] == ac])/len(X)
        sample_weight[X['aircraft_type'] == ac] = 1/ac_pop

    sample_weight[X['aircraft_type'].isin([aircraft_interest])] = max(sample_weight) * 5

    sample_weight[X['airline'].isin([3])] = max(sample_weight) * 1.5

    return sample_weight

## Start processing
dff = pd.read_csv("final_sub_linreg60.csv")

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


# %%
# dff_processed.to_csv('processed_df_final.csv', index = False)

object_columns = dff.select_dtypes(include=['object']).columns
print("Object: ", object_columns)

############### Train-Test check #################
# initialize data
#final
####
model = lgbreg(
    reg_alpha=0.1020,
    reg_lambda=2.59,
    random_state=42,
    num_leaves=419,
    learning_rate=0.0511,
    n_estimators= 680,
    importance_type="gain",
    bagging_fraction=0.2618,
    feature_fraction=0.5022,
    max_depth=45,
    min_child_weight=6.7588,
    min_data_in_leaf=36,
    min_split_gain=0.8109,
)

# %%
print("Start fitting. Vamos!")

best_pred = {}

nb_top = 65
top_features = list(pd.read_csv('feature_importance.csv')['feature'].iloc[:nb_top])

for important_feature in ['tow', 'flight_id', 'submission',
                        'adep', 'ades', 'aircraft_type',
                        ]:
    if important_feature not in top_features:
        top_features.append(important_feature)

# top_features.remove('climb_range_normalized')
top_features.remove('linreg_tow')

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

for ac in dff['aircraft_type'].unique():
    dff_processed = dff[top_features]

    interest_count = len(dff_processed[dff_processed['aircraft_type'] == ac])
    max_class_count = int(0.85 * interest_count)

    # Perform undersampling
    dff_processed = (
        dff_processed.groupby('aircraft_type')
        .apply(lambda x: x if x['aircraft_type'].iloc[0] == ac else x.sample(min(len(x), max_class_count), random_state=42))
        .reset_index(drop=True)
    )

    X = dff_processed.query("submission==0").drop(columns=["tow", "submission"])
    y = dff_processed.query("submission==0")["tow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    column_indices = [X_train.columns.get_loc(col) for col in ct_features]

    sample_weight = calculate_sample_weight(X_train, ac)

    model.fit(
        X_train.drop(columns=["flight_id"]),
        y_train,
        categorical_feature=ct_features,
        sample_weight=sample_weight  # Add sample weight here
    )

    with open(f'{model_name}/{ac}_{model_name}.pkl', 'wb') as f:
                    pickle.dump(model, f)

    expected_features = model.booster_.feature_name()
    preds = model.predict(
        X_test[expected_features], categorical_feature=ct_features
    )

    # make the prediction using the resulting model
    X_test = X_test.assign(tow=y_test, tow_pred=preds)

    train_preds = model.predict(
        X_train[expected_features], categorical_feature=ct_features
    )

    rmse = int(root_mean_squared_error(y_test, preds))
    rmse_train = int(root_mean_squared_error(train_preds, y_train))

    rmse_ac = int(root_mean_squared_error(y_test[X_test['aircraft_type'] == ac], preds[X_test['aircraft_type'] == ac]))
    print(f"RMSE Train: {rmse_train}, RMSE Test: {rmse}, RMSE Test AC: {rmse_ac}")

    X_train['test'] = 0
    X_test['test'] = 1

    # Recombine the train and test sets (optional)
    X_combined = pd.concat([X_train.assign(tow=y_train, tow_pred = train_preds), X_test], axis=0)
    X_combined = X_combined.drop_duplicates(subset='flight_id').reset_index(drop=True)

    # Save the combined dataframe with 'tow_true' and 'tow_pred'
    X_combined.to_csv(f'{model_name}/{model_name}_{ac}_{rmse}_with_tow_pred.csv', index=False)

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
    X = dff_processed.query("submission==0").drop(columns=["tow", "submission"])
    y = dff_processed.query("submission==0")["tow"]
    X_sub = dff.query("submission==1").drop(columns=["tow", "submission"])
    # initialize Pool

    column_indices = [X.drop(columns=["flight_id"]).columns.get_loc(col) for col in ct_features]

    # train the model
    sample_weight = calculate_sample_weight(X, ac)

    model.fit(
        X.drop(columns=["flight_id"]),
        y,
        categorical_feature=ct_features,
        sample_weight=sample_weight  # Add sample weight here
    )

    with open(f'{model_name}/{ac}_{model_name}_trained_all.pkl', 'wb') as f:
                    pickle.dump(model, f)

    # After training, get the predictions for X
    tow_prediction = model.predict(X[expected_features], categorical_feature=ct_features)

    rmse_train_all = int(root_mean_squared_error(y, tow_prediction))
    rmse_ac_all = int(root_mean_squared_error(y[X['aircraft_type'] == ac], tow_prediction[X['aircraft_type'] == ac]))

    print(f"RMSE Train All: {rmse_train}, RMSE Train All AC: {rmse_ac_all}")

    # make the prediction using the resulting model
    preds = model.predict(
        X_sub[expected_features], categorical_feature=ct_features
    )

    X_sub = X_sub.assign(tow=preds)
    X_sub = X_sub[["flight_id", "tow"]].astype(int)

    df_sub = pd.read_csv("data/final_submission_set.csv").drop(columns=["tow"])
    df_sub = df_sub.merge(X_sub)
    df_sub = df_sub[["flight_id", "tow"]].astype(int)
    df_sub.to_csv(f"{model_name}/{model_name}_{ac}_{rmse}.csv", index=False)

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
    save_feat_name = f"{model_name}/{model_name}_feature_importance_{ac}_{rmse}.csv"
    dfeat.to_csv(f"{save_feat_name}", index=False)
    print(f"Feature importance saved to {save_feat_name}")

# %%
