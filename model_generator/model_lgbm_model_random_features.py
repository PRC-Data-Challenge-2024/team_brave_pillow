# %%

## why do this?
## it is because I notice that the LightGBM features interact with each other
## when I train with 79 features, we have the rank of each features
## for example, adep is on number 50. But then I train with 60 features,
## adep becomes number 10. Then I realized the features interact with each other
## also, the worse performing prediction for different features used are different
## so, to take advantage of this fact, we train our model using different features
## next, we estimate the TOW by either taking the median of the predictions
## or we simply do weighting on each of them

from typing import Dict

import pandas as pd
import numpy as np

import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import os

import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import lightgbm as lgb
from lightgbm import LGBMRegressor as lgbreg


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

def calculate_sample_weight(X):
    # Initialize all weights to 1
    sample_weight = np.ones(len(X))

    # Assign weight of 6 if aircraft_type is in [0, 6, 7, 15]
    sample_weight[X['aircraft_type'].isin([0, 7])] = 4.0
    sample_weight[X['aircraft_type'].isin([6, 15])] = 6.0

    # Assign weight of 3 if aircraft_type is in [2, 4, 5]
    sample_weight[X['aircraft_type'].isin([2, 4, 5])] = 2.0

    # Assign weight of 9 if airline is 3
    sample_weight[X['airline'] == 3] = 9.0

    # Assign weight of 9 if aircraft_type is 16
    sample_weight[X['aircraft_type'] == 16] = 9.0

    return sample_weight


dff = pd.read_csv("with_linreg_tow_60.csv")

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

dff['spec_thrust1'] = dff['drag_term1'] + dff['avg_path_angle_0_5000'] + dff['spec_acc_mps_0_5000']
dff['spec_thrust2'] = dff['drag_term2'] + dff['avg_path_angle_5000_10000'] + dff['spec_acc_mps_5000_10000']
dff['spec_thrust3'] = dff['drag_term3'] + dff['avg_path_angle_10000_15000'] + dff['spec_acc_mps_10000_15000']
dff['spec_thrust4'] = dff['drag_term4'] + dff['avg_path_angle_15000_20000'] + dff['spec_acc_mps_15000_20000']

features_super_base = ["tow",
                        "min_tow_ch", "max_tow_ch", "flown_distance",
                        "adepdespair", "m_tow", "aircraft_type", "ceiling_diff",
                        "flown_distance_per_def_cruise", "airline", "flight_duration",
                        "countrypair", "great_circle_distance_km", "cruise_diff",
                        "day_of_year", "adep", "time_of_day_offblock", "ades",
                        "submission", "flight_id"
                    ]

features_base_model = ['tow',
                       'flight_id', 'adep', 'ades',
                       'aircraft_type', 'airline',
                       'flight_duration', 'flown_distance',
                       'flown_distance_per_def_cruise', 'great_circle_distance_km',
                       'cruise_diff', 'ceiling_diff',
                       'min_tow_ch', 'max_tow_ch', 'm_tow',
                       'adepdespair', 'countrypair',
                       'time_of_day_offblock', 'day_of_year',
                       'mean_roc', 'spec_acc_mps_15000_20000',
                       'spec_acc_mps_10000_15000', 'max_cruise_altitude',
                       'avg_climb_rate_15000_20000',
                       'mean_cruise_altitude', 'climb_range_normalized',
                       'mach_diff', 
                       'submission']

climb_features = ['tow',
                       'flight_id', 'adep', 'ades',
                       'aircraft_type', 'airline',
                       'flight_duration', 'flown_distance',
                       'flown_distance_per_def_cruise', 'great_circle_distance_km',
                       'cruise_diff', 'ceiling_diff',
                       'min_tow_ch', 'max_tow_ch', 'm_tow',
                       'adepdespair', 'countrypair',
                       'time_of_day_offblock', 'day_of_year',
                       'mean_roc',
                       'spec_acc_mps_0_5000', 'spec_acc_mps_5000_10000', 'spec_acc_mps_10000_15000', 'spec_acc_mps_15000_20000', 
                       'avg_climb_rate_0_5000', 'avg_climb_rate_5000_10000', 'avg_climb_rate_10000_15000', 'avg_climb_rate_15000_20000',
                       'avg_path_angle_0_5000', 'avg_path_angle_5000_10000', 'avg_path_angle_10000_15000', 'avg_path_angle_15000_20000',
                       'mean_cruise_altitude', 'climb_range_normalized',
                       'mach_diff',
                       'spec_thrust1', 'spec_thrust2', 'spec_thrust3', 'spec_thrust4',
                       'max_initroc', 'max_roc',
                       'submission'
                       ]

feature_of_interest_list = [list(pd.read_csv('feature_importance_linreg_2376.csv')['feature'][:60]),
                            list(pd.read_csv('feature_importance_climb_2405.csv')['feature']),
                            list(pd.read_csv('feature_importance_imputed_2388.csv')['feature'].iloc[:50]),
                    ]

for top_features in feature_of_interest_list:
    for important_feature in ['tow', 'flight_id', 'submission',
                            'adep', 'ades', 'aircraft_type',
                            'linreg_tow']:
        if important_feature not in top_features:
            top_features.append(important_feature)

dff = dff.drop(
    columns=[
        "wtc",
        "ac",
        "callsign",
        "name_adep",
        "name_ades",
        "actual_offblock_time",
        "arrival_time",
        "dataset",
        # "tow_pred",
        # "tow_pred_linreg",
        # 'max_flown_dist',
        # 'min_flown_dist',
    ]
)

# dff = dff[top_features]

ct_features = [
    "adep",
    "ades",
    "aircraft_type",
    "airline",
    "adepdespair",
    "countrypair",
]
dff[ct_features] = dff[ct_features].astype("category")

############### Train-Test check #################
# initialize data
X = dff.query("submission==0").drop(columns=["tow", "submission"])
y = dff.query("submission==0")["tow"]

####
# model = lgbreg(
#     reg_alpha=0.406,
#     reg_lambda=0.143,
#     random_state=42,
#     num_leaves=216,
#     learning_rate=0.0612,
#     n_estimators=969,
#     colsample_bytree=0.5,
#     importance_type="gain",
#     # force_col_wise=True,
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

counter_model = 0

for feature_of_interest in feature_of_interest_list:
    NUM_ITERATIONS = 75

    original_ct_features = ct_features

    features_for_sampling = list(X.columns)

    for i in range(NUM_ITERATIONS):
        counter_model += 1
        print(f"Training: {i+1}/{NUM_ITERATIONS}. ON Y VA!")
        init_feature_len = len(feature_of_interest)
        NUM_FEATURES = random.randint(init_feature_len, init_feature_len + 25)

        if(NUM_FEATURES > 85):
            NUM_FEATURES = 0

        # Exclude 'tow' and 'flight_id' from the remaining features for random sampling
        remaining_features = list(set(features_for_sampling) - set(feature_of_interest))

        # Randomly sample features to reach the total number of 40 (subtract the length of base features)
        sampled_additional_features = random.sample(remaining_features, NUM_FEATURES - len(feature_of_interest))
        
        # Combine base features with the sampled ones
        sampled_features = feature_of_interest + sampled_additional_features
        sampled_features.remove('tow')
        sampled_features.remove('submission')

        ct_features = [feature for feature in original_ct_features if feature in sampled_features]
        
        # Split data
        X_ = X[sampled_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X_, y, test_size=0.2, random_state=42  # Different random state for each iteration
        )

        column_indices = [X_train.columns.get_loc(col) for col in ct_features]

        model.fit(
            X_train.drop(columns=["flight_id"]),
            y_train,
            categorical_feature=ct_features,
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

        best_pred[counter_model] = rmse

        sorted_best_pred = dict(sorted(best_pred.items(), key=lambda item: item[1]))

        print(sorted_best_pred)

        X_train['test'] = 0
        X_test['test'] = 1

        # Recombine the train and test sets (optional)
        X_combined = pd.concat([X_train.assign(tow=y_train, tow_pred = train_preds), X_test], axis=0)
        X_combined[['tow', 'tow_pred']] = X_combined[['tow', 'tow_pred']].round()

        # Save the combined dataframe with 'tow_true' and 'tow_pred'
        X_combined[['flight_id', 'aircraft_type', 'airline', 'test', 'tow', 'tow_pred']].to_csv(f'{model_name}/{rmse}_{model_name}_{counter_model}_{NUM_FEATURES}_with_tow_pred.csv', index=False)

        ######### Training on full challenge dataset and predictiong submission #########

        dff[ct_features] = dff[ct_features].astype("category")
        # initialize data
        X = dff.query("submission==0").drop(columns=["tow", "submission"])
        y = dff.query("submission==0")["tow"]
        X_sub = dff.query("submission==1").drop(columns=["tow", "submission"])
        # initialize Pool

        column_indices = [X.drop(columns=["flight_id"]).columns.get_loc(col) for col in ct_features]

        X_ = X[sampled_features]
        sample_weight = calculate_sample_weight(X)

        try:
            # train the model
            model.fit(
                X_.drop(columns=["flight_id"]),
                y,
                categorical_feature=ct_features,
                sample_weight=sample_weight  # Add sample weight here
            )

            # Save using pickle
            with open(f'{model_name}/{rmse}_{model_name}_{counter_model}.pkl', 'wb') as f:
                pickle.dump(model, f)

            # After training, get the predictions for X
            tow_prediction = model.predict(X_.drop(columns=["flight_id"]))

            # make the prediction using the resulting model
            X_sub_ = X_sub[sampled_features]
            preds = model.predict(
                X_sub_.drop(columns=["flight_id"]), categorical_feature=ct_features
            )

            X_sub = X_sub.assign(tow=preds)
            X_sub = X_sub[["flight_id", "tow"]].astype(int)


            # df_sub = pd.read_csv("submission_set.csv").drop(columns=["tow"])
            # df_sub = df_sub.merge(X_sub)
            # df_sub = df_sub[["flight_id", "tow"]].astype(int)
            X_sub.to_csv(f"{model_name}/{rmse}_{model_name}_{counter_model}.csv", index=False)

            print(len(X.columns), len(model.feature_importances_))

            # After training and making predictions, add this:
            dfeat = pd.DataFrame({
                "feature": X_.drop(columns=["flight_id"]).columns,
                "importance": model.feature_importances_
            })

            # Sort and display the feature importances
            dfeat = dfeat.sort_values(by="importance", ascending=False)
            dfeat[['importance']] = dfeat[['importance']].round()
            print(dfeat)

            # Optionally, save the feature importance to a CSV file
            save_feat_name = f"feature_importance_{model_name}_{rmse}.csv"  
            dfeat.to_csv(f"{model_name}/feature_importance_{rmse}_{model_name}_{counter_model}.csv", index=False)
            print(f"Feature importance saved to {save_feat_name}")
            
        except:
            print(f"Error, skipping this {counter_model}")

