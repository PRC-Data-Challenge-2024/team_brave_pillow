import pandas as pd
from sklearn.metrics import root_mean_squared_error

import os

def get_model_rmse_aircraft_type(df_interest, default_rmse_aircraft_type_sorted):
    model_rmse_aircraft_type = default_rmse_aircraft_type_sorted.copy()
    
    for ac in df_interest['aircraft_type'].unique():
        try:
            mask_ac = df_interest['aircraft_type'] == ac
            df_check_rmse = df_interest[mask_ac]

            rmse = root_mean_squared_error(df_check_rmse[df_check_rmse['test'] == 1]['tow_pred'], df_check_rmse[df_check_rmse['test'] == 1]['tow'])
            model_rmse_aircraft_type[ac] = rmse
        except:
            model_rmse_aircraft_type[ac] = 100000
    
    return model_rmse_aircraft_type

def get_model_rmse_all_aircraft_type(df_interest, default_rmse_aircraft_type_sorted):
    model_rmse_aircraft_type = default_rmse_aircraft_type_sorted.copy()
    
    for ac in df_interest['aircraft_type'].unique():
        try:
            mask_ac = df_interest['aircraft_type'] == ac
            df_check_rmse = df_interest[mask_ac]

            rmse = root_mean_squared_error(df_check_rmse['tow_pred'], df_check_rmse['tow'])
            model_rmse_aircraft_type[ac] = rmse
        except:
            model_rmse_aircraft_type[ac] = 100000
    
    return model_rmse_aircraft_type

def get_best_tow_pred(row):
    tow_col = f'tow_{int(row["label"])}'
    tow_col_next = f'tow_{int(row["label"] + 1)}'

    if tow_col in df_processed.columns:
        return row[tow_col] if pd.notna(row[tow_col]) else row.get(tow_col_next, None)
    else:
        return None  # or a default value if desired
    
directory = 'final_sub'
file_names_list = os.listdir(directory)
file_names = [os.path.join(directory, file_name) for file_name in file_names_list if '_with_tow_pred' in file_name]
file_names = sorted(file_names, key=lambda x: int(x.split('_')[-5]))

dfs = [pd.read_csv(file_name) for file_name in file_names]

df_processed = pd.read_csv('data/processed_df_all.csv').query('submission==0')
df_processed = df_processed.merge(
                    dfs[-1][['flight_id', 'test']],
                    on='flight_id', how='left'
                )

df_interest = dfs[-1]


default_rmse_aircraft_type = {}
for ac in df_interest['aircraft_type'].unique():
    default_rmse_aircraft_type[ac] = 100000
    
default_rmse_aircraft_type_sorted = dict(sorted(default_rmse_aircraft_type.items()))

dict_all_model_rmse = {}
dict_all_model_rmse_on_test = {}
counter = 0

for file_name, df in zip(file_names, dfs):
    file_name_cleaned = file_name.split('_with_tow_pred')[0]
    dict_all_model_rmse[counter] = get_model_rmse_all_aircraft_type(df, default_rmse_aircraft_type_sorted)
    dict_all_model_rmse_on_test[counter] = get_model_rmse_aircraft_type(df, default_rmse_aircraft_type_sorted)
    
    df_processed = df_processed.merge(
                    df[['flight_id', 'tow_pred']].rename(columns={'tow_pred': f'tow_{counter}'}),
                    on='flight_id', how='left'
                )
    counter += 1
    
df_rmse = pd.DataFrame(dict_all_model_rmse)
df_rmse['label'] = df_rmse.idxmin(axis=1)

aircraft_to_label_map = df_rmse['label'].to_dict()
aircraft_to_label_map

df_processed['label'] = df_processed['aircraft_type'].map(aircraft_to_label_map)
df_processed['best_tow_pred'] = df_processed.apply(get_best_tow_pred, axis=1)

rmse = root_mean_squared_error(df_processed[df_processed['test'] == 1]['tow'], df_processed[df_processed['test'] == 1]['best_tow_pred'])
rmse = int(rmse)

print("RMSE: ", rmse)

file_sub_names = [f"{file_name.split('_with_tow_pred')[0]}.csv" for file_name in file_names]

dfs_sub = [pd.read_csv(file_sub) for file_sub in file_sub_names]

df_processed_sub = pd.read_csv('data/processed_df_all.csv').query('submission==1')

counter = 0
for file_sub, df_sub in zip(file_sub_names, dfs_sub):
    df_processed_sub = df_processed_sub.merge(
                        df_sub[['flight_id', 'tow']].rename(columns={'tow': f'tow_{counter}'}),
                        on='flight_id', how='left'
                    )
    counter += 1
    
df_processed_sub['label'] = df_processed_sub['aircraft_type'].map(aircraft_to_label_map)
df_processed_sub['tow'] = df_processed_sub.apply(get_best_tow_pred, axis=1)

df_final = pd.read_csv('data/final_submission_set.csv')[['flight_id']]

df_final = df_final.merge(
                    df_processed_sub[['flight_id', 'tow']],
                    on='flight_id', how='left'
                )

rmse_classifier_output = f'{directory}rmse_classifier_{rmse}.csv'
df_final.to_csv(rmse_classifier_output, index = False)

# %%
df_rmse_on_test = pd.DataFrame(dict_all_model_rmse_on_test)
df_rmse_on_test['label'] = df_rmse_on_test.idxmin(axis=1)

aircraft_to_label_map_on_test = df_rmse_on_test['label'].to_dict()
aircraft_to_label_map_on_test

#overwrite the prev ones
df_processed['label'] = df_processed['aircraft_type'].map(aircraft_to_label_map_on_test)
df_processed['best_tow_pred'] = df_processed.apply(get_best_tow_pred, axis=1)

rmse_on_test = root_mean_squared_error(df_processed[df_processed['test'] == 1]['tow'], df_processed[df_processed['test'] == 1]['best_tow_pred'])
rmse_on_test = int(rmse_on_test)

print("RMSE: ", rmse_on_test)

file_sub_names = [f"{file_name.split('_with_tow_pred')[0]}.csv" for file_name in file_names]

dfs_sub = [pd.read_csv(file_sub) for file_sub in file_sub_names]

df_processed_sub = pd.read_csv('data/processed_df_all.csv').query('submission==1')

counter = 0
for file_sub, df_sub in zip(file_sub_names, dfs_sub):
    df_processed_sub = df_processed_sub.merge(
                        df_sub[['flight_id', 'tow']].rename(columns={'tow': f'tow_{counter}'}),
                        on='flight_id', how='left'
                    )
    counter += 1
    
df_processed_sub['label'] = df_processed_sub['aircraft_type'].map(aircraft_to_label_map_on_test)
df_processed_sub['tow'] = df_processed_sub.apply(get_best_tow_pred, axis=1)

df_final = pd.read_csv('data/final_submission_set.csv')[['flight_id']]

df_final = df_final.merge(
                    df_processed_sub[['flight_id', 'tow']],
                    on='flight_id', how='left'
                )

rmse_classifier_based_on_test_output = f'{directory}rmse_classifier_based_on_test_{rmse_on_test}.csv'
df_final.to_csv(rmse_classifier_based_on_test_output, index = False)

df_mix = df_final.copy()

df_mix['tow'] = pd.read_csv(rmse_classifier_output)['tow'] * 0.5 + pd.read_csv(rmse_classifier_based_on_test_output)['tow'] * 0.5

output_mix = f'{directory}/rmse_classifier_mixed.csv'

df_mix.to_csv(output_mix, index = False)

file_name = [rmse_classifier_output,
             rmse_classifier_based_on_test_output,
             output_mix
            ]

df_check = pd.read_csv('data/final_submission_set.csv')

for file in file_name:
    df_test = pd.read_csv(file)

    length = len(df_test['tow'])
    isna = df_test['tow'].isna().sum()

    if(length != len(df_check)):
        print("Length not good")
    elif isna != 0:
        print("There's nan")
    elif (len(df_test.columns) != 2):
        print("Contain other columns")
    else:
        print(f"{file} is okay to submit. Lezgo")
    