# prc_atow_tbp
We are team_brave_pilow
Here is our model developed in the scope of EUROCONTROL's PRC Data Challenge.

## Preprocessing

In the preprocessing stage we used opensource tools, such as OpenAP, Traffic, AirportsData.
The requirements are given in `requirements.txt`

The preprocessing consists of several stages.
1. Resampling and deriving addintional trajectory features in `preprocessing.py`:

To run this file the "data" folder should contain all the trajectoy data. This script will create "processed" folder and store there processed flight .parquet files.

2. Extracting trajectory features in `trajectory_df.py`:

Here we extract the features from the trajectory data in "processed" folder and create a new dataframe `traj_df.parquet`. this script also create "traj_dfs" folder, where the flights are stored 365 files for each day in a year. This is made to keep the progress even in case of aborting running of script. This folder is not used after "traj_df.parquet" file is created.

3. Merging features together:

    1. `all_features.py` file merges trajectory features ("traj_df.parquet") with the other features provided in "challenge_set.csv" and "final_submission_set.csv"

    2. adding more climb-phase data:
    At some point we realized that climb-phase is quite important, and decided to extract more features from climb phase.
        (1) Preprocessing is similar, but now we didn't resample data, so we don't miss anything. The script is `preprocess_climb.py`
        (2) Extracting features `climb_features.py`
    
    3. `add_features.py` file add more features such as airport distance, 



### The final model structure:

![final_model](final_model.jpeg)
