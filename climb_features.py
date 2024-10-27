# %%
import pandas as pd
import numpy as np
import os
import warnings


# %%
# %%
# Add climb features
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )  # Convert degrees to radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # Distance in kilometers
    return distance


df = pd.read_csv("../processed_ch_sub_final.csv")
# Define the directory containing the .parquet files
directory = "climb/"

altitude_intervals = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)]

# Get all .parquet files in the directory and sort them
parquet_files = sorted(
    [file for file in os.listdir(directory) if file.endswith(".parquet")], reverse=False
)
# parquet_files = parquet_files[::-1]
# parquet_files = parquet_files[:]
# Total number of .parquet files
total_files = len(parquet_files)
counter = 0

print("Lezgooo")

# Create the columns in df_processed in advance for average climb rates and path angles
for altitude_range in ["0_5000", "5000_10000", "10000_15000", "15000_20000"]:
    df[f"avg_climb_rate_{altitude_range}"] = np.nan
    df[f"avg_path_angle_{altitude_range}"] = np.nan
    df[f"avg_tas_mps_{altitude_range}"] = np.nan
    df[f"spec_acc_mps_{altitude_range}"] = np.nan

# Iterate over sorted .parquet files
for idx, file_name in enumerate(parquet_files):
    print(f"{idx+1}/{len(parquet_files)}")
    # Load the .parquet file
    file_path = os.path.join(directory, file_name)
    df_traj = pd.read_parquet(file_path)
    df_traj = df_traj[df_traj["phase"] == "CLIMB"]

    len_before = len(df_traj)

    # Filter df_traj to include only rows where flight_id exists in df
    df_traj = df_traj[df_traj["flight_id"].isin(df["flight_id"])]
    len_after = len(df_traj)

    df_traj["track_rad"] = np.deg2rad(df_traj["track"])
    df_traj["tas"] = np.sqrt(
        (
            df_traj["groundspeed"] * np.cos(df_traj["track_rad"])
            - df_traj["u_component_of_wind"]
        )
        ** 2
        + (
            df_traj["groundspeed"] * np.sin(df_traj["track_rad"])
            - df_traj["v_component_of_wind"]
        )
        ** 2
    )

    # Drop the 'track_rad' column as it's no longer needed
    df_traj.drop(columns="track_rad", inplace=True)

    # Increment the counter
    counter += 1

    # Iterate through each flight_id in df_traj
    flight_ids = df_traj["flight_id"].unique()

    # Define altitude intervals
    altitude_intervals = [0, 5000, 10000, 15000, 20000]

    print("Processing features...")

    # Create an empty list to store climb data for each flight
    climb_data = []

    for idx, flight_id in enumerate(flight_ids, start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
            if idx % 100 == 0:
                print(f"{idx}/{len(flight_ids)}")

            # Filter the data for the current flight_id and sort by timestamp
            df_flight = (
                df_traj[df_traj["flight_id"] == flight_id]
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            # Convert TAS and vertical rate to m/s
            df_flight["tas_mps"] = df_flight["tas"] * 0.514444

            df_flight["smoothed_vertical_rate_mps"] = (
                df_flight["vertical_rate"] * 0.00508
            )

            # Calculate spec_path_angle
            df_flight["spec_path_angle"] = (
                df_flight["smoothed_vertical_rate_mps"] / df_flight["tas_mps"]
            )

            # Calculate the average climb rates for each altitude range
            avg_climb_rates = {}
            avg_spec_angles = {}
            avg_tas_vals = {}
            spec_acc_dict = {}

            for i in range(len(altitude_intervals) - 1):
                lower_bound = altitude_intervals[i]
                upper_bound = altitude_intervals[i + 1]

                # Filter data within the current altitude range
                df_interval = df_flight[
                    (df_flight["altitude"] > lower_bound)
                    & (df_flight["altitude"] <= upper_bound)
                ]

                if not df_interval.empty:
                    avg_climb_rate = round(
                        df_interval["smoothed_vertical_rate_mps"].mean(), 0
                    )
                    avg_path_angle = round(df_interval["spec_path_angle"].mean(), 5)
                    avg_tas_val = round(df_interval["tas_mps"].mean(), 0)

                    delta_v = (
                        df_interval["tas_mps"].iloc[-1] - df_interval["tas_mps"].iloc[0]
                    )
                    delta_t = (
                        df_interval["timestamp"].iloc[-1]
                        - df_interval["timestamp"].iloc[0]
                    ).total_seconds()
                    dh = (
                        df_interval["altitude"].iloc[-1]
                        - df_interval["altitude"].iloc[0]
                    )

                    v_at_alt = df_interval["tas_mps"].iloc[-1]
                    spec_acc = (
                        (v_at_alt * delta_v / delta_t + 9.81 * dh / delta_t)
                        if delta_t != 0
                        else np.nan
                    )
                else:
                    avg_climb_rate = np.nan
                    avg_path_angle = np.nan
                    avg_tas_val = np.nan
                    spec_acc = np.nan

                avg_climb_rates[f"avg_climb_rate_{lower_bound}_{upper_bound}"] = (
                    avg_climb_rate
                )
                avg_spec_angles[f"avg_path_angle_{lower_bound}_{upper_bound}"] = (
                    avg_path_angle
                )
                avg_tas_vals[f"avg_tas_mps_{lower_bound}_{upper_bound}"] = avg_tas_val
                spec_acc_dict[f"spec_acc_mps_{lower_bound}_{upper_bound}"] = spec_acc

            # Filling the last row with 0 as there's no next point to calculate distance for
            climb_range = haversine(
                df_flight["latitude"].values[-1],
                df_flight["longitude"].values[-1],
                df_flight["latitude"].values[0],
                df_flight["longitude"].values[0],
            )
            climb_dur = (
                df_flight["timestamp"].iloc[-1] - df_flight["timestamp"].iloc[0]
            ).total_seconds()

            # Update df with the calculated values
            for key, value in avg_climb_rates.items():
                df.loc[df["flight_id"] == flight_id, key] = value
            for key, value in avg_spec_angles.items():
                df.loc[df["flight_id"] == flight_id, key] = value
            for key, value in avg_tas_vals.items():
                df.loc[df["flight_id"] == flight_id, key] = value
            for key, value in spec_acc_dict.items():
                df.loc[df["flight_id"] == flight_id, key] = value

            df.loc[df["flight_id"] == flight_id, "climb_range"] = round(climb_range, 0)
            df.loc[df["flight_id"] == flight_id, "climb_rate"] = round(
                climb_range / climb_dur, 0
            )
df.to_csv("final_sub_dataframe_with_climb_feats.csv", index=False)
