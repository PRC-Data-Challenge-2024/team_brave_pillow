# %%
## This file derives all the necessary features from trajectory data
## the output dataframe has 1 row of features for 1 flight
import click
from pathlib import Path
import numpy as np
import openap
import pandas as pd
import glob
from openap import aero
from multiprocessing import Pool
from tqdm import tqdm
from traffic.core import Traffic, Flight
import os


# %%


def calc_duration(df_phase):
    if len(df_phase) == 0:
        return 0
    return (
        (df_phase.timestamp.values[-1] - df_phase.timestamp.values[0]) / 10**9
    ).astype(int)


def calc_cruise_duration(df_climb, df_cruise, df_descent):
    if len(df_climb) == 0 or len(df_descent) == 0:
        return calc_duration(df_cruise)
    return (
        (df_descent.timestamp.values[0] - df_climb.timestamp.values[-1]) / 10**9
    ).astype(int)


def traj_df_creation(f):
    overwrite = False

    if os.path.exists(f"traj_dfs/{f[10:-8]}-df.parquet") and not overwrite:
        print(f"traj_dfs/{f[10:-8]}-df.parquet already there")
        return

    results = []
    t = Traffic(pd.read_parquet(f))
    print(f)
    for flight in tqdm(t):

        df1 = flight.data.reset_index(drop=True).drop_duplicates(subset=["ts"])
        if df1.tas.isna().sum() == len(df1) or df1.altitude.isna().sum() == len(df1):
            continue

        df1 = df1.assign(
            vertical_acc=lambda x: x.vertical_rate.diff() / x.ts.diff() / 60
        )
        df1.loc[0, "vertical_acc"] = 0

        df_climb = df1.query("phase=='CLIMB'")
        df_cruise = df1.query("cruise==1")
        df_descent = df1.query("phase=='DESCENT'")

        if len(df_cruise) > 0:
            tas_percentiles = np.percentile(df_cruise.tas, [15, 85])
            altitude_percentiles = np.percentile(df_cruise.altitude, [15, 85])
            df_cruise = df_cruise.query(
                f"{tas_percentiles[0]} < tas < {tas_percentiles[1]}"
            )
            df_cruise = df_cruise.query(
                f"{altitude_percentiles[0]} < altitude < {altitude_percentiles[1]}"
            )

        if len(df_cruise) == 0:
            df_cruise = df1.query(
                f"altitude>={np.percentile(df1.altitude,80)}"
            ).reset_index(drop=True)
            start, end = int(len(df_cruise) * 0.2), int(len(df_cruise) * 0.8)
            df_cruise = df_cruise.iloc[start:end] if start != end else df_cruise

        df_climb_0 = df_climb
        if len(df_climb) > 0:
            df_climb_0 = df_climb.query("altitude<10000").reset_index(drop=True)
        results.append(
            {
                "flight_id": df1.flight_id.values[0],
                "mean_cruise_altitude": df_cruise.altitude.mean(),
                "max_cruise_altitude": df_cruise.altitude.max(),
                "mean_initclimb_tas": df_climb_0.tas.mean(),
                "max_initclimb_tas": df_climb_0.tas.max(),
                "mean_climb_tas": df_climb.tas.mean(),
                "max_climb_tas": df_climb.tas.max(),
                "mean_cruise_tas": df_cruise.tas.mean(),
                "max_cruise_tas": df_cruise.tas.max(),
                "mean_descent_tas": df_descent.tas.mean(),
                "max_descent_tas": df_descent.tas.max(),
                "distance": df1.airdist.max(),
                "mean_latitude": df_cruise.latitude.mean(),
                "mean_longitude": df_cruise.longitude.mean(),
                "mean_cruise_temperature": df_cruise.temperature.mean(),
                "mean_cruise_humidity": df_cruise.specific_humidity.mean(),
                "mean_initroc": df_climb_0.vertical_rate.mean(),
                "median_initroc": df_climb_0.vertical_rate.median(),
                "max_initroc": df_climb_0.vertical_rate.max(),
                "mean_roc": df_climb.vertical_rate.mean(),
                "median_roc": df_climb.vertical_rate.median(),
                "max_roc": df_climb.vertical_rate.max(),
                "mean_rod": df_descent.vertical_rate.mean(),
                "median_rod": df_descent.vertical_rate.median(),
                "min_rod": df_descent.vertical_rate.min(),
                "mean_pos_initclimb_vertacc": df_climb_0.query(
                    "vertical_acc>0"
                ).vertical_acc.mean(),
                "max_initclimb_vertacc": df_climb_0.vertical_acc.max(),
                "mean_pos_climb_vertacc": df_climb.query(
                    "vertical_acc>0"
                ).vertical_acc.mean(),
                "max_climb_vertacc": df_climb.vertical_acc.max(),
                "mean_neg_descent_vertacc": df_descent.query(
                    "vertical_acc<0"
                ).vertical_acc.mean(),
                "min_descent_vertacc": df_descent.vertical_acc.min(),
                "dur_initclimb": calc_duration(df_climb_0),
                "dur_climb": calc_duration(df_climb),
                "dur_cruise": calc_duration(df_cruise),
                "dur_descent": calc_duration(df_descent),
                "dur_cruise2": calc_cruise_duration(df_climb, df_cruise, df_descent),
                "mean_initclimb_thrust": df_climb_0.thrust.mean(),
                "mean_climb_thrust": df_climb.thrust.mean(),
                "mean_cruise_thrust": df_cruise.thrust.mean(),
                "mean_descent_thrust": df_descent.thrust.mean(),
                "mean_initclimb_fuelflow": df_climb_0.fuelflow.mean(),
                "mean_climb_fuelflow": df_climb.fuelflow.mean(),
                "mean_cruise_fuelflow": df_cruise.fuelflow.mean(),
                "mean_descent_fuelflow": df_descent.fuelflow.mean(),
                "mean_initclimb_mass_loss": df_climb_0.mass_loss.mean(),
                "mean_climb_mass_loss": df_climb.mass_loss.mean(),
                "mean_cruise_mass_loss": df_cruise.mass_loss.mean(),
                "mean_descent_mass_loss": df_descent.mass_loss.mean(),
                "fuel_spent": df1.mass_loss.max(),
            }
        )
    traj_df = pd.DataFrame.from_dict(results)
    traj_df["flight_id"] = traj_df["flight_id"].astype(int)
    traj_df.to_parquet(f"traj_dfs/{f[10:-8]}-df.parquet", index=False)


# %%
def main():
    files = glob.glob(f"processed/*.parquet")
    Path("traj_dfs/").mkdir(exist_ok=True)
    with Pool(10) as pool:
        pool.map(traj_df_creation, files)
    # for f in files:
    #     traj_df_creation(f)

    dfs = glob.glob("traj_dfs/*.parquet")
    df_all = pd.concat(pd.read_parquet(ff) for ff in dfs)
    df_all.to_parquet("traj_df.parquet", index=False)


# %%
if __name__ == "__main__":
    main()
