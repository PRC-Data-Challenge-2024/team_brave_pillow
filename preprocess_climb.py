# %%
import os
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from traffic.core import Traffic, Flight
from openap import aero
import numpy as np
from tqdm.contrib.concurrent import process_map  # or thread_map
import openap
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import gc

import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import warnings

warnings.filterwarnings("ignore")


# %%
def time_dups(f: Flight) -> Flight:
    if len(f.data) < 10 or np.percentile(f.data.altitude, 80) < 10_000:
        return None
    f = f.resample("1s")
    f = f.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    ts_diff = f.data.timestamp.diff().astype(int) / 10**9
    ts_diff[0] = 0
    f = f.assign(ts=ts_diff.cumsum()).phases().assign(cruise=0)
    df_flight = f.data
    if len(df_flight) >= 300:
        df_flight["altitude"] = savgol_filter(
            df_flight["altitude"], window_length=300, polyorder=2
        )
        df_flight["vertical_rate"] = savgol_filter(
            df_flight["vertical_rate"], window_length=100, polyorder=2
        )
    elif len(df_flight) >= 100:
        df_flight["altitude"] = savgol_filter(
            df_flight["altitude"], window_length=100, polyorder=2
        )
        df_flight["vertical_rate"] = savgol_filter(
            df_flight["vertical_rate"], window_length=100, polyorder=2
        )
    else:
        df_flight["altitude"] = df_flight["altitude"]
        df_flight["vertical_rate"] = df_flight["vertical_rate"]
    del ts_diff, f
    df_flight = df_flight.query("phase=='CLIMB'")
    if len(df_flight) < 2:
        return
    return Flight(df_flight)


def assign_thff(flight: Flight) -> Flight:
    df1 = flight.data.reset_index(drop=True).drop_duplicates(subset=["ts"])
    del flight
    ac = df1.aircraft_type[0].lower()
    ac_nosyn = [
        "a343",
        "b752",
        "b763",
        "b772",
        "b773",
        "b77w",
        "b788",
        "b789",
        "bcs1",
        "bcs3",
        "c56x",
        "crj9",
    ]
    if ac == "a20n":
        ac = "a320"
    if ac == "a359":
        ac = "a333"
    # df1 = label(df1)
    df1 = df1.assign(label="climb", thrust=np.nan, fuelflow=np.nan, mass_loss=np.nan)
    if ac in ac_nosyn:
        del ac, ac_nosyn
        return Flight(df1)

    thrust = openap.Thrust(ac=ac, use_synonym=True)
    fuel = openap.FuelFlow(ac=ac, use_synonym=True)

    df1["thrust"] = df1.apply(
        lambda row: getattr(thrust, row.label)(
            row.tas, row.altitude, row.vertical_rate
        ),
        axis=1,
    )
    df1 = df1.assign(fuelflow=fuel.at_thrust(acthr=df1.thrust, alt=df1.altitude))
    df1 = df1.assign(mass_loss=lambda x: (x.ts.diff() * x.fuelflow).fillna(0).cumsum())
    del ac_nosyn, thrust, fuel
    return Flight(df1)


def compute_airdist(flight):
    vg = flight.data.groundspeed * aero.kts
    dt = flight.data.timestamp.diff().mean().total_seconds()
    vgx = vg * np.sin(np.radians(flight.data.track))
    vgy = vg * np.cos(np.radians(flight.data.track))
    vax = vgx - flight.data.u_component_of_wind
    vay = vgy - flight.data.v_component_of_wind
    va = np.sqrt((vax**2 + vay**2))
    tas = va / aero.kts
    airdist = tas * dt * aero.kts
    flight = flight.assign(
        tas=tas,
        airdist=airdist.cumsum() / 1000,
    )
    del vg, dt, vgx, vgy, vax, vay, va, tas, airdist
    flight = flight.assign(distance=flight.airdist_max)

    return flight


# %%
def func(file):
    overwrite = False

    if os.path.exists(f"climb/" + file[19:]) and not overwrite:
        print(f"{file[19:]} already there")
        return

    df_ch = pd.read_csv("ch_sub_fin_.csv")

    print(f"file: {file[19:]} \t {datetime.now()}\n")
    df1 = pd.read_parquet(file)
    df1 = df1.merge(df_ch, on="flight_id", how="inner")
    if len(df1) <= 1:
        return
    t = Traffic(df1)
    t = (
        # t.filter()
        t.pipe(time_dups)
        # .resample(500)
        .pipe(compute_airdist)
        # .pipe(extract_cruise)
        .pipe(assign_thff).eval(10, desc=f"{file[19:]}")
    )
    df1 = t.data.dropna(subset=["flight_id"])
    df1["flight_id"] = df1["flight_id"].astype(int)
    df1.to_parquet(f"climb/" + file[19:], index=False)
    print(f"file: {file[19:]} is done \t {datetime.now()}\n")
    del df1, t, df_ch
    gc.collect()


# %%
def main():
    files = glob.glob("data/*.parquet")
    Path("climb/").mkdir(exist_ok=True)

    for file in tqdm(files):
        func(file)


# %%
if __name__ == "__main__":
    main()
