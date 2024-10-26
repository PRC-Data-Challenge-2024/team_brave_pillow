# %%
## This file resamples the trajectory data and
## derives some additional features, like phase, airdistance, airspeed, fueflow and thrust

import os
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from traffic.core import Traffic, Flight
from openap import aero
import numpy as np
import openap
import gc
from datetime import datetime


import warnings

warnings.filterwarnings("ignore")


# %%
def time_dups(f: Flight) -> Flight:
    if len(f.data) < 10 or np.percentile(f.data.altitude, 80) < 10_000:
        return None
    dur = f.duration.to_numpy().astype(int) / 10**9
    res_ = max(1, int(dur / 600))
    f = f.resample(f"{res_}s")
    f = f.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    ts_diff = f.data.timestamp.diff().astype(int) / 10**9
    ts_diff[0] = 0
    f = f.assign(ts=ts_diff.cumsum())
    del ts_diff
    return f


def label(df):
    fp = openap.FlightPhase()
    fp.set_trajectory(df.ts, df.altitude, df.tas, df.vertical_rate)
    label = fp.phaselabel()
    dc = {
        "CL": "climb",
        "CR": "cruise",
        "DE": "descent_idle",
        "NA": "descent_idle",
        "LVL": "descent_idle",
        "GND": "descent_idle",
    }
    phase = [dc.get(item, item) for item in label]
    df = df.assign(label=phase)
    return df


# more accurate cruise phase extraction
def extract_cruise(flight: Flight) -> Flight:
    flight = flight.phases().assign(cruise=0)
    df = flight.data
    del flight

    cli_ix = df.query("phase=='CLIMB'").index
    if len(cli_ix) == 0:
        id1 = df.index[0]
    else:
        id1 = cli_ix[-1]

    des_ix = df.query("phase=='DESCENT'").index
    if len(des_ix) == 0:
        id2 = df.index[-1]
    else:
        id2 = des_ix[0]

    df1 = df.loc[id1:id2, :].query(
        "(phase=='CRUISE' or phase=='LEVEL') "
        "and 45_000>altitude>20_000 and -500<vertical_rate<500 "
        "and tas==tas and altitude==altitude"
    )
    df.loc[df1.index, "cruise"] = 1
    del df1, cli_ix, des_ix

    return Flight(df)


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
    df1 = label(df1)
    df1 = df1.assign(thrust=np.nan, fuelflow=np.nan, mass_loss=np.nan)
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

    if os.path.exists(f"processed/" + file[5:]) and not overwrite:
        print(f"{file[5:]} already there")
        return

    df_ac = pd.concat(
        [
            pd.read_csv("data/challenge_set.csv"),
            pd.read_csv("data/final_submission_set.csv"),
        ]
    )[["flight_id", "aircraft_type"]]

    print(f"file: {file[5:]} \t {datetime.now()}\n")
    df1 = pd.read_parquet(file)
    df1 = df1.merge(df_ac, on="flight_id", how="inner")
    t = Traffic(df1)
    t = (
        t.filter()
        .pipe(time_dups)
        .resample(500)
        .pipe(compute_airdist)
        .pipe(extract_cruise)
        .pipe(assign_thff)
        .eval(10, desc=f"{file[5:]}")
    )
    df1 = t.data.dropna(subset=["flight_id"])
    df1["flight_id"] = df1["flight_id"].astype(int)
    df1.to_parquet(f"processed/" + file[5:], index=False)
    print(f"file: {file[5:]} is done \t {datetime.now()}\n")
    del df1, t, df_ac
    gc.collect()


# %%
def main():
    files = glob.glob("data/*.parquet")
    Path("processed/").mkdir(exist_ok=True)

    for file in files:
        func(file)


# %%
if __name__ == "__main__":
    main()
