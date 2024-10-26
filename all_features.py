# %% merge trajectory features with other features
import pandas as pd

df_challenge = pd.read_csv("data/challenge_set.csv").assign(dataset="challenge")
df_final = pd.read_csv("data/final_submission_set.csv").assign(dataset="final")
df = pd.concat([df_challenge, df_final], axis=0)

# read the trajectory data
df_traj = pd.read_parquet("traj_df.parquet")
dff = pd.merge(
    df,
    df_traj,
    on="flight_id",
    how="outer",
)
dff = dff.dropna(subset=["dataset"])

ac_tows = pd.read_csv("ac_tows.csv")
dff = dff.merge(
    ac_tows[["ac", "min_tow_ch_set", "max_tow_ch_set", "m_tow"]],
    left_on="aircraft_type",
    right_on="ac",
    how="left",
)
dff = dff.rename(
    columns={"min_tow_ch_set": "min_tow_ch", "max_tow_ch_set": "max_tow_ch"}
)
dff.to_csv("processed_ch_sub_final.csv", index=False)

# %%
