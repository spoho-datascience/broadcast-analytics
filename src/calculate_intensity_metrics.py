"""
calculate_intensity_metrics.py

This script processes the position data to calculate intensity metrics from Experiment 1.
The final output is a CSV file containing player-wise intensity statistics, ready for further analysis.

---
Information for the User:

Due to licensing restrictions the raw XML files (`Positions/`, `Infos/`) that are required by the
`read_position_data_xml()` function used in this script are not provided with this paper.

This script is provided only to show you how the results for Experiment 1 — Intensity
were internally calculated for one match.

The final processed CSV file (`intensity_metrics.csv`) containing the results for all matches is part of the
supplemental material.
"""

import copy
import jdata as jd
import numpy as np
import pandas as pd

from floodlight.io.dfl import read_position_data_xml
from floodlight.models.kinematics import DistanceModel, VelocityModel
from floodlight.transforms.filter import butterworth_lowpass

from src.utils import distance_covered_per_zone

# === Settings ===
match_id = "DFL-MAT-0002UK"
source = "SF"
base_path = "<PATH_TO_FILES>"

# Mapping roles
roles = {
    "TW": "GK",
    "LV": "DEF", "IVL": "DEF", "IVZ": "DEF", "IVR": "DEF", "RV": "DEF",
    "DML": "MID", "DMZ": "MID", "DMR": "MID", "LM": "MID", "HL": "MID", "MZ": "MID", "HR": "MID", "RM": "MID",
    "OLM": "OFF", "ZO": "OFF", "ORM": "OFF", "HST": "OFF", "LA": "OFF", "STL": "OFF", "STZ": "OFF", "STR": "OFF", "RA": "OFF"
}

# Load data
positions, possession, ballstatus, teamsheet, pitch = read_position_data_xml(
    f"{base_path}Positions/{match_id}.xml",
    f"{base_path}Infos/{match_id}.xml"
)

visible = jd.load(f"./data/player_visibility/{source}_{match_id}_visible_with_ballstatus.json")

# Cut to first 45 minutes (25 fps)
for half in visible:
    for team in visible[half]:
        visible[half][team] = visible[half][team][:45 * 60 * 25]

# Set pitch dimensions
pitch.xlim, pitch.ylim = ((0, 105), (0, 68))

# Process position and ball status
for half in positions:
    ballstatus[half] = ballstatus[half].slice(0, 45 * 60 * 25)
    for team in positions[half]:
        positions[half][team] = butterworth_lowpass(
            positions[half][team].slice(0, 45 * 60 * 25)
        )
        positions[half][team].translate((52.5, 34))  # center pitch

# Apply visibility mask
positions_visible = copy.deepcopy(positions)
if source in ["SF", "TV"]:
    for half in positions_visible:
        for team in positions_visible[half]:
            positions_visible[half][team].x = np.where(
                visible[half][team] == 0, np.nan, positions_visible[half][team].x
            )
            positions_visible[half][team].y = np.where(
                visible[half][team] == 0, np.nan, positions_visible[half][team].y
            )

# Calculate visibility statistics
visible_home = np.sum([np.sum(visible["firstHalf"]["Home"], axis=0),
                       np.sum(visible["secondHalf"]["Home"], axis=0)], axis=0)
visible_away = np.sum([np.sum(visible["firstHalf"]["Away"], axis=0),
                       np.sum(visible["secondHalf"]["Away"], axis=0)], axis=0)

visible_home_percent = visible_home / (len(positions["firstHalf"]["Home"]) + len(positions["secondHalf"]["Home"]))
visible_away_percent = visible_away / (len(positions["firstHalf"]["Away"]) + len(positions["secondHalf"]["Away"]))

# Match active ratio (ball in play)
match_active = (np.sum(ballstatus["firstHalf"].code) + np.sum(ballstatus["secondHalf"].code)) / \
               (len(ballstatus["firstHalf"]) + len(ballstatus["secondHalf"]))

# Active and inactive visibility
active_visible_home = np.nansum([
    np.nansum(visible["firstHalf"]["Home"][ballstatus["firstHalf"].code.astype(bool)], axis=0),
    np.nansum(visible["secondHalf"]["Home"][ballstatus["secondHalf"].code.astype(bool)], axis=0)
], axis=0) / (np.sum(~np.isnan(visible["firstHalf"]["Home"]), axis=0) + np.sum(~np.isnan(visible["secondHalf"]["Home"]), axis=0))

active_visible_away = np.nansum([
    np.nansum(visible["firstHalf"]["Away"][ballstatus["firstHalf"].code.astype(bool)], axis=0),
    np.nansum(visible["secondHalf"]["Away"][ballstatus["secondHalf"].code.astype(bool)], axis=0)
], axis=0) / (np.sum(~np.isnan(visible["firstHalf"]["Away"]), axis=0) + np.sum(~np.isnan(visible["secondHalf"]["Away"]), axis=0))

inactive_visible_home = np.nansum([
    np.nansum(visible["firstHalf"]["Home"][~ballstatus["firstHalf"].code.astype(bool)], axis=0),
    np.nansum(visible["secondHalf"]["Home"][~ballstatus["secondHalf"].code.astype(bool)], axis=0)
], axis=0) / (np.sum(~np.isnan(visible["firstHalf"]["Home"]), axis=0) + np.sum(~np.isnan(visible["secondHalf"]["Home"]), axis=0))

inactive_visible_away = np.nansum([
    np.nansum(visible["firstHalf"]["Away"][~ballstatus["firstHalf"].code.astype(bool)], axis=0),
    np.nansum(visible["secondHalf"]["Away"][~ballstatus["secondHalf"].code.astype(bool)], axis=0)
], axis=0) / (np.sum(~np.isnan(visible["firstHalf"]["Away"]), axis=0) + np.sum(~np.isnan(visible["secondHalf"]["Away"]), axis=0))

# Distance and Velocity
distance, distance_visible = {}, {}
velocity, velocity_visible = {}, {}

for half in positions:
    distance[half], distance_visible[half] = {}, {}
    velocity[half], velocity_visible[half] = {}, {}

    for team in positions[half]:
        dm = DistanceModel()
        dm.fit(positions[half][team])
        distance[half][team] = dm.distance_covered()

        dm.fit(positions_visible[half][team])
        distance_visible[half][team] = dm.distance_covered()

        vm = VelocityModel()
        vm.fit(positions[half][team])
        velocity[half][team] = vm.velocity()

        vm.fit(positions_visible[half][team])
        velocity_visible[half][team] = vm.velocity()

# Total distance calculations
dist_home = np.nansum([np.nansum(distance["firstHalf"]["Home"], axis=0),
                       np.nansum(distance["secondHalf"]["Home"], axis=0)], axis=0)
dist_away = np.nansum([np.nansum(distance["firstHalf"]["Away"], axis=0),
                       np.nansum(distance["secondHalf"]["Away"], axis=0)], axis=0)
dist_home_visible = np.nansum([np.nansum(distance_visible["firstHalf"]["Home"], axis=0),
                               np.nansum(distance_visible["secondHalf"]["Home"], axis=0)], axis=0)
dist_away_visible = np.nansum([np.nansum(distance_visible["firstHalf"]["Away"], axis=0),
                               np.nansum(distance_visible["secondHalf"]["Away"], axis=0)], axis=0)

# High-speed distance (>6.9 m/s)
high_speed_home = np.sum([
    np.array(distance_covered_per_zone(distance["firstHalf"]["Home"], velocity["firstHalf"]["Home"], [(6.9, np.inf)])["6.9 to inf"]),
    np.array(distance_covered_per_zone(distance["secondHalf"]["Home"], velocity["secondHalf"]["Home"], [(6.9, np.inf)])["6.9 to inf"])
], axis=0)

high_speed_away = np.sum([
    np.array(distance_covered_per_zone(distance["firstHalf"]["Away"], velocity["firstHalf"]["Away"], [(6.9, np.inf)])["6.9 to inf"]),
    np.array(distance_covered_per_zone(distance["secondHalf"]["Away"], velocity["secondHalf"]["Away"], [(6.9, np.inf)])["6.9 to inf"])
], axis=0)

high_speed_home_visible = np.sum([
    np.array(distance_covered_per_zone(distance_visible["firstHalf"]["Home"], velocity_visible["firstHalf"]["Home"], [(6.9, np.inf)])["6.9 to inf"]),
    np.array(distance_covered_per_zone(distance_visible["secondHalf"]["Home"], velocity_visible["secondHalf"]["Home"], [(6.9, np.inf)])["6.9 to inf"])
], axis=0)

high_speed_away_visible = np.sum([
    np.array(distance_covered_per_zone(distance_visible["firstHalf"]["Away"], velocity_visible["firstHalf"]["Away"], [(6.9, np.inf)])["6.9 to inf"]),
    np.array(distance_covered_per_zone(distance_visible["secondHalf"]["Away"], velocity_visible["secondHalf"]["Away"], [(6.9, np.inf)])["6.9 to inf"])
], axis=0)

# Percentages
dist_home_percent = dist_home_visible / dist_home
dist_away_percent = dist_away_visible / dist_away
high_speed_home_percent = high_speed_home_visible / high_speed_home
high_speed_away_percent = high_speed_away_visible / high_speed_away

# Merge into Teamsheets
teamsheet["Home"].teamsheet["role"] = teamsheet["Home"].teamsheet["position"].map(roles)
teamsheet["Away"].teamsheet["role"] = teamsheet["Away"].teamsheet["position"].map(roles)

for stat_name, data_home, data_away in [
    ("visible", visible_home_percent, visible_away_percent),
    ("distance", dist_home, dist_away),
    ("distance_visible", dist_home_visible, dist_away_visible),
    ("distance_percent", dist_home_percent, dist_away_percent),
    ("high_speed", high_speed_home, high_speed_away),
    ("high_speed_visible", high_speed_home_visible, high_speed_away_visible),
    ("high_speed_percent", high_speed_home_percent, high_speed_away_percent),
    ("active_visible", active_visible_home, active_visible_away),
    ("inactive_visible", inactive_visible_home, inactive_visible_away),
]:
    teamsheet["Home"].teamsheet[stat_name] = teamsheet["Home"].teamsheet["xID"].map(lambda x: data_home[x])
    teamsheet["Away"].teamsheet[stat_name] = teamsheet["Away"].teamsheet["xID"].map(lambda x: data_away[x])

# Match active for all players
teamsheet["Home"].teamsheet["match_active"] = match_active
teamsheet["Away"].teamsheet["match_active"] = match_active

# Final merge
results = pd.concat([teamsheet["Home"].teamsheet, teamsheet["Away"].teamsheet])
results["source"] = source
results["match"] = match_id
results = results.drop(["player", "team"], axis=1)
results = results.dropna(subset=["visible"])

# Save
results.to_csv(f"{base_path}intensity_metrics.csv", index=False)
