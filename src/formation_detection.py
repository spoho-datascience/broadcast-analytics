"""
formation_detection.py

This script detects the formations based on template matching for Experiment 2.
The output is a CSV file containing the predicted formations for each labeled possession phases.

---
Information for the User:

Due to licensing restrictions the raw XML files (`Positions/`, `Infos/`) that are required by the
`read_position_data_xml()` function used in this script are not provided with this paper.

This script is provided only to show you how the results for Experiment 2 — Formation Detection
were calculated internally.

The final processed CSV files (`formation_detection.csv`) containing the results for all matches is part of the
supplemental material.
"""

import json
import jdata as jd
import numpy as np
import pandas as pd

from floodlight.io.dfl import read_position_data_xml
from floodlight.models.geometry import CentroidModel

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# === Helper Functions ===

def role_assignment(slice_xy, avg_positions):
    """Role assignment algorithm from Bialkowski et al."""
    solved_positions = np.full((len(slice_xy), 10, 2), np.nan)

    nan_cols = np.argwhere(np.isnan(slice_xy).all(axis=0)).reshape(-1)
    slice_nonan = np.delete(slice_xy, nan_cols, 1)
    avg_positions = np.delete(avg_positions, nan_cols, 0)

    for i, frame in enumerate(slice_nonan):
        frame_nan = np.argwhere(np.isnan(frame)).reshape(-1)
        frame_nonan = np.delete(frame, frame_nan)

        cost_matrix = cdist(frame.reshape(-1, 2), avg_positions.reshape(-1, 2))
        cost_matrix = np.where(np.isnan(cost_matrix), 1e6, cost_matrix)

        row, col = linear_sum_assignment(cost_matrix)

        solved_frame = np.full((10, 2), np.nan)
        solved_frame[row] = frame.reshape(-1, 2)[col]
        solved_positions[i] = solved_frame

    return solved_positions


def template_matching(avg_positions_scaled, templates):
    """Template matching algorithm by Müller-Budack et al."""
    scores = {}
    for formation, coords in templates.items():
        coords = np.array(coords)

        form_min_x, form_max_x = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
        form_min_y, form_max_y = np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1])

        scaled_form_x = (coords[:, 0] - form_min_x) / (form_max_x - form_min_x)
        scaled_form_y = (coords[:, 1] - form_min_y) / (form_max_y - form_min_y)
        scaled_form = np.column_stack((scaled_form_x, scaled_form_y))

        cost_matrix = np.square(cdist(avg_positions_scaled, scaled_form))
        row, col = linear_sum_assignment(cost_matrix)

        cost = cost_matrix[row, col].mean()
        scores[formation] = 1 - cost * 3

    return scores


# === Main Script ===

# Load formation templates
with open("./data/templates.json") as f:
    templates = json.load(f)

# Load majority labels
labels = pd.read_csv("./data/majority.csv")

# Add timestamps in seconds
labels["start_seconds"] = labels["start"].str[:2].astype(int) * 60 + labels["start"].str[3:5].astype(int)
labels["end_seconds"] = labels["end"].str[:2].astype(int) * 60 + labels["end"].str[3:5].astype(int)

# Define halves
labels["half"] = "firstHalf"
labels.loc[labels["start_seconds"] >= 45 * 60, "half"] = "secondHalf"
labels.loc[labels['half'] == "secondHalf", 'start_seconds'] -= 45 * 60
labels.loc[labels['half'] == "secondHalf", 'end_seconds'] -= 45 * 60

# Split by match
label_by_match = {
    "DFL-MAT-0002UK": labels[labels["match"] == "Leverkusen - Gladbach"].reset_index(drop=True),
    "DFL-MAT-0002YP": labels[labels["match"] == "Leverkusen - Bremen"].reset_index(drop=True),
    "DFL-MAT-000303": labels[labels["match"] == "Bremen - Köln"].reset_index(drop=True),
    "DFL-MAT-000322": labels[labels["match"] == "Leverkusen - Köln"].reset_index(drop=True)
}

# Home/Away mapping
label_to_home = {
    "DFL-MAT-0002UK": {"Leverkusen": "Home", "Gladbach": "Away"},
    "DFL-MAT-0002YP": {"Leverkusen": "Home", "Bremen": "Away"},
    "DFL-MAT-000303": {"Bremen": "Home", "Köln": "Away"},
    "DFL-MAT-000322": {"Leverkusen": "Home", "Köln": "Away"}
}

# Direction of play
direction = {
    "firstHalf": {"Home": "lr", "Away": "rl"},
    "secondHalf": {"Home": "rl", "Away": "lr"}
}
rotation = {"lr": 90, "rl": -90}
framerate = 25

# Load position data
path = "<PATH_TO_FILES>"
match = "DFL-MAT-0002UK"
source = "SF"

kickoff = {
    "DFL-MAT-0002UK": {"firstHalf": 1695 - 1700, "secondHalf": 71593 - 71613},
    "DFL-MAT-0002YP": {"firstHalf": 4927 - 4935, "secondHalf": 74898 - 74885},
    "DFL-MAT-000303": {"firstHalf": 5669 - 5729, "secondHalf": 73869 - 73861},
    "DFL-MAT-000322": {"firstHalf": 3982 - 4003, "secondHalf": 73036 - 73040}
}

positions, _, _, teamsheet, pitch = read_position_data_xml(
    f"{path}/Positions/{match}.xml",
    f"{path}/Infos/{match}.xml"
)

if source in ["SF", "TV"]:
    visible = jd.load(f"./data/player_visibility/{source}_{match}_visible_with_ballstatus.json")

# Exclude goalkeepers
gk_home_xID = int(teamsheet["Home"].teamsheet.loc[teamsheet["Home"].teamsheet["position"] == "TW", "xID"])
gk_away_xID = int(teamsheet["Away"].teamsheet.loc[teamsheet["Away"].teamsheet["position"] == "TW", "xID"])

for half in positions:
    positions[half]["Home"].xy[:, 2 * gk_home_xID:2 * gk_home_xID + 2] = np.nan
    positions[half]["Away"].xy[:, 2 * gk_away_xID:2 * gk_away_xID + 2] = np.nan

# Apply visibility mask
if source in ["SF", "TV"]:
    for half in positions:
        for team in positions[half]:
            positions[half][team].x = np.where(visible[half][team] == 0, np.nan, positions[half][team].x)
            positions[half][team].y = np.where(visible[half][team] == 0, np.nan, positions[half][team].y)

# Calculate team centroids
centroids = {}
for half in positions:
    centroids.update({half: {}})
    for team in positions[half]:
        cm = CentroidModel()
        cm.fit(positions[half][team])
        centroids[half][team] = cm.centroid()

# Get home/away names
homeTeam = teamsheet["Home"].teamsheet["team"][0]
awayTeam = teamsheet["Away"].teamsheet["team"][0]

label_by_match[match]["predictions"] = None

# === Main Loop: Phase by Phase Detection ===
for idx, row in label_by_match[match].iterrows():
    start, end = row["start_seconds"], row["end_seconds"]
    half = row["half"]

    team_is_home = ((label_to_home[match][row["team"]] == "Home") and (row["possession"] == "in")) or \
                   ((label_to_home[match][row["team"]] == "Away") and (row["possession"] == "out"))
    in_pos = ["Away", "Home"][team_is_home]

    start_frame = max(start * framerate + kickoff[match][half], 0)
    end_frame = min(end * framerate + kickoff[match][half], len(positions[half]["Home"]))

    slice = positions[half][in_pos].slice(start_frame, end_frame)

    slice.rotate(rotation[direction[half][in_pos]])

    avg_pos = np.nanmean(slice.xy, axis=0)
    solved_pos = role_assignment(slice.xy, avg_pos)
    avg_pos_solved = np.nanmean(solved_pos, axis=0)

    # Normalize solved positions to match templates
    min_x, max_x = np.nanmin(avg_pos_solved[:, 0]), np.nanmax(avg_pos_solved[:, 0])
    min_y, max_y = np.nanmin(avg_pos_solved[:, 1]), np.nanmax(avg_pos_solved[:, 1])
    scaled_x = (avg_pos_solved[:, 0] - min_x) / (max_x - min_x)
    scaled_y = (avg_pos_solved[:, 1] - min_y) / (max_y - min_y)
    scaled_xy = np.column_stack((scaled_x, scaled_y))

    scaled_xy = scaled_xy[~np.isnan(scaled_xy).all(axis=1)]

    fsims = template_matching(scaled_xy, templates)

    # Save top 5 formation candidates
    label_by_match[match].at[idx, "predictions"] = sorted(fsims.items(), key=lambda x: x[1], reverse=True)[:5]

# Export
labels = pd.concat(label_by_match.values())
labels.to_csv(f"{path}formation_detection_{source}.csv", index=False)
