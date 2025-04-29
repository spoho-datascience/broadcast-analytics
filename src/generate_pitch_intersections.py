"""
generate_pitch_intersections.py

This script projects the camera field of view into pitch coordinates
by transforming pixel corners with homography matrices frame-by-frame.
The output is a JSON file containing frame-wise pitch polygons,
which are used for player visibility masking.
"""

import warnings
import numpy as np
import pandas as pd
import jdata as jd
from shapely.geometry import Polygon as Pol
from scipy.signal import savgol_filter
from alive_progress import alive_bar

from src.utils import vid2pos_reader, generate_topview_mask, mask2pitchpolygon
from src.constants import MATCH_LENGTH

# Suppress warnings
warnings.filterwarnings("ignore")

# Set base path and match details
base_path = "./data/"
match_id = "DFL-MAT-0002UK"
video_source = "TV"

# Define file paths for homography data
file_first_half = f"{video_source}_S_{match_id}_H0_filtered.jsonl"
file_second_half = f"{video_source}_S_{match_id}_H1_filtered.jsonl"

# Load homography matrices
homography_data = {
    "firstHalf": vid2pos_reader(f"{base_path}homography_matrices/{file_first_half}"),
    "secondHalf": vid2pos_reader(f"{base_path}homography_matrices/{file_second_half}")
}

# Filter out invalid entries
for half in homography_data:
    homography_data[half] = [
        frame for frame in homography_data[half] if frame["homography"][0][0] is not None
    ]

# Extract offset frame numbers
frame_offset = {
    "firstHalf": homography_data["firstHalf"][0]["frame_number_refs"],
    "secondHalf": homography_data["secondHalf"][0]["frame_number_refs"]
}

# Initialize homography matrices (as NaN arrays)
homography_matrices = {
    half: np.full((MATCH_LENGTH[match_id][half], 3, 3), np.nan)
    for half in homography_data
}

print("Extract and convert homography matrices...")

for half in homography_matrices:
    # Fill known homographies
    for frame in homography_data[half]:
        row_idx = frame["frame_number_refs"] - frame_offset[half]
        homography_matrices[half][row_idx] = np.array(frame["homography"])

    # Interpolate missing values (up to 25 frames)
    for i in range(3):
        for j in range(3):
            homography_matrices[half][:, i, j] = pd.Series(
                homography_matrices[half][:, i, j]
            ).interpolate("linear", limit=25, limit_direction="both")

    # Smooth homography matrices using Savitzky-Golay filter
    homography_matrices[half] = savgol_filter(
        homography_matrices[half], window_length=31, polyorder=3, axis=0, mode="nearest"
    )

    # Convert to list of matrices
    homography_matrices[half] = list(homography_matrices[half])

# Define camera and pitch bounds
camera_bounds = np.array([[0, 0], [0, 720], [1280, 720], [1280, 0]])
pitch_polygon = Pol([(0, 0), (105, 0), (105, 68), (0, 68)])

# Calculate top-view pitch polygons

# Initialize result dictionary
pitch_intersections = {half: [] for half in homography_matrices}
target_scale = 1  # Scale of the top-view projection

# Generate polygons for each frame
for half in homography_matrices:
    with alive_bar(len(homography_matrices[half]), force_tty=True) as bar:
        for homography in homography_matrices[half]:
            mask = generate_topview_mask(homography, target_scale=target_scale).numpy()
            polygon = mask2pitchpolygon(np.round(mask), target_scale)
            pitch_intersections[half].append(polygon)
            bar()

# Convert Shapely Polygons to arrays of (x, y) coordinates
for half in pitch_intersections:
    pitch_intersections[half] = [
        np.array(polygon.exterior.xy) if hasattr(polygon, "exterior") else None
        for polygon in pitch_intersections[half]
    ]

# Save result as .json
output_path = f"{base_path}pitch_intersections/{video_source}_{match_id}_intersection.json"
jd.save(pitch_intersections, output_path)
