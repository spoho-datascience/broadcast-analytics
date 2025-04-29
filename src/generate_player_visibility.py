"""
generate_player_visibility.py

This script demonstrates how player visibility was calculated by projecting
the camera field of view (FoV) onto player positions frame-by-frame.

In the study, the actual tracking data was used to determine which players were inside or
outside the visible pitch area at each moment.
Due to licensing restrictions, real tracking data cannot be published —
therefore, dummy player positions (based on standard formations) are generated here
to replicate the visibility calculation process transparently.

The output is a visibility mask saved as a JSON file,
indicating frame-by-frame player visibility for validation and illustration purposes.

---
Information for the User:

The dummy positions are generated using pre-defined 4-2-3-1 (home) and 3-5-2 (away) formations.
Precomputed pitch intersection files (`pitch_intersections/`) are provided in the supplemental material.
"""

import jdata as jd
import numpy as np
from floodlight import XY
from shapely.geometry import Point, Polygon
from alive_progress import alive_bar

from src.constants import MATCH_LENGTH, POSITIONS_4231, POSITIONS_352

# Match details
source = "TV"
match_id = "DFL-MAT-0002UK"

# Load precomputed pitch intersections
intersections = jd.load(f"./data/pitch_intersections/{source}_{match_id}_intersection.json")

# Formations for both teams (home: 4-2-3-1, away: 3-5-2)
home_formation = np.array(POSITIONS_4231)
away_formation = np.array(POSITIONS_352)

# Create dummy position data using fixed formations
dummy_positions = {
    "firstHalf": {
        "Home": XY(np.full((MATCH_LENGTH[match_id]["firstHalf"], 22), home_formation)),
        "Away": XY(np.full((MATCH_LENGTH[match_id]["firstHalf"], 22), away_formation))
    },
    "secondHalf": {
        "Home": XY(np.full((MATCH_LENGTH[match_id]["secondHalf"], 22), home_formation)),
        "Away": XY(np.full((MATCH_LENGTH[match_id]["secondHalf"], 22), away_formation))
    }
}

# Initialize visibility mask (1: visible, 0: not visible)
visibility = {
    half: {
        team: np.ones((MATCH_LENGTH[match_id][half], dummy_positions[half][team].N))
        for team in dummy_positions[half]
    }
    for half in dummy_positions
}

# Determine player visibility by checking if within the camera-view polygon
for half in dummy_positions:
    for team in dummy_positions[half]:
        print(f"Processing {half} - {team}")
        with alive_bar(len(dummy_positions[half][team]), force_tty=True) as bar:
            for frame_idx, frame in enumerate(dummy_positions[half][team]):
                polygon_coords = intersections[half][frame_idx]

                # Skip frame if intersection polygon is not available
                if polygon_coords is None:
                    visibility[half][team][frame_idx, :] = 0
                    continue

                # Create shapely Polygon from (x, y) coords
                pitch_polygon = Polygon(zip(polygon_coords[0], polygon_coords[1]))

                # CCheck if each player's position lies within the field of view polygon
                for player_idx, (x, y) in enumerate(zip(frame[::2], frame[1::2])):
                    if not np.isnan((x, y)).any():
                        player_point = Point(x, y)
                        if not player_point.within(pitch_polygon):
                            visibility[half][team][frame_idx, player_idx] = 0
                bar()

        # Mask visibility array where player positions are NaN
        visibility[half][team] = np.where(
            np.isnan(dummy_positions[half][team].x),
            np.nan,
            visibility[half][team]
        )

# Save the visibility dictionary
output_path = f"./data/player_visibility/{source}_{match_id}_visible_dummy.json"
jd.save(visibility, output_path)
