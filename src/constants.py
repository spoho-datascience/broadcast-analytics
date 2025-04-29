"""
constants.py

This module defines constants used across the project, including:

- MATCH_LENGTH: Number of frames for first and second halves per match.
- POSITIONS_4231: Idealized player coordinates for a 4-2-3-1 formation for dummy data.
- POSITIONS_352: Idealized player coordinates for a 3-5-2 formation for dummy data.
"""


MATCH_LENGTH = {
    "DFL-MAT-0002UK": {"firstHalf": 69090, "secondHalf": 72455},
    "DFL-MAT-0002YP": {"firstHalf": 69095, "secondHalf": 72717},
    "DFL-MAT-000303": {"firstHalf": 67634, "secondHalf": 70741},
    "DFL-MAT-000322": {"firstHalf": 67702, "secondHalf": 68178}
}

POSITIONS_4231 = [
    # Goalkeeper
    5, 34,
    # Defenders (LB, LCB, RCB, RB)
    20, 10,
    20, 25,
    20, 43,
    20, 58,
    # Defensive Mids (CDM1, CDM2)
    35, 25,
    35, 43,
    # Attacking Mids (LAM, CAM, RAM)
    50, 18,
    50, 34,
    50, 50,
    # Striker
    70, 34
]

POSITIONS_352 = [
    # Goalkeeper
    100, 34,
    # Center-backs (RCB, CB, LCB)
    85, 20,
    85, 34,
    85, 48,
    # Wing-backs (RWB, LWB)
    75, 10,
    75, 58,
    # Midfielders (RCM, CM, LCM)
    65, 24,
    65, 34,
    65, 44,
    # Forwards (Right ST, Left ST)
    40, 28,
    40, 40
]