"""
utils.py

This module provides helper functions for:

- Reading vid2pos output files containing homography matrices (`vid2pos_reader`).
- Warping video frames into pitch coordinates (`generate_topview_mask`).
- Converting field of view masks into Shapely polygon objects (`mask2pitchpolygon`).
- Calculating distance covered per player across defined speed zones (`distance_covered_per_zone`).
"""

import jsonlines
import torch
import kornia
import shapely
import rasterio.features
import shapely.affinity
import numpy as np
import pandas as pd


def vid2pos_reader(file):
    """
    Reads output from vid2pos.
    Parameters
    ----------
    file: str
        Path to position file

    Returns
    -------
    vid2pos_output: List of dict
        List with dicts for every frame:
            frame_number_refs: Frame number
            homography: homography matrix for the direct linear transformation between
            video and pitch coordinate systems
            loss_ndc_circles, loss_total: loss described in the original publication
    """

    vid2pos_output = []
    with jsonlines.open(file) as jlf:
        for frame in jlf.iter(type=dict, skip_invalid=True):
            vid2pos_output.append(frame)

    return vid2pos_output


def generate_topview_mask(h: torch.tensor, source_size=(720, 1280), target_size=(68, 105), target_scale=1.):
    def _warp_img(H, img):
        # scaling matrix for better image resolution
        S = torch.eye(3).unsqueeze(0)
        S[:, 0, 0] = S[:, 1, 1] = target_scale
        # translate center of the homography matrix to the correct image origin (upper left)
        T = torch.eye(3).unsqueeze(0)
        T[:, 0, -1] = target_size[1] / 2
        T[:, 1, -1] = target_size[0] / 2
        warped_top = kornia.geometry.transform.homography_warp(
            img.unsqueeze(0),
            S @ T @ H,
            dsize=(int(target_size[0] * target_scale), int(target_size[1] * target_scale)),
            normalized_homography=False,
            normalized_coordinates=False,
            mode="nearest",
        ).squeeze()
        return warped_top

    img_source = torch.ones(3, *source_size, dtype=torch.double)

    warped_top = _warp_img(h, img_source)
    warped_top = warped_top[0]  # return binary mask, i.e. first channel only
    return warped_top

def mask2pitchpolygon(mask: np.ndarray, target_scale: float):

    def _mask_to_polygons_layer(mask:np.array) -> shapely.geometry.Polygon:
        """Converting mask to polygon object

        Input:
            mask: (np.array): Image like Mask [0,1] where all 1 are consider as masks

        Output:
            shapely.geometry.Polygon: Polygons

        """
        all_polygons = []
        for shape, value in rasterio.features.shapes(mask.astype(np.int16), mask=(mask >0)):
            all_polygons.append(shapely.geometry.shape(shape))

        all_polygons = shapely.geometry.MultiPolygon(all_polygons)

        if not all_polygons.is_valid:
            all_polygons = all_polygons.buffer(0)
            # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
            # need to keep it a Multi throughout
            if all_polygons.geom_type == 'Polygon':
                all_polygons = shapely.geometry.MultiPolygon([all_polygons])

        return all_polygons

    polygons = _mask_to_polygons_layer(mask)
    # print(type(polygons))
    if len(list(polygons.geoms)) != 0:
        polygon = list(polygons.geoms)[0]
        polygon = shapely.affinity.scale(polygon, yfact=-1, origin=(105/2, 68/2)).convex_hull
        return polygon
    else:
        return None


def distance_covered_per_zone(distances, velocities, speed_zones, speed_zone_names=None):
    """Calculates the distance covered by each player for given speed thresholds.

    Parameters
    ----------
    distances: PlayerProperty
        Property object containing covered distances for each player and each frame
        (T x N_players), e.g. as returned by floodlight.models.kinematics.DistanceModel.
    velocities: PlayerProperty
        Property object containing current velocity for each player and each frame
        (T x N_players), e.g. as returned by floodlight.models.kinematics.VelocityModel.
    speed_zones

    Returns
    -------
    distance_covered_per_zone: pd.DataFrame
        DataFrame containing the total distance covered by each player in each speed
        zone.
    """
    # param
    N_zones = len(speed_zones)
    if speed_zone_names is None:
        speed_zone_names = [
            f"{min_speed} to {max_speed}" for min_speed, max_speed in speed_zones
        ]

    # bin
    distances_per_zone = np.full((velocities.property.shape[1], N_zones), np.nan)

    # loop zones
    for i, (min_speed, max_speed) in enumerate(speed_zones):
        # create mask and mask array
        speed_mask = np.bitwise_and(
            velocities.property >= min_speed,
            velocities.property < max_speed
        )
        masked_distances = np.ma.masked_array(distances.property, ~speed_mask)
        # calculate and assign total distance
        distances_in_speed_zone = np.nansum(masked_distances, axis=0)
        distances_per_zone[:, i] = distances_in_speed_zone.data

    # assemble
    df = pd.DataFrame(data=distances_per_zone, columns=speed_zone_names)

    return df