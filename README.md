# Broadcast analytics - Oportunities and limitations of single camera tracking systems

This repository contains the source code and processing pipelines used in the paper:
> Bassek, M., Theiner, J., Ewerth, R., Memmert, D., Raabe, D. (2025). Broadcast analytics - 
> opportunities and limitations of single camera trackings systems. In Revision.

---

## Project Structure

| Folder/File | Description                                                                                                                          |
|:------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| `data/` | Folder containing input data: templates, rater labels, and subfolders for processed outputs (pitch intersections, visibility masks). |
| `src/` | Folder containing all Python scripts for processing: intensity calculation, formation detection, visibility generation, etc.         |
| `prototype_plots.ipynb` | Jupyter notebook illustrating the visibility masking process on synthetic data.                                                      |
| `supplemental_statistics.ipynb` | Jupyter notebook reproducing the statistics and figures shown in the final publication.                                              |

---

## Main Scripts

| Script | Purpose |
|:-------|:--------|
| `src/calculate_intensity_metrics.py` | Calculates player intensity metrics based on tracking and visibility data. |
| `src/formation_detection.py` | Detects player formations using role assignment and template matching. |
| `src/generate_pitch_intersections.py` | Projects video field of view onto pitch coordinates frame-by-frame. |
| `src/generate_player_visibility.py` | Demonstrates the visibility masking process using dummy position data. |
| `src/constants.py` | Centralized constants such as formation templates and match lengths. |
| `src/utils.py` | Helper functions for reading files, projecting homographies, and more. |

---

## Data Availability

** Raw Tracking and Projection Data**

Due to file size restrictions, the following raw or intermediate files are **not included directly** in this repository:
- `data/homography_matrices/` (raw homography output from vid2pos)
- `data/pitch_intersections/` (per-frame field of view polygons)
- `data/player_visibility/` (per-frame player visibility masks)

These files are available as **supplemental material** associated with the publication.  
They will be hosted externally for download.

> **🔗 Placeholder:**  
> _[Download link to supplemental material will be added here after upload]_

---

## Reproducibility

Once the supplemental data is available, you can fully reproduce:
- The field of view projections
- Player visibility statistics
- Intensity and distance covered results
- Formation detection outcomes
- All figures shown in the paper

Simply run the Jupyter notebooks or individual scripts in the `src/` folder as needed.

---

## License

This project is released for academic purposes only.  
Please cite the associated publication if you use this code or reproduce any results.

---

