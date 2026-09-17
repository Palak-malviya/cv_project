# Copy-Move Forgery Detection

## Overview
This tool detects copy-move forgeries in images (where a region of an image is copied and pasted elsewhere in the same image) using classical computer vision techniques.

## How It Works
1. **SIFT Keypoint Extraction**: Identifies distinctive, scale- and rotation-invariant features in the image.
2. **Self-Matching**: Matches the image's keypoints against themselves using FLANN and a modified Lowe's ratio test to find duplicated features.
3. **RANSAC Filtering**: Fits a geometric transformation model (Affine) to the matched pairs to remove outliers and keep only spatially consistent matches.
4. **Clustering & Localization**: Uses DBSCAN to group the matched keypoints into distinct spatial regions and draws convex hulls around them.

## Features
- End-to-end classical CV pipeline without deep learning.
- High resilience to scaling, rotation, and slight noise in forged regions.
- Automatically generates a visual summary and a JSON report.
- Confidence scoring based on match strength and area.

## Tech Stack
- Python 3
- OpenCV (SIFT, RANSAC, geometry, visualization)
- scikit-learn (DBSCAN clustering)
- NumPy & SciPy (matrix operations)
- Matplotlib (summary figure generation)

## Installation & Setup
1. Clone this repository or navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the script via the command line:
```bash
python main.py --input sample_images/forged/my_image.jpg
```

Optional arguments:
- `--output`: Directory to save reports and visual summaries (default: `output/`).
- `--min-matches`: Minimum number of inlier matches to consider a forgery.
- `--ransac-threshold`: Reprojection error threshold for RANSAC.
- `--dbscan-eps`: Spatial distance threshold for clustering points into regions.
- `--verbose`: Enable verbose logging.

## Example Output
The tool will output a console summary, e.g.:
```
========================================
FORGERY DETECTION SUMMARY
========================================
Input Image       : sample_images/forged/my_image.jpg
Raw Matches       : 45
Inlier Matches    : 22
Suspected Regions : 2
Confidence Score  : 0.85 (0=Authentic, 1=Forged)
Total Time        : 0.35 s
========================================
```
It will also create a summary PNG and a JSON report in the `output/` directory.

## Testing
Run the unit tests using `unittest`:
```bash
python -m unittest discover tests/
```
