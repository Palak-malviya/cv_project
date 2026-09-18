#  Copy-Move Image Forgery Detection

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/license-MIT-blue)
[![Web App Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face_Spaces-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/copy-move-forgery-detector)

A robust, classical Computer Vision pipeline to detect single-image **copy-move forgeries** (where a region of an image is copied and pasted elsewhere in the same image to hide or duplicate elements). 

This project relies entirely on mathematical models and geometric verification rather than deep learning, meaning it requires **zero training data** and works out-of-the-box on any image.

---

##  Problem Statement
With the proliferation of digital media, image manipulation has become incredibly accessible. Detecting copy-move forgeries manually can be extremely difficult, especially if the pasted region has been slightly rotated, scaled, or blended.

This tool provides a fast, deterministic, and explainable way to flag suspicious images for digital forensics students, researchers, content moderators, and journalists.

---

##  How It Works (The Pipeline)

The detection algorithm is broken down into a 4-step pipeline:

1. **Keypoint Extraction (SIFT)**: Identifies distinctive, scale- and rotation-invariant features in the image. SIFT is used because it is highly robust to scaling, rotation, and slight noise in forged regions.
2. **Self-Matching (FLANN)**: Matches the image's keypoints against themselves. We use a modified Lowe's ratio test and a minimum spatial distance threshold to exclude trivial adjacent matches.
3. **Geometric Filtering (RANSAC)**: Fits an Affine transformation model to the matched pairs to remove outliers. This ensures that the matched points move together consistently, as a real copy-paste forgery would.
4. **Clustering & Localization (DBSCAN)**: Groups the verified matched keypoints into distinct spatial regions (the source and the destination) and computes convex hulls to localize the forgery boundaries.

---

##  Tech Stack
- **OpenCV**: SIFT feature extraction, RANSAC geometric filtering, visualization.
- **scikit-learn**: DBSCAN spatial clustering.
- **NumPy & SciPy**: Matrix operations and point math.
- **Matplotlib**: Generating visual summary grids.


##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ananyasharma-dev/CV-Project.git
   cd CV-Project
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: `opencv-contrib-python` is required for access to the SIFT algorithm).*


##  Usage

Run the script via the command line, pointing it to an input image:

```bash
python main.py --input sample_images/forged/my_image.jpg
```

### CLI Arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | **(Required)** Path to the input image to analyze. | None |
| `--output` | Directory to save reports and visual summaries. | `output/` |
| `--min-matches` | Minimum number of inlier matches to consider a forgery. | `4` |
| `--ransac-threshold` | Reprojection error threshold for RANSAC filtering. | `5.0` |
| `--dbscan-eps` | Spatial distance threshold for clustering points into regions. | `40.0` |
| `--verbose` | Enables detailed debug logging in the console. | `False` |


## Example Output

The tool will output a quick summary to the console:
```text
========================================
FORGERY DETECTION SUMMARY
========================================
Input Image       : sample_images/forged/my_image.jpg
Raw Matches       : 128
Inlier Matches    : 45
Suspected Regions : 2
Confidence Score  : 0.92 (0=Authentic, 1=Forged)
Total Time        : 0.42 s
========================================
```

It will also generate a **visual summary figure** (`.png`) and a **metrics report** (`.json`) inside the `output/` directory, showing the exact steps of the pipeline and the localized regions.



##  Testing

The repository includes unit tests for the core detection modules using synthetic data (ensuring controlled ground truth). To run the test suite:

```bash
python -m unittest discover tests/
```
## Project credits: **
Name = Ananya Sharma 
Reg. no = 24BAI10039
Computer vision project (Copy-Move Image Forgery Detection)
Prof. Neha Rathore 

## Web App Deployment
A Gradio web interface (`app.py`) is included. You can easily deploy this tool online for free using [Hugging Face Spaces](https://huggingface.co/spaces).

##  Known Limitations
Because this pipeline relies on classical keypoint matching, it may trigger **false positives** on images with highly symmetric or repetitive natural textures (e.g., brick walls, tiled floors, chain-link fences). This is a well-known limitation of local feature matching approaches.

Additionally, this tool specifically detects *copy-move* manipulations. It will not detect *splicing* (elements pasted from a completely different image) or AI-generated Deepfakes.
