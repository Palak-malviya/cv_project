import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import logging
from typing import List, Dict, Tuple
from utils import timeit

logger = logging.getLogger(__name__)

@timeit
def draw_keypoints(image: np.ndarray, keypoints: tuple) -> np.ndarray:
    """Draws SIFT keypoints on the image."""
    return cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

@timeit
def draw_matches(image: np.ndarray, matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], float]]) -> np.ndarray:
    """Draws lines connecting matched point pairs within the same image."""
    img_matches = image.copy()
    for pt1, pt2, dist in matched_pairs:
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        cv2.line(img_matches, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(img_matches, p1, 3, (0, 255, 0), -1)
        cv2.circle(img_matches, p2, 3, (255, 0, 0), -1)
    return img_matches

@timeit
def draw_forged_regions(image: np.ndarray, regions: List[Dict]) -> np.ndarray:
    """Draws convex hulls around forged regions and labels them."""
    img_regions = image.copy()
    for i, region in enumerate(regions):
        hull = region['hull']
        cv2.polylines(img_regions, [np.int32(hull)], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # Label the region
        cx, cy = region['centroid']
        label = f"Region {chr(65+i)}"
        cv2.putText(img_regions, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return img_regions

@timeit
def save_report(results_dict: Dict, output_path: str):
    """Writes the metrics and results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    logger.info(f"Saved JSON report to {output_path}")

@timeit
def create_summary_figure(original: np.ndarray, keypoints_img: np.ndarray, matches_img: np.ndarray, final_img: np.ndarray, save_path: str):
    """Creates a 2x2 matplotlib grid of the processing steps and saves it."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Convert BGR to RGB for matplotlib
    axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(cv2.cvtColor(keypoints_img, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('SIFT Keypoints')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Inlier Matches (RANSAC)')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Detected Forged Regions')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved summary figure to {save_path}")
