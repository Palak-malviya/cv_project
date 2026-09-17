import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple
from utils import timeit

logger = logging.getLogger(__name__)

def get_convex_hull(cluster_points: np.ndarray) -> np.ndarray:
    """Computes the convex hull polygon around a set of points."""
    if len(cluster_points) < 3:
        return np.array([])
    return cv2.convexHull(np.float32(cluster_points))

def compute_confidence_score(num_inliers: int, region_area: float, image_area: float) -> float:
    """
    Computes a confidence score (0.0 to 1.0) for the forgery detection.
    
    Rationale:
    - More inliers strongly correlate with a true positive. We use a logarithmic scale 
      since after a certain number of matches, confidence is very high.
    - If the forged region area is too small relative to the image, it might be noise.
    
    Args:
        num_inliers: Total number of RANSAC inlier matches.
        region_area: Area of the localized regions.
        image_area: Total area of the image.
    """
    if image_area <= 0 or num_inliers == 0:
        return 0.0
        
    # Score based on number of matches (peaks at ~20 matches)
    match_score = min(1.0, np.log1p(num_inliers) / np.log1p(20))
    
    # Score based on area ratio
    area_ratio = region_area / image_area
    area_score = min(1.0, area_ratio * 100) # Assumes 1% of image is a significant forgery
    
    confidence = (match_score * 0.7) + (area_score * 0.3)
    return float(np.clip(confidence, 0.0, 1.0))

@timeit
def localize_regions(clusters: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
    """
    Localizes the forged regions using convex hulls.
    
    Returns:
        List of dictionaries containing hull, centroid, area, and point_count.
    """
    regions = []
    
    for src_pts, dst_pts in clusters:
        for pts in [src_pts, dst_pts]:
            if len(pts) >= 3:
                hull = get_convex_hull(pts)
                if len(hull) >= 3:
                    area = cv2.contourArea(hull)
                    
                    # Compute centroid
                    M = cv2.moments(hull)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                        
                    regions.append({
                        'hull': hull,
                        'centroid': (cx, cy),
                        'area': area,
                        'point_count': len(pts)
                    })
                    
    logger.info(f"Localized {len(regions)} sub-regions from clusters.")
    return regions
