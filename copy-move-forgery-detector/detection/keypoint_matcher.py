import cv2
import numpy as np
import math
import logging
from typing import Tuple, List
from utils import NoKeypointsFoundError, timeit

logger = logging.getLogger(__name__)

@timeit
def extract_sift_features(gray_image: np.ndarray) -> Tuple[tuple, np.ndarray]:
    """
    Extracts SIFT keypoints and descriptors from a grayscale image.
    SIFT identifies distinctive local features that are invariant to scaling and rotation.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    
    if keypoints is None or len(keypoints) == 0:
        raise NoKeypointsFoundError("SIFT found no keypoints in the image.")
        
    logger.info(f"Extracted {len(keypoints)} SIFT keypoints.")
    return keypoints, descriptors

@timeit
def self_match(keypoints: tuple, descriptors: np.ndarray, min_pixel_distance: float = 20.0, ratio_threshold: float = 0.5) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Matches the image's keypoints against themselves to find copied regions.
    
    Uses FLANN matcher with k=3 to get the top 3 matches for each keypoint.
    - Match 1 is usually the keypoint itself (trivial).
    - Match 2 and Match 3 are used for Lowe's ratio test to ensure uniqueness.
    Also excludes matches that are spatially too close to each other.
    
    Returns:
        List of matches as pairs of (point1_xy, point2_xy, distance)
    """
    if descriptors is None or len(descriptors) < 3:
        return []

    # FLANN parameters for SIFT
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors, descriptors, k=3)
    
    filtered_matches = []
    
    for match_tuple in matches:
        if len(match_tuple) < 3:
            continue
            
        # The best match (m1) is usually the keypoint matching itself.
        # We look at m2 (second best) and m3 (third best) for the ratio test.
        m1, m2, m3 = match_tuple
        
        # Apply Lowe's ratio test between the 2nd and 3rd nearest neighbors
        if m2.distance < ratio_threshold * m3.distance:
            pt1 = keypoints[m2.queryIdx].pt
            pt2 = keypoints[m2.trainIdx].pt
            
            # Exclude matches that are spatially adjacent
            dist = math.dist(pt1, pt2)
            if dist > min_pixel_distance:
                filtered_matches.append((pt1, pt2, m2.distance))
                
    logger.info(f"Found {len(filtered_matches)} matches after self-matching and ratio test.")
    return filtered_matches
