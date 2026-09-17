import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import logging
from typing import List, Tuple, Dict

from utils import timeit

logger = logging.getLogger(__name__)

@timeit
def ransac_filter(matched_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], float]], threshold: float = 5.0) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], float]], np.ndarray]:
    """
    Filters matched pairs using RANSAC (Random Sample Consensus) by fitting a geometric transform.
    This step removes spurious matches (outliers) that do not follow a consistent geometric
    transformation (like affine or homography), which is typical in copy-move forgeries.
    
    Returns:
        inlier_pairs: List of matches that fit the model.
        matrix: The estimated affine transformation matrix (or None if failed).
    """
    if len(matched_pairs) < 4:
        logger.warning("Not enough matches for RANSAC (need at least 4).")
        return [], None
        
    src_pts = np.float32([m[0] for m in matched_pairs]).reshape(-1, 1, 2)
    dst_pts = np.float32([m[1] for m in matched_pairs]).reshape(-1, 1, 2)
    
    # We use Affine2D as copy-move usually involves translation, rotation, and scaling.
    # Homography could also be used but Affine is often sufficient and more stable here.
    matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    
    if inliers is None:
        return [], None
        
    inliers = inliers.ravel().tolist()
    inlier_pairs = [matched_pairs[i] for i, val in enumerate(inliers) if val == 1]
    
    logger.info(f"RANSAC filtering retained {len(inlier_pairs)} inliers from {len(matched_pairs)} matches.")
    return inlier_pairs, matrix

@timeit
def cluster_matches(inlier_pairs: List[Tuple[Tuple[float, float], Tuple[float, float], float]], eps: float = 40.0, min_samples: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Clusters the inlier matched points to separate different forged regions.
    Uses DBSCAN on the combined locations of source and destination points.
    
    Returns:
        List of tuples, where each tuple is (source_cluster_points, dest_cluster_points).
    """
    if not inlier_pairs:
        return []
        
    # Combine all points for clustering
    src_pts = [m[0] for m in inlier_pairs]
    dst_pts = [m[1] for m in inlier_pairs]
    all_pts = np.array(src_pts + dst_pts)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_pts)
    labels = clustering.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
        
    clusters = []
    # DBSCAN clusters all points. A genuine copy-move involves matching points that 
    # typically fall into two distinct spatial clusters (source and target).
    # We simply group by labels.
    for label in unique_labels:
        # Get indices of points in this cluster
        indices = np.where(labels == label)[0]
        
        cluster_src = []
        cluster_dst = []
        
        for idx in indices:
            n = len(src_pts)
            if idx < n:
                cluster_src.append(src_pts[idx])
            else:
                cluster_dst.append(dst_pts[idx - n])
                
        if len(cluster_src) > 0 and len(cluster_dst) > 0:
            clusters.append((np.array(cluster_src), np.array(cluster_dst)))
            
    logger.info(f"Clustered matches into {len(clusters)} regions.")
    return clusters
