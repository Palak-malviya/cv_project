import unittest
import numpy as np
import cv2
from detection.keypoint_matcher import extract_sift_features, self_match

class TestKeypointMatcher(unittest.TestCase):
    def test_self_match(self):
        # Create a synthetic image with a clear copy-move forgery
        img = np.zeros((300, 300), dtype=np.uint8)
        
        # Draw a shape (source region)
        cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
        # Add some texture to make it feature-rich for SIFT
        cv2.putText(img, "TEST", (55, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 1)
        
        # Copy the region and paste it elsewhere (forged region)
        patch = img[50:100, 50:100].copy()
        img[200:250, 200:250] = patch
        
        keypoints, descriptors = extract_sift_features(img)
        matches = self_match(keypoints, descriptors, min_pixel_distance=20)
        
        # We expect to find multiple matches between the two squares
        self.assertTrue(len(matches) > 0, "Should find matches between copied regions.")
