import unittest
import numpy as np
from detection.geometric_filter import ransac_filter, cluster_matches

class TestGeometricFilter(unittest.TestCase):
    def test_ransac_filter(self):
        # Create synthetic consistent matches (translation by dx=10, dy=20)
        consistent = [
            ((0.0, 0.0), (10.0, 20.0), 0.1),
            ((5.0, 5.0), (15.0, 25.0), 0.1),
            ((10.0, 0.0), (20.0, 20.0), 0.1),
            ((0.0, 10.0), (10.0, 30.0), 0.1)
        ]
        
        # Add random noise matches
        noise = [
            ((100.0, 100.0), (0.0, 0.0), 0.5),
            ((50.0, 50.0), (200.0, 200.0), 0.6)
        ]
        
        matches = consistent + noise
        
        inliers, matrix = ransac_filter(matches, threshold=5.0)
        
        self.assertEqual(len(inliers), 4)
        self.assertIsNotNone(matrix)
