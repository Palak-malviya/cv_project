import unittest
import numpy as np
from detection.region_localizer import get_convex_hull, compute_confidence_score, localize_regions

class TestRegionLocalizer(unittest.TestCase):
    def test_get_convex_hull(self):
        # 4 points forming a square
        pts = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        hull = get_convex_hull(pts)
        self.assertGreaterEqual(len(hull), 3)

    def test_localize_regions(self):
        src_pts = np.array([[0, 0], [0, 10], [10, 10]])
        dst_pts = np.array([[20, 20], [20, 30], [30, 30]])
        
        clusters = [(src_pts, dst_pts)]
        regions = localize_regions(clusters)
        
        self.assertEqual(len(regions), 2, "Should find 2 regions (source and destination).")
        
    def test_compute_confidence_score(self):
        score_high = compute_confidence_score(50, 1000, 10000)
        score_low = compute_confidence_score(2, 10, 10000)
        
        self.assertGreater(score_high, score_low)
