"""Unit tests for Task 2: Feature Extraction and Image Matching.
Validates HOG descriptor generation, SIFT keypoint detection,
Lowe's ratio matching, and geometric transformation invariance.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from src.task2_feature_extraction import (
    extract_and_visualize_hog,
    detect_sift_keypoints,
    match_sift_features,
    apply_geometric_transform,
    run_feature_extraction_and_matching,
)


class TestTask2FeatureExtraction(unittest.TestCase):
    """Test suite for Task 2 feature extraction and matching algorithms."""

    @classmethod
    def setUpClass(cls):
        """Create two synthetic textured images with rich corner and gradient structures."""
        cls.test_dir = Path(__file__).resolve().parent / "temp_task2_output"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        # Image 1: Rich textured geometric pattern
        cls.img1 = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add high-contrast checkerboard corners and shapes
        for y in range(0, 300, 30):
            for x in range(0, 300, 30):
                if ((x // 30) + (y // 30)) % 2 == 0:
                    cls.img1[y:y+30, x:x+30] = [240, 240, 240]
        # Draw distinctive shapes
        cv2.circle(cls.img1, (100, 100), 25, (50, 180, 50), -1)
        cv2.rectangle(cls.img1, (180, 150), (250, 220), (50, 50, 220), -1)

        # Image 2: Slightly shifted and transformed version of Image 1
        cls.img2 = apply_geometric_transform(cls.img1, angle_degrees=15.0, scale_factor=0.95)

    def test_hog_extraction(self):
        """HOG extraction should produce positive descriptor vector and valid metadata."""
        metrics = extract_and_visualize_hog(self.img1, self.test_dir)
        self.assertGreater(metrics["descriptor_length"], 100)
        self.assertEqual(metrics["orientations"], 9)
        self.assertTrue((self.test_dir / "01_hog_feature_map.png").exists())
        self.assertTrue((self.test_dir / "fig1_hog_visualization.png").exists())

    def test_sift_detection(self):
        """SIFT detector should discover keypoints with 128-dim descriptor vectors."""
        kp, desc, kp_img = detect_sift_keypoints(self.img1)
        self.assertGreater(len(kp), 10)
        self.assertIsNotNone(desc)
        self.assertEqual(desc.shape[1], 128)

    def test_sift_matching_with_geometric_transformation(self):
        """SIFT matching should find matches between textured image and transformed version."""
        kp1, desc1, _ = detect_sift_keypoints(self.img1)
        kp2, desc2, _ = detect_sift_keypoints(self.img2)
        
        good_matches, inliers, H, vis = match_sift_features(
            kp1, desc1, kp2, desc2, self.img1, self.img2
        )
        self.assertGreater(len(good_matches), 4)

    def test_full_task2_pipeline(self):
        """Test complete run_feature_extraction_and_matching pipeline execution."""
        results = run_feature_extraction_and_matching(self.img1, self.img2, self.test_dir)
        self.assertEqual(results["status"], "success")
        self.assertIn("original_matching", results)
        self.assertIn("transformed_matching", results)
        self.assertTrue((self.test_dir / "fig3_matching_comparison_summary.png").exists())

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
