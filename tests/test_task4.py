"""Unit tests for Task 4: Video Motion Estimation.
Validates frame differencing, GMM MOG2 background subtraction,
and Lucas–Kanade optical flow using synthetic motion sequences.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from src.task4_motion_estimation import (
    compute_frame_differencing,
    compute_gmm_background_subtraction,
    compute_lucas_kanade_optical_flow,
    compute_dense_optical_flow_hsv,
    run_motion_estimation_pipeline,
)


class TestTask4MotionEstimation(unittest.TestCase):
    """Test suite for Task 4 motion estimation algorithms."""

    @classmethod
    def setUpClass(cls):
        """Generate a synthetic motion sequence: moving circle across stationary background."""
        cls.test_dir = Path(__file__).resolve().parent / "temp_task4_output"
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.frames = []
        for i in range(6):
            # Background
            frame = np.full((240, 320, 3), 40, dtype=np.uint8)
            # Add some stationary background textures
            cv2.rectangle(frame, (20, 20), (80, 80), (120, 120, 120), -1)
            # Moving object (moving diagonally 15 pixels per step)
            cx = 100 + (i * 15)
            cy = 100 + (i * 8)
            cv2.circle(frame, (cx, cy), 22, (230, 230, 230), -1)
            cls.frames.append(frame)

    def test_frame_differencing_detects_motion(self):
        """Frame differencing must produce non-empty mask when object translates."""
        mask, annotated, regions = compute_frame_differencing(self.frames[0], self.frames[1])
        self.assertGreater(regions, 0)
        self.assertGreater(np.count_nonzero(mask), 50)

    def test_gmm_background_subtraction(self):
        """GMM MOG2 should identify foreground and construct background model."""
        clean_fg, bg_model, annotated = compute_gmm_background_subtraction(self.frames)
        self.assertEqual(clean_fg.shape, (240, 320))
        self.assertEqual(bg_model.shape, (240, 320, 3))
        self.assertGreater(np.count_nonzero(clean_fg), 0)

    def test_lucas_kanade_optical_flow(self):
        """Lucas-Kanade must detect and track motion vectors on moving features."""
        vis, vector_count, avg_speed = compute_lucas_kanade_optical_flow(self.frames[0], self.frames[1])
        self.assertGreater(vector_count, 0)
        self.assertGreater(avg_speed, 0.5)

    def test_dense_optical_flow(self):
        """Dense optical flow must yield an RGB HSV visualization."""
        hsv_vis = compute_dense_optical_flow_hsv(self.frames[0], self.frames[1])
        self.assertEqual(hsv_vis.shape, (240, 320, 3))
        self.assertEqual(hsv_vis.dtype, np.uint8)

    def test_full_task4_pipeline(self):
        """Complete pipeline must execute and generate montage."""
        results = run_motion_estimation_pipeline(self.frames, fps=30.0, output_dir=self.test_dir)
        self.assertEqual(results["status"], "success")
        self.assertIn("frame_differencing", results)
        self.assertIn("lucas_kanade", results)
        self.assertTrue((self.test_dir / "fig1_motion_estimation_comparison_grid.png").exists())

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
