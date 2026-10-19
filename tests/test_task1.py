"""Unit tests for Task 1: Image Restoration and Segmentation.
Tests noise models, filter restorations, PSNR/SSIM calculation,
and morphological/segmentation pipeline using synthetic test patterns.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from src.utils import (
    compute_psnr,
    compute_ssim,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
)
from src.task1_restoration_segmentation import (
    run_image_restoration,
    run_morphological_and_segmentation,
)


class TestTask1RestorationAndSegmentation(unittest.TestCase):
    """Test suite for Task 1 mathematical and algorithmic operations."""

    @classmethod
    def setUpClass(cls):
        """Create a synthetic test image with touching geometric objects."""
        cls.test_dir = Path(__file__).resolve().parent / "temp_task1_output"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Synthetic RGB image with two touching circles and background
        cls.synthetic_img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Background gradient
        for y in range(200):
            cls.synthetic_img[y, :, 0] = y // 2
        # Circle 1
        cv2.circle(cls.synthetic_img, (75, 100), 40, (200, 200, 255), -1)
        # Circle 2 (touching Circle 1)
        cv2.circle(cls.synthetic_img, (135, 100), 40, (255, 200, 200), -1)

    def test_psnr_identical_images(self):
        """PSNR of an image with itself must be infinite."""
        psnr_val = compute_psnr(self.synthetic_img, self.synthetic_img)
        self.assertEqual(psnr_val, float("inf"))

    def test_ssim_identical_images(self):
        """SSIM of identical images must be exactly 1.0."""
        ssim_val = compute_ssim(self.synthetic_img, self.synthetic_img)
        self.assertAlmostEqual(ssim_val, 1.0, places=4)

    def test_noise_injection_degrades_psnr(self):
        """Noisy images must have lower PSNR and SSIM than clean images."""
        noisy_gauss = add_gaussian_noise(self.synthetic_img, mean=0, sigma=25, seed=42)
        psnr_gauss = compute_psnr(self.synthetic_img, noisy_gauss)
        ssim_gauss = compute_ssim(self.synthetic_img, noisy_gauss)
        
        self.assertLess(psnr_gauss, 35.0)
        self.assertLess(ssim_gauss, 0.90)

    def test_median_filter_superior_for_salt_and_pepper(self):
        """Median filter must yield higher PSNR than Gaussian filter on impulse noise."""
        noisy_sp = add_salt_and_pepper_noise(self.synthetic_img, salt_prob=0.03, pepper_prob=0.03, seed=42)
        
        restored_gauss = cv2.GaussianBlur(noisy_sp, (5, 5), sigmaX=1.2)
        restored_median = cv2.medianBlur(noisy_sp, 5)
        
        psnr_gauss = compute_psnr(self.synthetic_img, restored_gauss)
        psnr_median = compute_psnr(self.synthetic_img, restored_median)
        
        self.assertGreater(psnr_median, psnr_gauss)

    def test_full_task1_restoration_execution(self):
        """Verify that run_image_restoration executes cleanly and produces metrics."""
        metrics = run_image_restoration(self.synthetic_img, self.test_dir)
        self.assertIn("gaussian_noisy", metrics)
        self.assertIn("sp_noisy_restored_median", metrics)
        self.assertTrue((self.test_dir / "fig1_noise_comparison.png").exists())
        self.assertTrue((self.test_dir / "fig4_comprehensive_restoration_grid.png").exists())

    def test_full_task1_morphological_execution(self):
        """Verify that morphology and watershed segmentation execute and produce artifacts."""
        metrics = run_morphological_and_segmentation(self.synthetic_img, self.test_dir)
        self.assertIn("otsu_threshold", metrics)
        self.assertIn("detected_marker_regions", metrics)
        self.assertTrue((self.test_dir / "fig6_morphological_transforms.png").exists())
        self.assertTrue((self.test_dir / "fig8_watershed_segmentation_pipeline.png").exists())

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
