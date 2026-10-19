"""Unit tests for Task 3: Object Recognition (Viola-Jones and YOLO).
Tests Haar cascade classifier loading, YOLO model inference,
bounding box rendering, and comparative reporting.
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from src.task3_object_recognition import (
    get_haar_cascade_path,
    detect_faces_haar,
    detect_objects_yolo,
    run_object_recognition,
)


class TestTask3ObjectRecognition(unittest.TestCase):
    """Test suite for Task 3 classical and deep learning detectors."""

    @classmethod
    def setUpClass(cls):
        """Prepare test environment and test image."""
        cls.test_dir = Path(__file__).resolve().parent / "temp_task3_output"
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a synthetic image containing geometric patterns
        cls.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        cls.test_img[:] = (200, 200, 200)
        # Draw a simulated face shape (head circle, eyes, mouth)
        cv2.circle(cls.test_img, (150, 150), 80, (230, 180, 150), -1)
        cv2.circle(cls.test_img, (120, 130), 10, (50, 50, 50), -1)
        cv2.circle(cls.test_img, (180, 130), 10, (50, 50, 50), -1)
        cv2.ellipse(cls.test_img, (150, 180), (30, 15), 0, 0, 180, (50, 50, 50), 3)

    def test_haar_xml_exists(self):
        """Haar cascade XML file must exist on the local filesystem."""
        path = get_haar_cascade_path()
        self.assertTrue(path.exists(), f"Haar XML missing at {path}")

    def test_haar_detector_execution(self):
        """Haar detector must execute without errors on synthetic input."""
        annotated, detections = detect_faces_haar(self.test_img)
        self.assertIsInstance(detections, list)
        self.assertEqual(annotated.shape, self.test_img.shape)

    def test_yolo_detector_execution(self):
        """YOLO detector must execute inference on an image without raising exceptions."""
        annotated, detections = detect_objects_yolo(self.test_img, model_name="yolov8n.pt", conf_threshold=0.25)
        self.assertIsInstance(detections, list)
        self.assertEqual(annotated.shape, self.test_img.shape)

    def test_full_task3_pipeline(self):
        """Complete Task 3 pipeline must execute and generate montage."""
        results = run_object_recognition(self.test_img, self.test_dir)
        self.assertEqual(results["status"], "success")
        self.assertIn("haar", results)
        self.assertIn("yolo", results)
        self.assertTrue((self.test_dir / "fig1_detection_comparison_montage.png").exists())

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
