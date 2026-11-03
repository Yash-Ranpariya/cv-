# CCA 3 Computer Vision Practical Assignment (Units 3, 4, 5)

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.14+-green.svg)](https://opencv.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end, reproducible practical coding assignment implementing core Computer Vision techniques across **Unit 3 (Restoration & Segmentation)**, **Unit 4 (Feature Extraction & Matching)**, and **Unit 5 (Object Recognition & Motion Estimation)**.

---

## 1. Project Overview & Academic Metadata

- **Course**: Master of Computer Applications (MCA)
- **Subject**: Computer Vision (Continuous Assessment — CCA 3)
- **Institution**: School of Computer Engineering and Technology, MIT World Peace University (MIT-WPU), Pune
- **Faculty Coordinator**: Prof. Rahul Inchal (`rahul.inchal.inchal24@gmail.com`, `rahul.inchal.vf@mitwpu.edu.in`)
- **Student Name**: `Yash Ranpariya`
- **PRN / Roll Number**: `272251066`
- **GitHub Repository**: `https://github.com/Yash-Ranpariya/cv-`
- **Project Explanation Video (Drive)**: `https://drive.google.com/`

---

## 2. Assignment Objectives

Each student implements and evaluates the following computer vision algorithms using their own captured photograph and self-recorded video:

1. **Task 1 — Image Restoration & Segmentation (Unit 3)**:
   - Inject Additive White Gaussian Noise (AWGN) and Salt-and-Pepper impulse noise.
   - Restore images using 2D spatial Gaussian Filtering and non-linear Median Filtering.
   - Measure reconstruction quality quantitatively using **PSNR (dB)** and **SSIM**.
   - Perform morphological operations (Erosion, Dilation, Opening, Closing), Canny Edge Detection, and marker-controlled Watershed Segmentation.

2. **Task 2 — Feature Extraction & Image Matching (Unit 4)**:
   - Extract and visualize **Histogram of Oriented Gradients (HOG)** 9-bin orientation vector fields.
   - Detect **Scale-Invariant Feature Transform (SIFT)** keypoints and compute 128-dimensional descriptors.
   - Perform feature matching between two photos of the same scene using Lowe's Ratio Test and RANSAC homography.
   - Subject the scene to geometric stress (35° rotation and 0.8x scaling) to verify scale and rotation invariance.

3. **Task 3 — Object Recognition (Unit 5)**:
   - Implement classical **Viola–Jones** face detection using Haar Cascade classifiers.
   - Implement modern deep learning multi-class object detection using **Ultralytics YOLOv8**.
   - Compare architectures, bounding box localization, and continuous confidence calibration.
   - Document empirical detector failure cases without fabricating results.

4. **Task 4 — Video Motion Estimation (Unit 5)**:
   - Extract consecutive frames from an original 15–30 second self-recorded video.
   - Implement Temporal Frame Differencing.
   - Segment moving foreground using Gaussian Mixture Model (**GMM MOG2**) background subtraction.
   - Estimate velocity vector fields using **Lucas–Kanade (LK)** optical flow and dense Farneback flow (HSV representation).

---

## 3. Project Directory Structure

```text
CCA3_Computer_Vision/
├── dataset/                                # Required student personal media files
│   ├── original_photo.jpg                  # Primary captured photograph
│   ├── second_photo.jpg                    # Second photo of same scene/object
│   ├── original_video.mp4                  # 15-30s self-recorded video clip
│   └── README.md                           # Dataset guidelines & specifications
├── outputs/                                # Generated output figures & montages
│   ├── task1_restoration_segmentation/     # PSNR/SSIM tables, filters, watershed
│   ├── task2_feature_extraction/           # HOG maps, SIFT keypoints, matching
│   ├── task3_object_recognition/           # Viola-Jones & YOLO bounding boxes
│   └── task4_motion_estimation/            # Differences, GMM masks, optical flow
├── src/                                    # Modular Python source modules
│   ├── __init__.py
│   ├── dataset_validator.py                # Dataset checks and video frame extractor
│   ├── task1_restoration_segmentation.py   # Unit 3 implementation
│   ├── task2_feature_extraction.py         # Unit 4 implementation
│   ├── task3_object_recognition.py         # Unit 5 (Detection) implementation
│   ├── task4_motion_estimation.py          # Unit 5 (Motion) implementation
│   ├── utils.py                            # Metrics, noise injection, plot helpers
│   └── models/                             # Bundled Haar cascade XML models
├── tests/                                  # Automated unit test suite (19 tests)
│   ├── test_task1.py
│   ├── test_task2.py
│   ├── test_task3.py
│   └── test_task4.py
├── report/                                 # Academic documentation
│   ├── academic_report.md                  # Comprehensive markdown report
│   ├── generate_pdf_report.py              # Automated ReportLab PDF generator
│   └── CCA3_CV_Academic_Report.pdf         # Compiled university-grade PDF
├── docs/                                   # Supplementary guides
│   ├── presentation_script.md              # 5-8 minute spoken video script
│   └── git_setup_guide.md                  # Git, GitHub, and Drive sharing steps
├── main.py                                 # Interactive CLI master entry point
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Excludes venv, weights, raw video
└── README.md                               # Project documentation (this file)
```

---

## 4. Setup and Installation

### Step 1: Clone or Open the Repository
```bash
git clone https://github.com/YOUR_USERNAME/CCA3_Computer_Vision.git
cd CCA3_Computer_Vision
```

### Step 2: Create and Activate a Python Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

Capture your own original media and copy the files into the `dataset/` directory:

| Filename | Type | Recommended Resolution | Description |
|---|---|---|---|
| `dataset/original_photo.jpg` | Photo | 1080p (1920x1080) | Clear photo of your study desk, laptop, books, or face |
| `dataset/second_photo.jpg` | Photo | 1080p (1920x1080) | Photo of the **same scene/object** from a different angle |
| `dataset/original_video.mp4` | Video | 720p / 1080p @ 30 FPS | 15–30s steady clip with a person/object moving |

*Tip: To check if your dataset files are placed correctly, run:*
```bash
python main.py --validate
```

---

## 6. How to Run the Project

### Interactive Command-Line Menu:
Simply launch:
```bash
python main.py
```
This presents the interactive menu:
```text
===========================================================================
      CCA 3 COMPUTER VISION — PRACTICAL CODING ASSIGNMENT (MCA)
      Units 3, 4 & 5 | OpenCV, NumPy, Matplotlib, skimage, YOLO
===========================================================================
 Select an option to execute:
  1. Run Task 1 — Image Restoration and Segmentation
  2. Run Task 2 — Feature Extraction and Image Matching
  3. Run Task 3 — Object Recognition (Viola–Jones & YOLO)
  4. Run Task 4 — Video Motion Estimation
  5. Run All Tasks
  6. Verify Dataset Status & Guidance
  7. Exit
---------------------------------------------------------------------------
```

### Non-Interactive CLI Commands:
You can also run specific tasks directly via terminal arguments:
```bash
# Run Task 1 (Restoration and Segmentation)
python main.py --task 1

# Run Task 2 (Feature Extraction & SIFT Matching)
python main.py --task 2

# Run Task 3 (Viola-Jones & YOLO Recognition)
python main.py --task 3

# Run Task 4 (Video Motion Estimation)
python main.py --task 4

# Run All Tasks sequentially
python main.py --task all
```

---

## 7. Running the Automated Test Suite

To verify code syntax, algorithms, transformations, and metric formulas without needing dataset files:
```bash
python -m unittest discover tests
```
*All 19 unit tests will execute and report passing status.*

---

## 8. Expected Outputs and Generated Artifacts

| Task | Output Directory | Key Generated Artifacts |
|---|---|---|
| **Task 1** | `outputs/task1_restoration_segmentation/` | `fig1_noise_comparison.png`<br>`fig2_gaussian_noise_restoration.png`<br>`fig3_sp_noise_restoration.png`<br>`fig4_comprehensive_restoration_grid.png`<br>`fig5_thresholding_comparison.png`<br>`fig6_morphological_transforms.png`<br>`fig7_canny_edge_detection.png`<br>`fig8_watershed_segmentation_pipeline.png` |
| **Task 2** | `outputs/task2_feature_extraction/` | `fig1_hog_visualization.png`<br>`fig2_sift_keypoints_detected.png`<br>`fig3_matching_comparison_summary.png` |
| **Task 3** | `outputs/task3_object_recognition/` | `fig1_detection_comparison_montage.png`<br>`01_haar_face_detections.png`<br>`02_yolo_object_detections.png` |
| **Task 4** | `outputs/task4_motion_estimation/` | `fig1_motion_estimation_comparison_grid.png`<br>`03_frame_difference_mask.png`<br>`05_gmm_foreground_mask.png`<br>`08_lucas_kanade_motion_vectors.png`<br>`09_dense_optical_flow_hsv.png` |

---

## 9. Compiling the Technical PDF Report

To compile the professional academic PDF report with university formatting:
```bash
python report/generate_pdf_report.py
```
Output PDF is saved at: `report/CCA3_CV_Academic_Report.pdf`.

---

## 10. Common Errors and Solutions

| Symptom / Error | Root Cause | Solution |
|---|---|---|
| `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'` | OpenCV 5 preview builds moved object detection modules. | Ensure `opencv-python<5.0.0` is installed via `pip install "opencv-python<5"`. |
| `[!] Input Missing for Task X` | Dataset files are not placed in `dataset/`. | Copy your photos and video to `dataset/` with exact names (`original_photo.jpg`, `second_photo.jpg`, `original_video.mp4`). |
| `ModuleNotFoundError: No module named 'ultralytics'` | Ultralytics package is missing. | Run `pip install ultralytics` or `pip install -r requirements.txt`. |
| Memory issues on 4K/8K images | Mobile cameras produce enormous resolution images. | The project automatically rescales images exceeding 1920px proportionally for compute efficiency. |

---

## 11. Git and Faculty Submission Instructions

1. Initialize Git and make your initial commit:
   ```bash
   git init -b main
   git add .
   git commit -m "feat: complete CCA 3 Computer Vision assignment implementation"
   ```
2. Create a repository on GitHub named `CCA3_Computer_Vision` and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/CCA3_Computer_Vision.git
   git push -u origin main
   ```
3. Upload raw video files and your 5–8 minute explanation video to Google Drive.
4. Share Drive access with:
   - `rahul.inchal.inchal24@gmail.com`
   - `rahul.inchal.vf@mitwpu.edu.in`
5. Follow the spoken script in `docs/presentation_script.md` to record your video presentation.
