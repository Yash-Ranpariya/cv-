# CCA 3 — Computer Vision | Video Presentation Script
### Student: **Yash Ranpariya** | PRN: **272251066**
### Course: MCA | MIT World Peace University, Pune | Faculty: Prof. Rahul Inchal

---

## ⏱️ Video Overview & Time Allocation
- **Target Duration**: 6 to 8 minutes
- **Recording Setup**: Screen + Webcam recording (OBS Studio, Zoom, or Loom)
- **Audio**: Quiet environment with crisp microphone

---

## 🎬 [0:00 – 1:00] Section 1: Introduction & Dataset

**📺 What to show on screen:**
Open `main.py` in VS Code, and show the `dataset/` folder in the explorer sidebar.

**🗣️ Spoken Script:**
> "Hello, respected faculty and evaluators.
>
> My name is **Yash Ranpariya**, PRN **272251066**, studying Master of Computer Applications at MIT World Peace University, Pune.
>
> Today, I am presenting my Continuous Assessment 3 (CCA 3) practical coding assignment in Computer Vision, covering **Units 3, 4, and 5**, under the guidance of **Prof. Rahul Inchal**.
>
> For this assignment, I captured my own original media files without relying on stock or downloaded images:
> 1. `original_photo.jpg` — my primary photograph used for image restoration, feature extraction, and object recognition.
> 2. `second_photo.jpg` — a second photograph of the same scene captured from a different perspective for feature matching.
> 3. `original_video.mp4` — my 15-second self-recorded video clip with moving foreground for motion estimation.
>
> The codebase is organized into modular Python files inside `src/`, automated tests inside `tests/`, and all generated visualizations in `outputs/`.
>
> Let us begin with Task 1."

---

## 🎬 [1:00 – 2:30] Section 2: Task 1 — Image Restoration & Segmentation (Unit 3)

**📺 What to show on screen:**
Open `outputs/task1_restoration_segmentation/fig4_comprehensive_restoration_grid.png` and `fig8_watershed_segmentation_pipeline.png`.

**🗣️ Spoken Script:**
> "Task 1 addresses two fundamental problems: degraded image reconstruction and object boundary segmentation.
>
> **Part A: Image Restoration:**
> I degraded the photograph using two distinct noise formulations:
> 1. Additive White Gaussian Noise (AWGN) with sigma 25, simulating electronic sensor thermal noise.
> 2. Salt-and-Pepper impulse noise with 3% probability, simulating transmission bit corruption.
>
> To restore the image, I implemented both a linear 2D Gaussian filter and a non-linear Median filter.
>
> Looking at our quantitative evaluation:
> - On Gaussian noise, the Gaussian filter smoothed high-frequency variance, achieving a PSNR of **28.82 dB** and SSIM of **0.7405**.
> - On Salt-and-Pepper noise, the Gaussian filter failed because it linearly averaged the extreme 0 and 255 outliers into blurry patches, yielding only 26.67 dB.
> - In contrast, the Median filter achieved an outstanding PSNR of **31.31 dB** and SSIM of **0.8949**. This is because the median operation sorts the local neighborhood; extreme salt and pepper spikes fall at the far tails of the sorted list and get completely rejected, leaving edges crisp.
>
> **Part B: Morphological Operations and Segmentation:**
> Next, I converted the image to grayscale and applied Otsu's thresholding to automatically maximize inter-class variance.
>
> I then applied morphological operations:
> - **Erosion** to strip outer boundaries and eliminate small specks.
> - **Dilation** to bridge small gaps.
> - **Opening and Closing** to clean the masks.
>
> Finally, I computed the Euclidean Distance Transform to extract sure foreground seeds and executed marker-controlled **Watershed Segmentation**, preventing oversegmentation and isolating touching regions with red boundary contours."

---

## 🎬 [2:30 – 4:00] Section 3: Task 2 — Feature Extraction and Matching (Unit 4)

**📺 What to show on screen:**
Open `outputs/task2_feature_extraction/fig1_hog_visualization.png` and `fig3_matching_comparison_summary.png`.

**🗣️ Spoken Script:**
> "Task 2 investigates local and structural feature descriptors from Unit 4.
>
> **Histogram of Oriented Gradients (HOG):**
> HOG computes 9-bin gradient orientation histograms within 8x8 pixel cells, normalized across 2x2 block groups. In my photo, HOG extracted a feature vector of **489,168 values**. The inferno-colored visualization on screen highlights the dominant edge orientations of the objects in the scene.
>
> **Scale-Invariant Feature Transform (SIFT):**
> SIFT detects stable interest points across a Difference-of-Gaussians scale space, assigns a canonical dominant orientation, and builds a 128-dimensional descriptor for each keypoint.
>
> I detected **2,000 keypoints** across both photos and matched them using **Lowe's Ratio Test** with a threshold of 0.75, backed by **RANSAC homography estimation**.
>
> - On the original scene pair, SIFT achieved **220 good matches with 111 RANSAC inliers (50.5% inlier ratio)**.
> - To rigorously verify invariance, I rotated the second photo by 35 degrees and scaled it down to 0.8x.
> - Remarkably, SIFT maintained **219 good matches and 106 inliers (48.4% inlier ratio)**.
>
> This demonstrates that SIFT reliably preserves matching fidelity despite significant scale and rotational variations."

---

## 🎬 [4:00 – 5:30] Section 4: Task 3 — Object Recognition (Unit 5)

**📺 What to show on screen:**
Open `outputs/task3_object_recognition/fig1_detection_comparison_montage.png`.

**🗣️ Spoken Script:**
> "Task 3 presents a comparative study between two milestones in computer vision: the classical Viola–Jones detector from 2001 and the modern Ultralytics YOLOv8 deep learning network.
>
> **Viola–Jones (Haar Cascade):**
> Viola–Jones uses rectangular Haar-like features computed in constant time via an Integral Image, selected by AdaBoost in an attentional cascade.
> - On my photo, it successfully detected **2 frontal faces**, highlighted with blue bounding boxes.
> - Its limitation is rigid feature templates: when a face is tilted beyond 20 degrees or under asymmetric lighting, it fails completely, and it provides no continuous confidence score.
>
> **Ultralytics YOLOv8 (Deep Learning):**
> YOLOv8 is a single-stage anchor-free convolutional architecture that detects across 80 MS COCO classes with calibrated continuous confidence probabilities.
> - It accurately localized **3 objects** with bounding boxes and high confidence scores.
>
> This illustrates the paradigm shift from handcrafted heuristic filters to deep hierarchical representation learning."

---

## 🎬 [5:30 – 7:00] Section 5: Task 4 — Video Motion Estimation (Unit 5)

**📺 What to show on screen:**
Open `outputs/task4_motion_estimation/fig1_motion_estimation_comparison_grid.png`, `08_lucas_kanade_motion_vectors.png`, and `09_dense_optical_flow_hsv.png`.

**🗣️ Spoken Script:**
> "Task 4 evaluates spatio-temporal dynamics on my self-recorded video using four complementary motion estimation methods:
>
> 1. **Temporal Frame Differencing:** Subtracting adjacent frames detected **6 motion clusters**. It is fast, but suffers from the ghosting problem: when a moving object stops, its mask disappears immediately.
> 2. **Gaussian Mixture Model (GMM MOG2):** Models each background pixel with 3 to 5 Gaussians, isolating moving foreground silhouettes while filtering out dynamic shadows.
> 3. **Lucas–Kanade Sparse Optical Flow:** Solves the brightness constancy constraint over Shi-Tomasi corners. On screen, the yellow arrow vectors indicate motion direction and velocity. It tracked **215 motion vectors with an average speed of 2.64 pixels per frame**.
> 4. **Dense Farneback Optical Flow:** Computes a continuous per-pixel flow field visualized via an HSV color wheel, where hue encodes motion angle and brightness represents speed.
>
> While frame differencing and GMM supply spatial detection masks, optical flow provides true velocity vectors essential for tracking and trajectory forecasting."

---

## 🎬 [7:00 – 7:30] Section 6: Conclusion

**📺 What to show on screen:**
Show the terminal running `python main.py --validate` or the GitHub repository page.

**🗣️ Spoken Script:**
> "In conclusion, this project implemented and verified the full pipeline of computer vision:
> - Restoration and morphological segmentation in Unit 3,
> - HOG and SIFT invariant feature matching in Unit 4,
> - Classical versus deep learning detection in Unit 5, and
> - Multi-algorithm motion estimation in Unit 5.
>
> All code is pushed to my GitHub repository at `github.com/Yash-Ranpariya/cv-`, and the video recording has been shared on Google Drive with Prof. Rahul Inchal.
>
> Thank you very much for your time and guidance."
