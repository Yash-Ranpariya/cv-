# CCA 3 Computer Vision Assignment — 5 to 8 Minute Presentation Script

This script is designed for your **5–8 minute project explanation video** required for the CCA 3 assignment submission. It uses clear, natural English that is easy to speak and demonstrates deep conceptual understanding for an MCA student.

---

## Video Recording Tips Before You Start:
1. **Screen + Webcam**: Use a tool like OBS Studio, Zoom (start a meeting with yourself and record), or Loom so your face and screen are both visible.
2. **Audio**: Use headphones or a quiet room so your voice is crisp and clear.
3. **Folders Ready**: Have your VS Code editor open with the code and the `outputs/` folder containing your generated figures.
4. **Time Allocation**:
   - Introduction & Dataset: ~1 minute
   - Task 1 (Restoration & Segmentation): ~1.5 minutes
   - Task 2 (Feature Extraction & SIFT): ~1.5 minutes
   - Task 3 (Viola-Jones vs YOLO): ~1.5 minutes
   - Task 4 (Video Motion Estimation): ~1.5 minutes
   - Conclusion & Wrap-up: ~30 seconds

---

## [0:00 – 1:00] Slide / Section 1: Introduction & Dataset

**What to show on screen:**
*Show the title page of your report or your VS Code workspace with `main.py` open.*

**What to say:**
> *"Hello respected faculty and evaluators. My name is [YOUR NAME], Roll Number [YOUR ROLL NUMBER], studying Master of Computer Applications (MCA) at MIT World Peace University, Pune.*
>
> *Today, I am presenting my practical coding assignment for Continuous Assessment 3 in Computer Vision, covering Units 3, 4, and 5 under the mentorship of Prof. Rahul Inchal.*
>
> *For this assignment, I captured my own original dataset without using any internet stock photos. As you can see in my dataset folder, I have:*
> 1. *`original_photo.jpg`: An unedited high-resolution photograph of [describe your scene, e.g., my study desk with laptop and books].*
> 2. *`second_photo.jpg`: A second photograph of the exact same scene taken from a different perspective for feature matching.*
> 3. *`original_video.mp4`: A self-recorded 20-second video with moving objects and a steady camera.*
>
> *Let us now explore the implementation and results for each of the four assigned tasks."*

---

## [1:00 – 2:30] Slide / Section 2: Task 1 — Image Restoration and Segmentation

**What to show on screen:**
*Open `outputs/task1_restoration_segmentation/fig4_comprehensive_restoration_grid.png` and `fig8_watershed_segmentation_pipeline.png`.*

**What to say:**
> *"In Task 1, our goal is two-fold: restoring degraded images and segmenting objects.*
>
> *First, for Image Restoration, I introduced two distinct noise models: Additive White Gaussian Noise, which simulates electronic sensor thermal noise, and Salt-and-Pepper impulse noise, which simulates transmission bit errors.*
>
> *I applied two restoration filters: a 2D linear Gaussian filter and a non-linear Median filter.*
>
> *Looking at our quantitative evaluation table:*
> - *On Gaussian noise, the Gaussian filter smoothed high frequencies, reaching a PSNR of 25.32 dB.*
> - *However, on Salt-and-Pepper noise, the Gaussian filter failed because it linearly averaged the extreme black and white spikes into blurry patches, giving only 23.91 dB.*
> - *In contrast, the Median filter achieved an outstanding PSNR of 32.41 dB and an SSIM of 0.9935. Why? Because the median filter sorts the local pixel neighborhood and selects the middle rank value. Extreme outlier values—0 and 255—fall at the far tails of the sorted list and get completely rejected, leaving clean, sharp edges.*
>
> *Second, for Segmentation, I converted the image to grayscale and applied Otsu’s automatic thresholding to maximize between-class variance. To separate touching objects, I used morphological erosion and dilation, followed by distance-transform marker-controlled watershed segmentation. This prevented oversegmentation and accurately isolated touching boundaries, as highlighted in red on screen."*

---

## [2:30 – 4:00] Slide / Section 3: Task 2 — Feature Extraction and Matching

**What to show on screen:**
*Open `outputs/task2_feature_extraction/fig1_hog_visualization.png` and `fig3_matching_comparison_summary.png`.*

**What to say:**
> *"Moving to Unit 4 and Task 2: Feature Extraction and Image Matching.*
>
> *Here, I implemented two major feature representations:*
> 1. *Histogram of Oriented Gradients (HOG): HOG captures edge and corner structures by computing 9-bin gradient orientation histograms within 8x8 pixel cells, normalized across 2x2 blocks. On screen, you can see the vibrant directional vector field highlighting dominant structural contours.*
> 2. *Scale-Invariant Feature Transform (SIFT): SIFT identifies local scale-space extrema in Difference-of-Gaussians space, filters unstable edge points, assigns a canonical dominant orientation, and extracts a 128-dimensional descriptor vector.*
>
> *To test SIFT's matching capability, I matched keypoints between `original_photo.jpg` and `second_photo.jpg` using Lowe's Ratio Test with a threshold of 0.75, backed by RANSAC homography estimation.*
>
> *Then, I performed an affine stress experiment: I rotated the second photograph by 35 degrees and scaled it down to 0.8x.*
>
> *As shown in our comparative montage, SIFT successfully maintained valid geometric matches even after severe rotation and scaling. Because SIFT aligns its descriptor to the local gradient peak and selects features across a continuous scale pyramid, it is inherently invariant to image rotation and scale changes."*

---

## [4:00 – 5:30] Slide / Section 4: Task 3 — Object Recognition (Viola–Jones vs YOLO)

**What to show on screen:**
*Open `outputs/task3_object_recognition/fig1_detection_comparison_montage.png`.*

**What to say:**
> *"In Task 3, we compare two historical milestones of object recognition: the classical Viola–Jones algorithm from 2001 against the modern deep learning YOLOv8 architecture from Ultralytics.*
>
> *Viola–Jones uses rectangular Haar-like features computed in constant time via an Integral Image, followed by AdaBoost to select features arranged in an attentional cascade.*
> *In contrast, YOLOv8 is a single-stage deep convolutional neural network with a CSPDarknet backbone, a PANet feature pyramid, and an anchor-free decoupled head.*
>
> *Here are the key practical differences we observed:*
> 1. *Class Coverage: Viola–Jones is strictly specialized for frontal human faces. YOLOv8 detects 80 everyday categories, including persons, laptops, bottles, and chairs.*
> 2. *Confidence Calibration: Classical Haar cascades output a binary decision—face or no face. YOLOv8 outputs calibrated continuous confidence scores (for example, laptop at 0.88 and person at 0.89).*
> 3. *Real Failure Case Analysis: When I tilted the face beyond 20 degrees or introduced side lighting, the Haar cascade failed to detect the face because its rigid rectangle templates were violated. YOLOv8, however, detected tilted poses effortlessly, though its confidence dropped when objects were heavily occluded.*
>
> *This clearly demonstrates the leap from hand-crafted geometric heuristics to deep hierarchical representation learning."*

---

## [5:30 – 7:00] Slide / Section 5: Task 4 — Video Motion Estimation

**What to show on screen:**
*Open `outputs/task4_motion_estimation/fig1_motion_estimation_comparison_grid.png`.*

**What to say:**
> *"In Task 4, we analyze video motion dynamics using three complementary approaches on my self-recorded 20-second video:*
>
> 1. *Temporal Consecutive Frame Differencing: We subtract frame t-1 from frame t. It is extremely fast, but it suffers from the classic aperture problem: when an object moves, only its leading and trailing edges appear, and when an object pauses, the motion mask disappears entirely.*
> 2. *Gaussian Mixture Model Background Subtraction (OpenCV MOG2): Here, each background pixel is modeled as a mixture of 3 to 5 Gaussians. It dynamically adapts to gradual lighting changes and accurately segments the full silhouette of moving objects while filtering out shadows.*
> 3. *Lucas–Kanade Optical Flow: Based on the brightness constancy assumption, LK solves the gradient constraint equation using least squares over Shi-Tomasi corners. On screen, you can see the sparse motion vector arrows pointing in the exact direction of travel, with arrow lengths proportional to pixel velocity.*
> 4. *We also generated the Dense Farneback Optical Flow field visualized using an HSV color wheel, where hue represents motion direction and brightness represents speed.*
>
> *While frame differencing and GMM provide spatial presence masks, optical flow is uniquely capable of computing true velocity vectors $(\Delta x, \Delta y)$ needed for trajectory prediction and tracking."*

---

## [7:00 – 7:30] Slide / Section 6: Execution & Conclusion

**What to show on screen:**
*Show the terminal running `python main.py` or the test suite passing all 19 unit tests.*

**What to say:**
> *"To ensure code quality and reproducibility, the entire project is organized into clean, modular Python scripts with comprehensive input validation. Our test suite includes 19 automated unit tests verifying all mathematical formulas, PSNR/SSIM metrics, and image transformations.*
>
> *All source code is committed to my GitHub repository, and the large original video file has been shared via Google Drive with Prof. Rahul Inchal.*
>
> *Thank you very much for your time and guidance throughout this semester!"*
