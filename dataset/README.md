# Dataset Guidelines & Setup Instructions

## Overview
As per the CCA 3 Computer Vision Assignment guidelines, each student must use their **own personal photograph** and **self-recorded video**. Do not use stock or downloaded photos.

---

## Required Files and Placement

Please place your files directly inside this `dataset/` folder using the exact filenames below:

| File Name | Description | Assignment Task Usage | Recommendations |
|---|---|---|---|
| `original_photo.jpg` | Your primary captured photo | **Task 1** (Restoration & Segmentation)<br>**Task 2** (HOG & SIFT)<br>**Task 3** (Haar Cascade & YOLO) | Take a photo using your mobile phone of a scene with good lighting and clear objects (e.g., your desk with a notebook, bottle, pens, laptop, or your own face/portrait). |
| `second_photo.jpg` | A second photo of the **same object or scene** | **Task 2** (Feature Matching before and after transformation) | Take a second photo from a slightly different angle, distance, or lighting. The feature matching algorithm will correlate keypoints between `original_photo.jpg` and `second_photo.jpg`. |
| `original_video.mp4` | A 15–30 second self-recorded video | **Task 3** (Video frame recognition)<br>**Task 4** (Motion Estimation & Optical Flow) | Record a 15–30 second clip with a stable camera (keep hand steady or place phone on a desk). Have a person walk across or an object move across the frame so motion estimation can detect the foreground. |

---

## File Formats & Specifications
- **Image format**: `.jpg`, `.jpeg`, or `.png` (Standard RGB)
- **Image resolution**: Recommended between 720p (1280x720) and 1080p (1920x1080). Very high resolutions (4K/8K) will be automatically resized for fast processing.
- **Video format**: `.mp4` (H.264 encoded)
- **Video length**: 15 to 30 seconds

---

## Video File Sharing Note
Because video files are large, `*.mp4` is excluded in `.gitignore` so your Git repository stays lightweight.
Upload your raw video to your Google Drive and share the link with the instructor as requested:
- Faculty email: `rahul.inchal.inchal24@gmail.com` and `rahul.inchal.vf@mitwpu.edu.in`
- Include the Google Drive link in your final technical report.
