"""Automated High-Quality Video Presentation Generator for Yash Ranpariya (PRN 272251066).
Generates Full HD 1080p MP4 presentation video with slide visuals and SAPI5 voice narration.
"""

import os
import subprocess
import wave
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import win32com.client
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = PROJECT_ROOT / "dataset"
TEMP_DIR = PROJECT_ROOT / "temp_video_build"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FINAL_VIDEO_PATH = PROJECT_ROOT / "CCA3_Presentation_Yash_Ranpariya.mp4"

# Dimensions
WIDTH = 1920
HEIGHT = 1080
FPS = 30

# Colors
C_BG = (245, 247, 250)
C_HEADER_BG = (13, 59, 102)
C_HEADER_TEXT = (255, 255, 255)
C_CARD_BG = (255, 255, 255)
C_CARD_BORDER = (203, 213, 225)
C_PRIMARY = (13, 59, 102)
C_ACCENT = (230, 81, 0)
C_TEXT_MAIN = (30, 41, 59)
C_TEXT_MUTED = (100, 116, 139)
C_FOOTER_BG = (241, 245, 249)

# Fonts
FONT_TITLE = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
FONT_SUBTITLE = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
FONT_HEADING = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 26)
FONT_BODY = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
FONT_BOLD = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
FONT_SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
FONT_BADGE = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)


def generate_audio(text: str, output_wav: Path) -> float:
    """Generate audio file via SAPI5 and return its duration in seconds."""
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    
    # Set speech rate (-10 to 10; 0 is normal, 1 is slightly brisker, -1 is relaxed)
    speaker.Rate = 0
    speaker.Volume = 100
    
    stream.Open(str(output_wav), 3, False)  # SSFMCreateForWrite = 3
    speaker.AudioOutputStream = stream
    speaker.Speak(text)
    stream.Close()
    
    with wave.open(str(output_wav), "r") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return duration


def draw_header_footer(draw: ImageDraw.ImageDraw, section_num: int, total_sections: int, slide_title: str):
    """Draw standard university header and bottom info bar."""
    # Top Header Banner
    draw.rectangle([(0, 0), (WIDTH, 110)], fill=C_HEADER_BG)
    
    # Institution & Subject
    draw.text((45, 18), "MIT WORLD PEACE UNIVERSITY, PUNE — MCA DEPARTMENT", fill=(200, 220, 245), font=FONT_SUBTITLE)
    draw.text((45, 48), "CCA 3: COMPUTER VISION PRACTICAL ASSIGNMENT (UNITS 3, 4, 5)", fill=C_HEADER_TEXT, font=FONT_TITLE)
    
    # Student Badge on Right
    badge_x = WIDTH - 430
    draw.rounded_rectangle([(badge_x, 15), (WIDTH - 45, 95)], radius=8, fill=(24, 75, 125))
    draw.text((badge_x + 15, 23), "STUDENT: Yash Ranpariya", fill=(255, 255, 255), font=FONT_BADGE)
    draw.text((badge_x + 15, 47), "PRN: 272251066 | Faculty: Prof. Rahul Inchal", fill=(215, 230, 250), font=FONT_SMALL)
    draw.text((badge_x + 15, 68), "GitHub: github.com/Yash-Ranpariya/cv-", fill=(255, 215, 0), font=FONT_SMALL)
    
    # Section Subheader
    draw.rectangle([(0, 110), (WIDTH, 160)], fill=(235, 242, 250))
    draw.text((45, 122), f"SECTION {section_num} OF {total_sections}: {slide_title.upper()}", fill=C_PRIMARY, font=FONT_HEADING)
    
    # Bottom Footer Bar
    draw.rectangle([(0, HEIGHT - 45), (WIDTH, HEIGHT)], fill=C_FOOTER_BG)
    draw.line([(0, HEIGHT - 46), (WIDTH, HEIGHT - 46)], fill=C_CARD_BORDER, width=1)
    draw.text((45, HEIGHT - 33), "Continuous Assessment 3 | Computer Vision | Comprehensive Empirical Benchmark", fill=C_TEXT_MUTED, font=FONT_SMALL)
    draw.text((WIDTH - 420, HEIGHT - 33), "Automated Academic Walkthrough & Demonstration", fill=C_TEXT_MUTED, font=FONT_SMALL)


def paste_fitted_image(canvas: Image.Image, img_path: Path, target_box: tuple):
    """Load image from path and paste it centered inside target_box (x1, y1, x2, y2)."""
    if not img_path.exists():
        return
    x1, y1, x2, y2 = target_box
    box_w = x2 - x1
    box_h = y2 - y1
    
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    scale = min(box_w / img_w, box_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    paste_x = x1 + (box_w - new_w) // 2
    paste_y = y1 + (box_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))


# Slide Building Functions
def create_slide_1() -> Image.Image:
    """Slide 1: Introduction & Dataset."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 1, 8, "Project Introduction & Authentic Personal Dataset")
    
    # Left Card: Metadata & Assignment Objectives
    draw.rounded_rectangle([(45, 180), (780, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((75, 205), "Academic Project Overview", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(75, 245), (750, 245)], fill=C_CARD_BORDER, width=1)
    
    info_lines = [
        ("Candidate Name:", "Yash Ranpariya"),
        ("PRN / Roll Number:", "272251066"),
        ("Degree & Department:", "MCA, School of Computer Engineering"),
        ("University:", "MIT World Peace University, Pune"),
        ("Course Faculty:", "Prof. Rahul Inchal"),
        ("Subject:", "Computer Vision (CCA 3)"),
        ("Curriculum Scope:", "Units 3, 4, and 5 Implementation"),
        ("GitHub Repository:", "https://github.com/Yash-Ranpariya/cv-"),
    ]
    y = 265
    for label, val in info_lines:
        draw.text((75, y), label, fill=C_TEXT_MUTED, font=FONT_BOLD)
        draw.text((310, y), val, fill=C_TEXT_MAIN, font=FONT_BODY)
        y += 42
        
    y += 15
    draw.line([(75, y), (750, y)], fill=C_CARD_BORDER, width=1)
    y += 20
    draw.text((75, y), "Dataset Authenticity Guarantee:", fill=C_ACCENT, font=FONT_BOLD)
    y += 32
    draw.text((75, y), "• Strictly personal, unedited photographs taken from mobile camera.", fill=C_TEXT_MAIN, font=FONT_SMALL)
    y += 26
    draw.text((75, y), "• Self-recorded 15-second real-world video footage.", fill=C_TEXT_MAIN, font=FONT_SMALL)
    y += 26
    draw.text((75, y), "• Zero stock images or internet downloads used in experiments.", fill=C_TEXT_MAIN, font=FONT_SMALL)

    # Right Top Card: original_photo.jpg
    draw.rounded_rectangle([(820, 180), (1330, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((840, 200), "Primary Photo: original_photo.jpg", fill=C_PRIMARY, font=FONT_BOLD)
    draw.text((840, 226), "739 x 1600 px | Personal Capture", fill=C_TEXT_MUTED, font=FONT_SMALL)
    paste_fitted_image(img, DATASET_DIR / "original_photo.jpg", (840, 255, 1310, HEIGHT - 85))

    # Right Bottom Card: second_photo.jpg
    draw.rounded_rectangle([(1370, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1390, 200), "Secondary Photo: second_photo.jpg", fill=C_PRIMARY, font=FONT_BOLD)
    draw.text((1390, 226), "739 x 1600 px | Alternate Perspective", fill=C_TEXT_MUTED, font=FONT_SMALL)
    paste_fitted_image(img, DATASET_DIR / "second_photo.jpg", (1390, 255, 1855, HEIGHT - 85))

    return img


def create_slide_2() -> Image.Image:
    """Slide 2: Task 1 Part A - Image Restoration."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 2, 8, "Task 1 (Part A) — Gaussian vs Median Filter Restoration Benchmark")
    
    # Image Display Card
    draw.rounded_rectangle([(45, 180), (1280, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((65, 200), "Comprehensive Restoration Comparison Matrix (AWGN & Impulse S&P Noise)", fill=C_PRIMARY, font=FONT_BOLD)
    paste_fitted_image(img, OUTPUTS_DIR / "task1_restoration_segmentation" / "fig4_comprehensive_restoration_grid.png", (65, 235, 1260, HEIGHT - 80))

    # Right Panel: Quantitative Metrics Table & Insights
    draw.rounded_rectangle([(1310, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1335, 205), "Empirical Results (dB / SSIM)", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(1335, 245), (1850, 245)], fill=C_CARD_BORDER, width=1)
    
    y = 265
    draw.text((1335, y), "Gaussian Noise (sigma=25):", fill=C_ACCENT, font=FONT_BOLD)
    y += 28
    draw.text((1335, y), "• Corrupted PSNR:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1560, y), "20.59 dB (SSIM 0.2859)", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1335, y), "• Gaussian Filter:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1560, y), "28.82 dB (SSIM 0.7405)", fill=(34, 139, 34), font=FONT_BOLD)
    y += 28
    draw.text((1335, y), "• Median Filter:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1560, y), "28.44 dB (SSIM 0.7022)", fill=C_TEXT_MAIN, font=FONT_BOLD)
    
    y += 45
    draw.text((1335, y), "Salt & Pepper Noise (3%):", fill=C_ACCENT, font=FONT_BOLD)
    y += 28
    draw.text((1335, y), "• Corrupted PSNR:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1560, y), "17.06 dB (SSIM 0.2805)", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1335, y), "• Gaussian Filter:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1560, y), "26.67 dB (Fails / Blurs)", fill=(200, 0, 0), font=FONT_BOLD)
    y += 28
    draw.text((1335, y), "• Median Filter:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1560, y), "31.31 dB (SSIM 0.8949)", fill=(34, 139, 34), font=FONT_BOLD)
    
    y += 45
    draw.line([(1335, y), (1850, y)], fill=C_CARD_BORDER, width=1)
    y += 18
    draw.text((1335, y), "Theoretical Explanation:", fill=C_PRIMARY, font=FONT_BOLD)
    y += 28
    explanation = [
        "Gaussian filter is a linear weighted average.",
        "It smears extreme 0 & 255 spikes into cloudy",
        "blotches. Median filter sorts the neighborhood;",
        "impulses fall at the tails and get 100% rejected,",
        "recovering crisp edges."
    ]
    for line in explanation:
        draw.text((1335, y), line, fill=C_TEXT_MAIN, font=FONT_SMALL)
        y += 22

    return img


def create_slide_3() -> Image.Image:
    """Slide 3: Task 1 Part B - Morphology & Watershed."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 3, 8, "Task 1 (Part B) — Morphological Transforms & Watershed Segmentation")
    
    # Image Display Card
    draw.rounded_rectangle([(45, 180), (1330, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((65, 200), "Marker-Controlled Watershed Segmentation Pipeline (Distance Transform)", fill=C_PRIMARY, font=FONT_BOLD)
    paste_fitted_image(img, OUTPUTS_DIR / "task1_restoration_segmentation" / "fig8_watershed_segmentation_pipeline.png", (65, 235, 1310, HEIGHT - 80))

    # Right Card: Algorithmic Steps
    draw.rounded_rectangle([(1360, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1385, 205), "Segmentation Workflow", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(1385, 245), (1850, 245)], fill=C_CARD_BORDER, width=1)
    
    steps = [
        ("1. Grayscale Conversion", "Luminance reduction from 3-channel RGB."),
        ("2. Otsu's Thresholding", "Maximizes between-class variance automatically."),
        ("3. Morphological Opening", "Erosion followed by Dilation; removes noise."),
        ("4. Morphological Closing", "Dilation followed by Erosion; fills holes."),
        ("5. Distance Transform", "Calculates Euclidean distance to nearest zero pixel."),
        ("6. Sure Foreground Seeds", "Thresholded distance peaks act as marker seeds."),
        ("7. Watershed Flooding", "Topological flood divides touching objects with red boundaries."),
    ]
    y = 265
    for title, desc in steps:
        draw.text((1385, y), title, fill=C_ACCENT, font=FONT_BOLD)
        y += 25
        draw.text((1385, y), desc, fill=C_TEXT_MAIN, font=FONT_SMALL)
        y += 35

    return img


def create_slide_4() -> Image.Image:
    """Slide 4: Task 2 Part A - HOG Features."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 4, 8, "Task 2 (Part A) — Histogram of Oriented Gradients (HOG) Feature Extraction")
    
    draw.rounded_rectangle([(45, 180), (1350, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((65, 200), "HOG Gradient Orientations & Vector Visualization (9 Bins, 8x8 Cells)", fill=C_PRIMARY, font=FONT_BOLD)
    paste_fitted_image(img, OUTPUTS_DIR / "task2_feature_extraction" / "fig1_hog_visualization.png", (65, 235, 1330, HEIGHT - 80))

    draw.rounded_rectangle([(1380, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1405, 205), "HOG Architecture", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(1405, 245), (1850, 245)], fill=C_CARD_BORDER, width=1)
    
    hog_specs = [
        ("Extracted Vector Length:", "489,168 features"),
        ("Cell Dimensions:", "8 x 8 pixels"),
        ("Block Normalization:", "2 x 2 cells (L2-norm)"),
        ("Orientation Bins:", "9 bins (0 deg - 180 deg)"),
        ("Gradient Operator:", "Horizontal & Vertical Sobel"),
    ]
    y = 265
    for label, val in hog_specs:
        draw.text((1405, y), label, fill=C_TEXT_MUTED, font=FONT_BODY)
        y += 26
        draw.text((1405, y), val, fill=C_ACCENT, font=FONT_BOLD)
        y += 36
        
    y += 15
    draw.line([(1405, y), (1850, y)], fill=C_CARD_BORDER, width=1)
    y += 20
    draw.text((1405, y), "Core Purpose:", fill=C_PRIMARY, font=FONT_BOLD)
    y += 28
    draw.text((1405, y), "HOG characterizes shape and local edge", fill=C_TEXT_MAIN, font=FONT_SMALL)
    y += 22
    draw.text((1405, y), "directions, maintaining robustness against", fill=C_TEXT_MAIN, font=FONT_SMALL)
    y += 22
    draw.text((1405, y), "global illumination changes.", fill=C_TEXT_MAIN, font=FONT_SMALL)

    return img


def create_slide_5() -> Image.Image:
    """Slide 5: Task 2 Part B - SIFT Matching & Invariance."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 5, 8, "Task 2 (Part B) — SIFT Keypoint Detection & Invariance Verification")
    
    draw.rounded_rectangle([(45, 180), (1330, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((65, 200), "SIFT Feature Matching Under Geometric Stress (Original vs 35 deg Rotated + 0.8x Scale)", fill=C_PRIMARY, font=FONT_BOLD)
    paste_fitted_image(img, OUTPUTS_DIR / "task2_feature_extraction" / "fig3_matching_comparison_summary.png", (65, 235, 1310, HEIGHT - 80))

    draw.rounded_rectangle([(1360, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1385, 205), "SIFT Invariance Benchmark", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(1385, 245), (1850, 245)], fill=C_CARD_BORDER, width=1)
    
    y = 265
    draw.text((1385, y), "Experiment A: Original Scene Pair", fill=C_PRIMARY, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• Detected Keypoints (P2):", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "2000 points", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• Lowe's Ratio Matches:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "220 matches", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• RANSAC Inlier Matches:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "111 inliers", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• Inlier Ratio:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "50.5%", fill=(34, 139, 34), font=FONT_BOLD)

    y += 45
    draw.text((1385, y), "Experiment B: 35 deg Rot + 0.8x Scale", fill=C_ACCENT, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• Detected Keypoints (P2):", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "2000 points", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• Lowe's Ratio Matches:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "219 matches", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• RANSAC Inliers:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "106 inliers", fill=C_TEXT_MAIN, font=FONT_BOLD)
    y += 28
    draw.text((1385, y), "• Inlier Ratio:", fill=C_TEXT_MUTED, font=FONT_BODY)
    draw.text((1660, y), "48.4%", fill=(34, 139, 34), font=FONT_BOLD)

    y += 45
    draw.line([(1385, y), (1850, y)], fill=C_CARD_BORDER, width=1)
    y += 20
    draw.text((1385, y), "Conclusion: SIFT retains high inlier ratio", fill=C_PRIMARY, font=FONT_SMALL)
    y += 22
    draw.text((1385, y), "confirming full Scale & Rotation invariance.", fill=C_PRIMARY, font=FONT_SMALL)

    return img


def create_slide_6() -> Image.Image:
    """Slide 6: Task 3 - Viola-Jones vs YOLOv8."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 6, 8, "Task 3 — Object Recognition: Classical Viola–Jones vs Modern Ultralytics YOLOv8")
    
    draw.rounded_rectangle([(45, 180), (1330, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((65, 200), "Detection Comparison: Input Scene vs Haar Face Cascade vs YOLOv8 Multi-Object", fill=C_PRIMARY, font=FONT_BOLD)
    paste_fitted_image(img, OUTPUTS_DIR / "task3_object_recognition" / "fig1_detection_comparison_montage.png", (65, 235, 1310, HEIGHT - 80))

    draw.rounded_rectangle([(1360, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1385, 205), "Architecture Contrast", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(1385, 245), (1850, 245)], fill=C_CARD_BORDER, width=1)
    
    comparisons = [
        ("Viola-Jones (Haar)", "Ultralytics YOLOv8"),
        ("Haar rectangular features", "Deep ConvNet (CSPDarknet)"),
        ("AdaBoost cascade tree", "Anchor-free decoupled head"),
        ("Frontal human faces only", "80 real-world COCO classes"),
        ("Binary (No confidence)", "Calibrated probability [0, 1]"),
        ("Fails on >20 deg tilt/angle", "High perspective/pose tolerance"),
        ("Result: 2 faces detected", "Result: 3 objects localized"),
    ]
    y = 265
    for left, right in comparisons:
        draw.text((1385, y), left, fill=C_PRIMARY if "Viola" in left else C_TEXT_MUTED, font=FONT_BOLD if "Viola" in left else FONT_SMALL)
        draw.text((1630, y), right, fill=C_ACCENT if "YOLO" in right else C_TEXT_MAIN, font=FONT_BOLD if "YOLO" in right else FONT_SMALL)
        y += 42
        
    y += 10
    draw.line([(1385, y), (1850, y)], fill=C_CARD_BORDER, width=1)
    y += 20
    draw.text((1385, y), "Failure Mode Observed:", fill=C_ACCENT, font=FONT_BOLD)
    y += 26
    draw.text((1385, y), "Haar missed profile angles; YOLO exhibited", fill=C_TEXT_MAIN, font=FONT_SMALL)
    y += 22
    draw.text((1385, y), "lower confidence under partial occlusion.", fill=C_TEXT_MAIN, font=FONT_SMALL)

    return img


def create_slide_7() -> Image.Image:
    """Slide 7: Task 4 - Video Motion Estimation."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 7, 8, "Task 4 — Video Motion Estimation & Optical Flow Vector Fields")
    
    draw.rounded_rectangle([(45, 180), (1330, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((65, 200), "Multi-Paradigm Motion Estimation: Frame Diff, GMM (MOG2), Lucas–Kanade, & Farneback Flow", fill=C_PRIMARY, font=FONT_BOLD)
    paste_fitted_image(img, OUTPUTS_DIR / "task4_motion_estimation" / "fig1_motion_estimation_comparison_grid.png", (65, 235, 1310, HEIGHT - 80))

    draw.rounded_rectangle([(1360, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1385, 205), "Motion Pipeline Results", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(1385, 245), (1850, 245)], fill=C_CARD_BORDER, width=1)
    
    motion_info = [
        ("1. Frame Differencing:", "Detected 6 motion clusters. Very fast, but suffers from aperture/ghosting when objects stop."),
        ("2. GMM MOG2 Subtractor:", "Models pixels as 3-5 Gaussians. Separates clean foreground mask and background model."),
        ("3. Lucas-Kanade Flow:", "Tracked 215 active motion vectors with average velocity of 2.64 px/frame (yellow arrows)."),
        ("4. Farneback Dense Flow:", "Computes continuous pixel field visualized as HSV wheel (Hue=Angle, Value=Velocity)."),
    ]
    y = 265
    for title, desc in motion_info:
        draw.text((1385, y), title, fill=C_ACCENT, font=FONT_BOLD)
        y += 26
        # split desc into 2 lines
        words = desc.split(" ")
        line1 = " ".join(words[:7])
        line2 = " ".join(words[7:])
        draw.text((1385, y), line1, fill=C_TEXT_MAIN, font=FONT_SMALL)
        y += 20
        draw.text((1385, y), line2, fill=C_TEXT_MAIN, font=FONT_SMALL)
        y += 34
        
    y += 10
    draw.line([(1385, y), (1850, y)], fill=C_CARD_BORDER, width=1)
    y += 20
    draw.text((1385, y), "Core Distinction:", fill=C_PRIMARY, font=FONT_BOLD)
    y += 26
    draw.text((1385, y), "Spatial masks indicate presence; Optical Flow", fill=C_TEXT_MAIN, font=FONT_SMALL)
    y += 22
    draw.text((1385, y), "supplies true directional velocity vectors.", fill=C_TEXT_MAIN, font=FONT_SMALL)

    return img


def create_slide_8() -> Image.Image:
    """Slide 8: Conclusion & Summary."""
    img = Image.new("RGB", (WIDTH, HEIGHT), C_BG)
    draw = ImageDraw.Draw(img)
    draw_header_footer(draw, 8, 8, "Project Summary, Verification, & Academic Conclusions")
    
    # Left Card: Summary Table
    draw.rounded_rectangle([(45, 180), (1050, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((75, 205), "Practical Implementation Summary Table", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(75, 245), (1020, 245)], fill=C_CARD_BORDER, width=1)
    
    rows = [
        ("Unit 3: Image Denoising", "Median Filter optimal for S&P (31.31 dB, 0.8949 SSIM)"),
        ("Unit 3: Segmentation", "Otsu threshold + Marker-controlled Watershed boundaries"),
        ("Unit 4: HOG Feature Maps", "489,168 gradient orientation values extracted"),
        ("Unit 4: SIFT Invariance", "220 matches (50.5%) -> 219 matches (48.4%) after 35 deg rot"),
        ("Unit 5: Object Recognition", "Viola-Jones 2 faces vs YOLOv8 3 objects with continuous conf"),
        ("Unit 5: Motion Dynamics", "6 diff clusters, GMM MOG2, 215 LK vectors @ 2.64 px/f"),
    ]
    y = 265
    for left, right in rows:
        draw.text((75, y), left, fill=C_PRIMARY, font=FONT_BOLD)
        y += 25
        draw.text((75, y), right, fill=C_TEXT_MAIN, font=FONT_BODY)
        y += 38

    # Right Card: Repository & Links
    draw.rounded_rectangle([(1085, 180), (1875, HEIGHT - 65)], radius=12, fill=C_CARD_BG, outline=C_CARD_BORDER, width=2)
    draw.text((1115, 205), "Verification & Academic Submission", fill=C_PRIMARY, font=FONT_HEADING)
    draw.line([(1115, 245), (1845, 245)], fill=C_CARD_BORDER, width=1)
    
    y = 275
    draw.text((1115, y), "GitHub Repository:", fill=C_ACCENT, font=FONT_BOLD)
    y += 28
    draw.text((1115, y), "https://github.com/Yash-Ranpariya/cv-", fill=C_PRIMARY, font=FONT_BOLD)
    
    y += 45
    draw.text((1115, y), "Course Mentorship:", fill=C_TEXT_MUTED, font=FONT_BOLD)
    y += 28
    draw.text((1115, y), "Prof. Rahul Inchal", fill=C_TEXT_MAIN, font=FONT_BODY)
    draw.text((1115, y + 25), "rahul.inchal.inchal24@gmail.com", fill=C_TEXT_MUTED, font=FONT_SMALL)
    
    y += 75
    draw.text((1115, y), "Submission Status:", fill=C_TEXT_MUTED, font=FONT_BOLD)
    y += 28
    draw.text((1115, y), "[PASS] 19/19 Unit Tests Passed", fill=(34, 139, 34), font=FONT_BOLD)
    draw.text((1115, y + 25), "[PASS] All 49 Output Figures Generated", fill=(34, 139, 34), font=FONT_BOLD)
    draw.text((1115, y + 50), "[PASS] Academic PDF Report Compiled", fill=(34, 139, 34), font=FONT_BOLD)

    # Big Thank you banner
    draw.rounded_rectangle([(1115, HEIGHT - 180), (1845, HEIGHT - 90)], radius=10, fill=C_PRIMARY)
    draw.text((1290, HEIGHT - 150), "THANK YOU FOR WATCHING!", fill=(255, 255, 255), font=FONT_HEADING)

    return img


# Slides and Speech Scripts
SLIDES_CONFIG = [
    {
        "id": 1,
        "title": "Introduction & Dataset",
        "create_func": create_slide_1,
        "speech": (
            "Hello, respected faculty and evaluators. My name is Yash Ranpariya, PRN 272251066, "
            "studying Master of Computer Applications at MIT World Peace University, Pune. "
            "Today, I am presenting my Continuous Assessment 3 practical coding assignment in Computer Vision, "
            "covering Units 3, 4, and 5, under the guidance of Professor Rahul Inchal. "
            "For this assignment, I captured my own authentic dataset without using stock photos: "
            "First, original_photo.jpg for restoration, feature extraction, and recognition; "
            "Second, second_photo.jpg from an alternate perspective for feature matching; and "
            "Third, original_video.mp4, my 15-second self-recorded video clip for motion estimation. "
            "Let us now examine the implementation and results for each task."
        ),
    },
    {
        "id": 2,
        "title": "Task 1: Image Restoration",
        "create_func": create_slide_2,
        "speech": (
            "Task 1 addresses two problems: image restoration and semantic object segmentation. "
            "In Part A, Image Restoration, I degraded my photo with two noise models: Additive White Gaussian Noise, "
            "simulating electronic sensor noise; and Salt-and-Pepper impulse noise, simulating transmission bit errors. "
            "I applied both a linear 2D Gaussian filter and a non-linear Median filter. "
            "On Gaussian noise, Gaussian filtering achieved 28.82 dB PSNR and 0.7405 SSIM. "
            "However, on Salt-and-Pepper noise, Gaussian filtering smeared impulses into cloudy blotches, yielding only 26.67 dB. "
            "In contrast, the Median filter achieved an outstanding 31.31 dB PSNR and 0.8949 SSIM, because sorting the neighborhood "
            "completely rejects extreme 0 and 255 outliers, recovering crisp object edges."
        ),
    },
    {
        "id": 3,
        "title": "Task 1: Morphology & Watershed",
        "create_func": create_slide_3,
        "speech": (
            "Next in Part B, Morphological Operations and Segmentation, I applied Otsu's thresholding to maximize between-class variance, "
            "followed by morphological erosion, dilation, opening, and closing. "
            "To segment touching regions without oversegmentation, I computed the Euclidean Distance Transform to extract sure foreground seeds "
            "and executed marker-controlled Watershed Segmentation, producing the precise red boundary contours shown on screen."
        ),
    },
    {
        "id": 4,
        "title": "Task 2: HOG Features",
        "create_func": create_slide_4,
        "speech": (
            "Task 2 investigates local gradient and scale-invariant descriptors. "
            "First, Histogram of Oriented Gradients, or HOG, characterizes object appearance by computing 9-bin gradient orientation histograms "
            "within 8 by 8 pixel cells, normalized across 2 by 2 block groups. "
            "HOG extracted a feature vector of 489,168 values from my photo, capturing prominent structural edge contours visible in the inferno visualization."
        ),
    },
    {
        "id": 5,
        "title": "Task 2: SIFT Invariant Matching",
        "create_func": create_slide_5,
        "speech": (
            "Second, Scale-Invariant Feature Transform, or SIFT, detects stable interest points across a Difference-of-Gaussians scale space, "
            "assigns a canonical dominant orientation, and builds a 128-dimensional descriptor for each keypoint. "
            "I detected 2,000 keypoints per photo and matched them using Lowe's Ratio Test with a 0.75 cutoff, backed by RANSAC homography estimation. "
            "On the original scene pair, SIFT achieved 220 good matches with 111 RANSAC inliers, a 50.5 percent inlier ratio. "
            "To rigorously verify invariance, I rotated the second photo by 35 degrees and scaled it down to 0.8x. "
            "Remarkably, SIFT maintained 219 good matches and 106 inliers, achieving a 48.4 percent inlier ratio, experimentally proving scale and rotational invariance."
        ),
    },
    {
        "id": 6,
        "title": "Task 3: Viola-Jones vs YOLOv8",
        "create_func": create_slide_6,
        "speech": (
            "Task 3 contrasts classical geometric detection against deep convolutional neural networks. "
            "Viola-Jones uses rectangular Haar-like features computed in constant time via an Integral Image, selected by AdaBoost in an attentional cascade. "
            "It successfully detected 2 frontal faces. However, it is restricted to frontal human faces with no continuous confidence score, failing when head tilt exceeds 20 degrees. "
            "In contrast, Ultralytics YOLOv8 is an anchor-free single-stage deep CNN with a CSPDarknet backbone and PANet pyramid. "
            "It accurately localized 3 objects across MS COCO classes with calibrated continuous probability scores and high angle tolerance."
        ),
    },
    {
        "id": 7,
        "title": "Task 4: Video Motion Estimation",
        "create_func": create_slide_7,
        "speech": (
            "Task 4 analyzes temporal dynamics in my video using four complementary algorithms. "
            "First, Temporal Frame Differencing: subtracting adjacent frames detected 6 motion clusters. Very fast, but subject to ghosting when motion pauses. "
            "Second, Gaussian Mixture Model Background Subtraction using MOG2: models each background pixel with 3 to 5 Gaussians, isolating moving foreground silhouettes while filtering dynamic shadows. "
            "Third, Lucas-Kanade Sparse Optical Flow: under the brightness constancy assumption, it tracked 215 active motion vectors with an average speed of 2.64 pixels per frame, rendered as yellow arrows. "
            "Fourth, Dense Farneback Flow computes a continuous per-pixel flow field visualized via an HSV color wheel, where hue encodes direction and brightness represents speed."
        ),
    },
    {
        "id": 8,
        "title": "Conclusion & Repository Verification",
        "create_func": create_slide_8,
        "speech": (
            "In conclusion, this project implemented and verified the entire computer vision continuum: restoration, segmentation, invariant feature extraction, "
            "object recognition, and multi-paradigm video motion estimation. "
            "All source code is committed to my GitHub repository at github.com/Yash-Ranpariya/cv-, and the original video recording has been shared on Google Drive with Professor Rahul Inchal. "
            "Thank you very much for your time and guidance."
        ),
    },
]


def render_presentation_video():
    """Build all slides, synthesize audio, and encode Full HD video using ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print("=" * 70)
    print("      CCA 3 — PRESENTATION VIDEO GENERATOR")
    print(f"      Student: Yash Ranpariya (PRN 272251066)")
    print(f"      FFmpeg Engine: {ffmpeg_exe}")
    print("=" * 70)
    
    slide_video_parts = []
    
    for idx, cfg in enumerate(SLIDES_CONFIG):
        slide_num = cfg["id"]
        title = cfg["title"]
        print(f"\n[Processing Slide {slide_num}/8] {title}...")
        
        # 1. Synthesize audio
        wav_path = TEMP_DIR / f"audio_slide_{slide_num}.wav"
        duration = generate_audio(cfg["speech"], wav_path)
        # add 1.0 second pause at the end of each slide for natural pacing
        total_duration = duration + 1.0
        print(f"  -> Generated speech audio: {duration:.2f}s (Total slide display: {total_duration:.2f}s)")
        
        # 2. Render Slide Image
        slide_img = cfg["create_func"]()
        slide_png = TEMP_DIR / f"slide_{slide_num}.png"
        slide_img.save(slide_png, "PNG")
        
        # 3. Encode Slide Video Segment with FFmpeg
        part_mp4 = TEMP_DIR / f"part_{slide_num}.mp4"
        
        # Command to create a video of the static image matching the audio duration
        cmd = [
            ffmpeg_exe, "-y",
            "-loop", "1",
            "-i", str(slide_png),
            "-i", str(wav_path),
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-r", str(FPS),
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(total_duration),
            "-shortest",
            str(part_mp4)
        ]
        
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"  [!] FFmpeg error on slide {slide_num}: {res.stderr.decode('utf-8', errors='ignore')}")
            return
        
        slide_video_parts.append(part_mp4)
        print(f"  -> Encoded segment: {part_mp4.name}")

    # 4. Concatenate all parts into the final video
    print("\n[Concatenating all segments into final presentation video]...")
    concat_list_file = TEMP_DIR / "concat_list.txt"
    with open(concat_list_file, "w") as f:
        for p in slide_video_parts:
            # use forward slashes for ffmpeg concat
            f.write(f"file '{p.as_posix()}'\n")

    concat_cmd = [
        ffmpeg_exe, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(FINAL_VIDEO_PATH)
    ]
    
    res = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode == 0:
        size_mb = FINAL_VIDEO_PATH.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 70)
        print(" SUCCESS! FINAL PRESENTATION VIDEO GENERATED SUCCESSFULLY!")
        print(f" Output Location: {FINAL_VIDEO_PATH}")
        print(f" File Size: {size_mb:.2f} MB")
        print("=" * 70)
    else:
        print("[!] Concatenation error:", res.stderr.decode("utf-8", errors="ignore"))


if __name__ == "__main__":
    render_presentation_video()
