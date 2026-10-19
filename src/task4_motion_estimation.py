"""Task 4: Video Motion Estimation.
Part of CCA 3 Computer Vision Assignment (Unit 5).

Techniques implemented:
  1. Consecutive frame extraction and validation from video
  2. Temporal Frame Differencing motion detection
  3. Gaussian Mixture Model (GMM) Background Subtraction (MOG2)
  4. Lucas–Kanade (LK) Optical Flow motion vector field estimation
  5. Dense Optical Flow (Farneback) magnitude and angle visualization (HSV)
  6. Multi-algorithm comparative benchmark and evaluation table
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils import (
    ensure_dir,
    load_image,
    save_image,
    save_grid_figure,
)


def extract_sample_frames(
    video_path: Path, 
    start_frame: int = 15, 
    num_frames: int = 5,
    stride: int = 2
) -> Tuple[List[np.ndarray], float]:
    """Safely extract consecutive frames from an MP4 video file.
    
    Guarantees video resources are properly opened and released.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Bound start frame
    if start_frame >= total_frames - (num_frames * stride):
        start_frame = max(0, total_frames - (num_frames * stride) - 1)

    frames: List[np.ndarray] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for i in range(num_frames):
            for _ in range(stride - 1):
                cap.grab()
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
    finally:
        cap.release()

    if len(frames) < 2:
        raise ValueError(f"Insufficient frames extracted from video ({len(frames)} frames).")

    return frames, fps


def compute_frame_differencing(
    frame1: np.ndarray, 
    frame2: np.ndarray, 
    threshold: int = 30
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Detect motion regions via consecutive temporal frame subtraction.
    
    Formula: D(x, y, t) = |I(x, y, t) - I(x, y, t-1)|
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if frame1.ndim == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if frame2.ndim == 3 else frame2

    # Smooth to suppress high-frequency sensor noise
    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Absolute difference
    diff = cv2.absdiff(blur1, blur2)
    
    # Threshold to binary motion mask
    _, thresh_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological opening and dilation to remove speckles and join components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)

    # Overlay bounding boxes on current frame
    annotated = frame2.copy()
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_regions = 0
    for c in contours:
        if cv2.contourArea(c) > 150:  # Ignore tiny noise artifacts
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 69, 0), 2)  # Red-Orange box
            motion_regions += 1

    return cleaned_mask, annotated, motion_regions


def compute_gmm_background_subtraction(
    frames: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment moving foreground using Gaussian Mixture Model (MOG2).
    
    Returns:
        (Binary foreground mask, learned background model, annotated frame)
    """
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=50,
        varThreshold=25,
        detectShadows=True,
    )

    fg_mask = None
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.ndim == 3 else f
        fg_mask = subtractor.apply(bgr)

    # Background image model
    bg_model = subtractor.getBackgroundImage()
    bg_model_rgb = cv2.cvtColor(bg_model, cv2.COLOR_BGR2RGB) if bg_model is not None else frames[0]

    # In MOG2 with shadows: 255 = foreground, 127 = shadow, 0 = background
    # Keep only true foreground (remove shadows for clean segmentation)
    _, clean_fg = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Morphological noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_fg = cv2.morphologyEx(clean_fg, cv2.MORPH_OPEN, kernel)
    clean_fg = cv2.dilate(clean_fg, kernel, iterations=2)

    # Annotate bounding boxes on final frame
    annotated = frames[-1].copy()
    contours, _ = cv2.findContours(clean_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 200:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (50, 205, 50), 2)  # Lime green

    return clean_fg, bg_model_rgb, annotated


def compute_lucas_kanade_optical_flow(
    frame1: np.ndarray,
    frame2: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    """Estimate sparse motion vectors between two frames using Lucas-Kanade Optical Flow.
    
    Returns:
        (Annotated image with motion vectors, number of tracked vectors, average velocity)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if frame1.ndim == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if frame2.ndim == 3 else frame2

    # Shi-Tomasi Corner Detection for reliable trackable features
    feature_params = dict(
        maxCorners=250,
        qualityLevel=0.15,
        minDistance=10,
        blockSize=7,
    )
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    vis = frame2.copy()
    if p0 is None or len(p0) == 0:
        return vis, 0, 0.0

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(19, 19),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    p1, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Select good points
    good_curr = p1[status == 1]
    good_prev = p0[status == 1]

    magnitudes: List[float] = []
    vector_count = 0

    for curr, prev in zip(good_curr, good_prev):
        x1, y1 = curr.ravel().astype(int)
        x0, y0 = prev.ravel().astype(int)
        dx = x1 - x0
        dy = y1 - y0
        mag = np.hypot(dx, dy)
        
        # Only render vectors with noticeable movement (> 0.5 pixels) to filter stationary noise
        if mag > 0.5:
            magnitudes.append(mag)
            vector_count += 1
            # Draw motion vector arrow (Yellow shaft, red tip)
            cv2.arrowedLine(vis, (x0, y0), (x1, y1), (255, 255, 0), 2, tipLength=0.3)
            cv2.circle(vis, (x1, y1), 3, (255, 0, 0), -1)

    avg_speed = float(np.mean(magnitudes)) if magnitudes else 0.0
    return vis, vector_count, round(avg_speed, 2)


def compute_dense_optical_flow_hsv(
    frame1: np.ndarray,
    frame2: np.ndarray,
) -> np.ndarray:
    """Compute dense optical flow using Farneback algorithm and render HSV color wheel.
    
    Hue encodes motion direction (0-360 deg); Value encodes motion velocity/magnitude.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY) if frame1.ndim == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) if frame2.ndim == 3 else frame2

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue = angle in [0, 180]
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value = magnitude

    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_flow


def run_motion_estimation_pipeline(
    frames: List[np.ndarray],
    fps: float,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute complete Task 4 comparative motion analysis on extracted frames."""
    ensure_dir(output_dir)
    print("\n--- Running Task 4: Video Motion Estimation Pipeline ---")

    f_prev = frames[0]
    f_curr = frames[1]

    save_image(output_dir / "01_frame_previous.png", f_prev)
    save_image(output_dir / "02_frame_current.png", f_curr)

    # 1. Frame Differencing
    diff_mask, diff_annotated, diff_regions = compute_frame_differencing(f_prev, f_curr)
    save_image(output_dir / "03_frame_difference_mask.png", diff_mask, is_rgb=False)
    save_image(output_dir / "04_frame_difference_annotated.png", diff_annotated)
    print(f" [DONE] Frame Differencing detected {diff_regions} candidate motion cluster(s).")

    # 2. GMM Background Subtraction
    gmm_mask, bg_model, gmm_annotated = compute_gmm_background_subtraction(frames)
    save_image(output_dir / "05_gmm_foreground_mask.png", gmm_mask, is_rgb=False)
    save_image(output_dir / "06_gmm_learned_background.png", bg_model)
    save_image(output_dir / "07_gmm_annotated.png", gmm_annotated)
    print(" [DONE] GMM MOG2 background subtractor separated foreground mask and background model.")

    # 3. Lucas–Kanade Optical Flow (Sparse Motion Vectors)
    lk_vis, lk_vectors, lk_speed = compute_lucas_kanade_optical_flow(f_prev, f_curr)
    save_image(output_dir / "08_lucas_kanade_motion_vectors.png", lk_vis)
    print(f" [DONE] Lucas-Kanade Optical Flow tracked {lk_vectors} motion vector(s) (Avg speed: {lk_speed} px/frame).")

    # 4. Dense Optical Flow (Farneback HSV Field)
    dense_hsv = compute_dense_optical_flow_hsv(f_prev, f_curr)
    save_image(output_dir / "09_dense_optical_flow_hsv.png", dense_hsv)

    # 5. Save Comprehensive Comparison Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    axes[0, 0].imshow(f_prev)
    axes[0, 0].set_title("1. Previous Frame (t-1)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(f_curr)
    axes[0, 1].set_title("2. Current Frame (t)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(diff_mask, cmap="gray")
    axes[0, 2].set_title(f"3. Frame Differencing Mask\n(Regions: {diff_regions})", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(gmm_mask, cmap="gray")
    axes[1, 0].set_title("4. GMM (MOG2) Foreground Mask\n(Probabilistic Mixture Model)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(lk_vis)
    axes[1, 1].set_title(f"5. Lucas–Kanade Motion Vectors\n({lk_vectors} vectors | {lk_speed} px/f)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(dense_hsv)
    axes[1, 2].set_title("6. Dense Flow Field (HSV)\n(Hue=Angle, Value=Velocity)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")

    fig.suptitle("Task 4: Video Motion Estimation Comparative Benchmark", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(output_dir / "fig1_motion_estimation_comparison_grid.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 6. Display Comparison Table
    print("\n[Motion Estimation Algorithms Comparison Table]")
    print("-" * 85)
    print(f"{'Method':<24} | {'Output Type':<20} | {'Velocity Vector':<16} | {'Stopped Objects'}")
    print("-" * 85)
    print(f"{'Frame Differencing':<24} | {'Binary Pixel Mask':<20} | {'No (Magnitude only)':<16} | {'Lost immediately (Ghost)'}")
    print(f"{'GMM Background (MOG2)':<24} | {'Probabilistic Mask':<20} | {'No':<16} | {'Maintained in history'}")
    print(f"{'Lucas–Kanade (LK)':<24} | {'Sparse Vectors (dx, dy)':<20} | {'Yes (Direction+Speed)':<16} | {'Vectors drop to zero'}")
    print(f"{'Dense Flow (Farneback)':<24} | {'Continuous Flow Map':<20} | {'Yes (Dense 2D grid)':<16} | {'Near-zero vector field'}")
    print("-" * 85)

    return {
        "status": "success",
        "fps": fps,
        "frame_differencing": {"motion_clusters": diff_regions},
        "lucas_kanade": {
            "active_vectors": lk_vectors,
            "average_speed_px_per_frame": lk_speed,
        },
        "output_dir": str(output_dir),
    }


def run_task4(
    video_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    start_frame: int = 15,
) -> Dict[str, Any]:
    """Execute complete Task 4 runner with validation and graceful degradation."""
    project_root = Path(__file__).resolve().parent.parent
    if video_path is None:
        video_path = project_root / "dataset" / "original_video.mp4"
    if output_dir is None:
        output_dir = project_root / "outputs" / "task4_motion_estimation"

    print("=" * 70)
    print("      TASK 4: VIDEO MOTION ESTIMATION")
    print("=" * 70)
    print(f"Video path  : {video_path}")
    print(f"Output path : {output_dir}")

    if not video_path.exists():
        msg = (
            f"ERROR: Video file not found at '{video_path}'.\n"
            "Please record a 15–30 second personal video and place it at 'dataset/original_video.mp4'."
        )
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    try:
        frames, fps = extract_sample_frames(video_path, start_frame=start_frame, num_frames=6, stride=2)
    except Exception as e:
        msg = f"ERROR reading video '{video_path}': {e}"
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    # Proportional downscale if video is 4K
    processed_frames = []
    for f in frames:
        h, w = f.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            f_res = cv2.resize(f, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            processed_frames.append(f_res)
        else:
            processed_frames.append(f)

    results = run_motion_estimation_pipeline(processed_frames, fps, output_dir)
    print("\n" + "=" * 70)
    print(" Task 4 completed successfully! Check outputs/task4_motion_estimation/")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_task4()
