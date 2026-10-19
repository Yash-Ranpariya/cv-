"""Task 2: Feature Extraction and Image Matching.
Part of CCA 3 Computer Vision Assignment (Unit 4).

Techniques implemented:
  1. Histogram of Oriented Gradients (HOG) extraction & gradient map visualization
  2. Scale-Invariant Feature Transform (SIFT) keypoint detection & descriptor calculation
  3. Feature matching between two photographs using Lowe's Ratio Test and RANSAC
  4. Geometric transformation experiment (Rotation & Scaling)
  5. Quantitative matching quality comparison (Keypoints, Good Matches, Inlier Ratio)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage import exposure

from src.utils import (
    ensure_dir,
    load_image,
    save_image,
    save_grid_figure,
)


def extract_and_visualize_hog(
    image: np.ndarray, 
    output_dir: Path
) -> Dict[str, Any]:
    """Extract Histogram of Oriented Gradients (HOG) features and save visualization."""
    ensure_dir(output_dir)
    print("\n--- Running Task 2: HOG Feature Extraction ---")
    
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    # Resize to standard aspect if huge for computational efficiency
    h, w = gray.shape
    target_w = min(w, 640)
    target_h = int(h * (target_w / w))
    gray_resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Compute HOG descriptor and visual representation
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    
    fd, hog_image = hog(
        gray_resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        channel_axis=None,
    )
    
    # Rescale intensity of HOG image for optimal contrast visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_visual_uint8 = (hog_image_rescaled * 255.0).clip(0, 255).astype(np.uint8)

    save_image(output_dir / "01_hog_feature_map.png", hog_visual_uint8, is_rgb=False)

    # Plot figure: Original, Grayscale, and HOG Features
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(image if image.ndim == 3 else gray, cmap=None if image.ndim == 3 else "gray")
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gray_resized, cmap="gray")
    axes[1].set_title(f"Processed Grayscale ({target_w}x{target_h})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(hog_image_rescaled, cmap="inferno")
    axes[2].set_title(f"HOG Gradient Orientations\n(Vector length: {len(fd):,} features)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(output_dir / "fig1_hog_visualization.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f" [DONE] HOG vector extracted with {len(fd):,} features.")
    return {
        "descriptor_length": int(len(fd)),
        "orientations": orientations,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block,
    }


def detect_sift_keypoints(
    image: np.ndarray
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray], np.ndarray]:
    """Detect SIFT keypoints and compute 128-dimensional descriptors."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw rich keypoints with scale and orientation circles
    keypoints_img = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )
    return keypoints, descriptors, keypoints_img


def match_sift_features(
    kp1: List[cv2.KeyPoint],
    desc1: Optional[np.ndarray],
    kp2: List[cv2.KeyPoint],
    desc2: Optional[np.ndarray],
    img1: np.ndarray,
    img2: np.ndarray,
    ratio_thresh: float = 0.75,
) -> Tuple[List[cv2.DMatch], List[cv2.DMatch], Optional[np.ndarray], np.ndarray]:
    """Match SIFT descriptors between two images using Lowe's Ratio Test and RANSAC."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        empty_img = np.zeros((100, 200, 3), dtype=np.uint8)
        return [], [], None, empty_img

    # Use FLANN or BFMatcher with L2 norm for SIFT descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test: discard matches where second best is too close in distance
    good_matches: List[cv2.DMatch] = []
    for m_pair in matches_knn:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    inlier_matches: List[cv2.DMatch] = []
    homography_matrix = None
    inlier_mask = None

    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_mask = mask.ravel().tolist()
            inlier_matches = [m for m, is_inlier in zip(good_matches, inlier_mask) if is_inlier == 1]
    
    # Draw matched features
    if inlier_mask is not None and len(inlier_matches) > 0:
        matches_vis = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=inlier_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    else:
        matches_vis = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches[:50], None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    return good_matches, inlier_matches, homography_matrix, matches_vis


def apply_geometric_transform(
    image: np.ndarray, 
    angle_degrees: float = 35.0, 
    scale_factor: float = 0.8
) -> np.ndarray:
    """Rotate and resize an image to test SIFT invariance properties."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, scale_factor)
    
    # Calculate bounding box to avoid clipping
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    
    transformed = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return transformed


def run_feature_extraction_and_matching(
    image1: np.ndarray,
    image2: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute complete Task 2 pipeline and comparative transformation benchmark."""
    ensure_dir(output_dir)
    print("\n--- Running Task 2: SIFT Keypoints, Descriptors & Matching ---")

    # 1. Extract HOG on primary photograph
    hog_metrics = extract_and_visualize_hog(image1, output_dir)

    # 2. SIFT Keypoints and Descriptors
    kp1, desc1, kp_img1 = detect_sift_keypoints(image1)
    kp2, desc2, kp_img2 = detect_sift_keypoints(image2)

    save_image(output_dir / "02_sift_keypoints_photo1.png", kp_img1)
    save_image(output_dir / "03_sift_keypoints_photo2.png", kp_img2)

    # Save visual comparison of keypoints
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(kp_img1)
    axes[0].set_title(f"Photo 1: SIFT Keypoints ({len(kp1):,} detected)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(kp_img2)
    axes[1].set_title(f"Photo 2: SIFT Keypoints ({len(kp2):,} detected)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(str(output_dir / "fig2_sift_keypoints_detected.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Match SIFT features between Original Photo 1 and Photo 2
    good_matches_orig, inliers_orig, H_orig, match_vis_orig = match_sift_features(
        kp1, desc1, kp2, desc2, image1, image2, ratio_thresh=0.75
    )
    save_image(output_dir / "04_sift_matches_original.png", match_vis_orig)

    # 4. Apply Rotation & Resizing Transformation to Photo 2
    transform_angle = 35.0
    transform_scale = 0.8
    image2_transformed = apply_geometric_transform(image2, angle_degrees=transform_angle, scale_factor=transform_scale)
    save_image(output_dir / "05_photo2_transformed.png", image2_transformed)

    # Detect SIFT on transformed image
    kp2_trans, desc2_trans, kp_img2_trans = detect_sift_keypoints(image2_transformed)

    # Match SIFT features after transformation
    good_matches_trans, inliers_trans, H_trans, match_vis_trans = match_sift_features(
        kp1, desc1, kp2_trans, desc2_trans, image1, image2_transformed, ratio_thresh=0.75
    )
    save_image(output_dir / "06_sift_matches_transformed.png", match_vis_trans)

    # 5. Quality metrics comparison
    inlier_ratio_orig = (len(inliers_orig) / len(good_matches_orig) * 100) if good_matches_orig else 0.0
    inlier_ratio_trans = (len(inliers_trans) / len(good_matches_trans) * 100) if good_matches_trans else 0.0

    print("\n[Feature Matching Evaluation Table]")
    print("-" * 75)
    print(f"{'Experiment':<30} | {'Keypoints (P2)':<15} | {'Good Matches':<13} | {'RANSAC Inliers':<14} | {'Inlier %'}")
    print("-" * 75)
    print(f"{'Original Scene Pair':<30} | {len(kp2):<15} | {len(good_matches_orig):<13} | {len(inliers_orig):<14} | {inlier_ratio_orig:.1f}%")
    print(f"{'Rotated (35 deg) & Scaled (0.8x)':<30} | {len(kp2_trans):<15} | {len(good_matches_trans):<13} | {len(inliers_trans):<14} | {inlier_ratio_trans:.1f}%")
    print("-" * 75)

    # 6. Save Side-by-Side Comparison Figure
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    axes[0].imshow(match_vis_orig)
    axes[0].set_title(
        f"Experiment A: SIFT Matching on Original Scene Pair\n"
        f"Lowe's Matches: {len(good_matches_orig)} | RANSAC Inliers: {len(inliers_orig)} ({inlier_ratio_orig:.1f}%)",
        fontsize=12, fontweight="bold", pad=8
    )
    axes[0].axis("off")

    axes[1].imshow(match_vis_trans)
    axes[1].set_title(
        f"Experiment B: SIFT Matching After Geometric Transformation (Rotated {transform_angle}° + Scaled {transform_scale}x)\n"
        f"Lowe's Matches: {len(good_matches_trans)} | RANSAC Inliers: {len(inliers_trans)} ({inlier_ratio_trans:.1f}%)",
        fontsize=12, fontweight="bold", pad=8
    )
    axes[1].axis("off")

    fig.suptitle("Task 2: SIFT Feature Matching Invariance Benchmark (Scale & Rotation)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(output_dir / "fig3_matching_comparison_summary.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "success",
        "hog": hog_metrics,
        "keypoints_photo1": len(kp1),
        "keypoints_photo2": len(kp2),
        "original_matching": {
            "good_matches": len(good_matches_orig),
            "inliers": len(inliers_orig),
            "inlier_ratio": round(inlier_ratio_orig, 2),
        },
        "transformed_matching": {
            "transform_angle": transform_angle,
            "transform_scale": transform_scale,
            "keypoints_photo2_trans": len(kp2_trans),
            "good_matches": len(good_matches_trans),
            "inliers": len(inliers_trans),
            "inlier_ratio": round(inlier_ratio_trans, 2),
        },
        "output_dir": str(output_dir),
    }


def run_task2(
    photo1_path: Optional[Path] = None,
    photo2_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete Task 2 runner with dataset validation."""
    project_root = Path(__file__).resolve().parent.parent
    if photo1_path is None:
        photo1_path = project_root / "dataset" / "original_photo.jpg"
    if photo2_path is None:
        photo2_path = project_root / "dataset" / "second_photo.jpg"
    if output_dir is None:
        output_dir = project_root / "outputs" / "task2_feature_extraction"

    print("=" * 70)
    print("      TASK 2: FEATURE EXTRACTION AND IMAGE MATCHING")
    print("=" * 70)
    print(f"Photo 1     : {photo1_path}")
    print(f"Photo 2     : {photo2_path}")
    print(f"Output path : {output_dir}")

    if not photo1_path.exists() or not photo2_path.exists():
        msg = (
            "ERROR: Both 'dataset/original_photo.jpg' and 'dataset/second_photo.jpg' are required.\n"
            "Please ensure both photographs of the scene are present in the 'dataset/' directory."
        )
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    img1 = load_image(photo1_path)
    img2 = load_image(photo2_path)
    if img1 is None or img2 is None:
        msg = "ERROR: Failed to decode one or both input photos."
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    # Proportional downscale if images are excessively large
    for idx, (img, name) in enumerate([(img1, "photo1"), (img2, "photo2")]):
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            scale = 1600.0 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if idx == 0:
                img1 = resized
            else:
                img2 = resized
            print(f"[*] Resized {name} to ({new_w}x{new_h}) for compute efficiency.")

    results = run_feature_extraction_and_matching(img1, img2, output_dir)
    print("\n" + "=" * 70)
    print(" Task 2 completed successfully! Check outputs/task2_feature_extraction/")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_task2()
