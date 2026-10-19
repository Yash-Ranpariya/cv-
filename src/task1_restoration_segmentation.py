"""Task 1: Image Restoration and Segmentation.
Part of CCA 3 Computer Vision Assignment (Unit 3).

Techniques implemented:
  Part A - Image Restoration:
    1. Gaussian Noise addition
    2. Salt-and-Pepper Noise addition
    3. Gaussian Filtering restoration
    4. Median Filtering restoration
    5. Quantitative comparison: PSNR (dB) and SSIM calculation

  Part B - Morphological Processing and Segmentation:
    1. Grayscale conversion
    2. Thresholding (Otsu & Adaptive) for binarization
    3. Morphological operations: Erosion, Dilation, Opening, Closing
    4. Canny Edge Detection
    5. Watershed Segmentation with Distance Transform markers
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils import (
    ensure_dir,
    load_image,
    save_image,
    compute_psnr,
    compute_ssim,
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    save_grid_figure,
)


def run_image_restoration(
    image: np.ndarray, 
    output_dir: Path
) -> Dict[str, Any]:
    """Execute Part A: Noise injection, filtering, and metric calculation."""
    ensure_dir(output_dir)
    print("\n--- Running Task 1 Part A: Image Restoration ---")
    
    # 1. Add synthetic noises
    noisy_gauss = add_gaussian_noise(image, mean=0.0, sigma=25.0, seed=42)
    noisy_sp = add_salt_and_pepper_noise(image, salt_prob=0.03, pepper_prob=0.03, seed=42)
    
    save_image(output_dir / "01_original.png", image)
    save_image(output_dir / "02_noisy_gaussian.png", noisy_gauss)
    save_image(output_dir / "03_noisy_salt_and_pepper.png", noisy_sp)
    
    # 2. Apply Gaussian Filtering (ksize=5, sigma=1.2)
    restored_gauss_by_gauss = cv2.GaussianBlur(noisy_gauss, (5, 5), sigmaX=1.2)
    restored_sp_by_gauss = cv2.GaussianBlur(noisy_sp, (5, 5), sigmaX=1.2)
    
    # 3. Apply Median Filtering (ksize=5)
    restored_gauss_by_median = cv2.medianBlur(noisy_gauss, 5)
    restored_sp_by_median = cv2.medianBlur(noisy_sp, 5)
    
    save_image(output_dir / "04_gauss_noisy_restored_gaussian_filter.png", restored_gauss_by_gauss)
    save_image(output_dir / "05_gauss_noisy_restored_median_filter.png", restored_gauss_by_median)
    save_image(output_dir / "06_sp_noisy_restored_gaussian_filter.png", restored_sp_by_gauss)
    save_image(output_dir / "07_sp_noisy_restored_median_filter.png", restored_sp_by_median)
    
    # 4. Compute Metrics (PSNR in dB, SSIM)
    metrics = {
        "gaussian_noisy": {
            "psnr": round(compute_psnr(image, noisy_gauss), 2),
            "ssim": round(compute_ssim(image, noisy_gauss), 4),
        },
        "gaussian_noisy_restored_gauss": {
            "psnr": round(compute_psnr(image, restored_gauss_by_gauss), 2),
            "ssim": round(compute_ssim(image, restored_gauss_by_gauss), 4),
        },
        "gaussian_noisy_restored_median": {
            "psnr": round(compute_psnr(image, restored_gauss_by_median), 2),
            "ssim": round(compute_ssim(image, restored_gauss_by_median), 4),
        },
        "sp_noisy": {
            "psnr": round(compute_psnr(image, noisy_sp), 2),
            "ssim": round(compute_ssim(image, noisy_sp), 4),
        },
        "sp_noisy_restored_gauss": {
            "psnr": round(compute_psnr(image, restored_sp_by_gauss), 2),
            "ssim": round(compute_ssim(image, restored_sp_by_gauss), 4),
        },
        "sp_noisy_restored_median": {
            "psnr": round(compute_psnr(image, restored_sp_by_median), 2),
            "ssim": round(compute_ssim(image, restored_sp_by_median), 4),
        },
    }
    
    # Display comparison table
    print("\n[Quantitative Restoration Evaluation Table]")
    print("-" * 65)
    print(f"{'Condition / Filter':<35} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-" * 65)
    print(f"{'Gaussian Noise (Corrupted)':<35} | {metrics['gaussian_noisy']['psnr']:<12} | {metrics['gaussian_noisy']['ssim']:<10}")
    print(f"{'Gaussian Noise + Gaussian Filter':<35} | {metrics['gaussian_noisy_restored_gauss']['psnr']:<12} | {metrics['gaussian_noisy_restored_gauss']['ssim']:<10}")
    print(f"{'Gaussian Noise + Median Filter':<35} | {metrics['gaussian_noisy_restored_median']['psnr']:<12} | {metrics['gaussian_noisy_restored_median']['ssim']:<10}")
    print(f"{'Salt & Pepper Noise (Corrupted)':<35} | {metrics['sp_noisy']['psnr']:<12} | {metrics['sp_noisy']['ssim']:<10}")
    print(f"{'Salt & Pepper + Gaussian Filter':<35} | {metrics['sp_noisy_restored_gauss']['psnr']:<12} | {metrics['sp_noisy_restored_gauss']['ssim']:<10}")
    print(f"{'Salt & Pepper + Median Filter':<35} | {metrics['sp_noisy_restored_median']['psnr']:<12} | {metrics['sp_noisy_restored_median']['ssim']:<10}")
    print("-" * 65)

    # 5. Generate Matplotlib Comparative Visualizations
    # Montage 1: Noise injection
    save_grid_figure(
        images=[image, noisy_gauss, noisy_sp],
        titles=[
            "Original Reference",
            f"Gaussian Noise (PSNR: {metrics['gaussian_noisy']['psnr']} dB)",
            f"Salt & Pepper Noise (PSNR: {metrics['sp_noisy']['psnr']} dB)",
        ],
        save_path=output_dir / "fig1_noise_comparison.png",
        rows=1,
        cols=3,
        figsize=(15, 5),
        main_title="Task 1A: Noise Degradation Comparison",
    )
    
    # Montage 2: Gaussian Noise Restoration Comparison
    save_grid_figure(
        images=[image, noisy_gauss, restored_gauss_by_gauss, restored_gauss_by_median],
        titles=[
            "Original Ground Truth",
            f"Gaussian Noisy ({metrics['gaussian_noisy']['psnr']} dB)",
            f"Gaussian Filter ({metrics['gaussian_noisy_restored_gauss']['psnr']} dB | SSIM {metrics['gaussian_noisy_restored_gauss']['ssim']})",
            f"Median Filter ({metrics['gaussian_noisy_restored_median']['psnr']} dB | SSIM {metrics['gaussian_noisy_restored_median']['ssim']})",
        ],
        save_path=output_dir / "fig2_gaussian_noise_restoration.png",
        rows=1,
        cols=4,
        figsize=(18, 5),
        main_title="Task 1A: Gaussian Noise Restoration Analysis",
    )

    # Montage 3: Salt-and-Pepper Restoration Comparison
    save_grid_figure(
        images=[image, noisy_sp, restored_sp_by_gauss, restored_sp_by_median],
        titles=[
            "Original Ground Truth",
            f"S&P Noisy ({metrics['sp_noisy']['psnr']} dB)",
            f"Gaussian Filter ({metrics['sp_noisy_restored_gauss']['psnr']} dB | SSIM {metrics['sp_noisy_restored_gauss']['ssim']})",
            f"Median Filter ({metrics['sp_noisy_restored_median']['psnr']} dB | SSIM {metrics['sp_noisy_restored_median']['ssim']})",
        ],
        save_path=output_dir / "fig3_sp_noise_restoration.png",
        rows=1,
        cols=4,
        figsize=(18, 5),
        main_title="Task 1A: Salt-and-Pepper Noise Restoration Analysis",
    )
    
    # Comprehensive 2x3 Grid
    save_grid_figure(
        images=[
            noisy_gauss, restored_gauss_by_gauss, restored_gauss_by_median,
            noisy_sp, restored_sp_by_gauss, restored_sp_by_median
        ],
        titles=[
            f"Gaussian Corrupted\nPSNR: {metrics['gaussian_noisy']['psnr']} dB",
            f"Gauss + Gaussian Filter\nPSNR: {metrics['gaussian_noisy_restored_gauss']['psnr']} dB | SSIM: {metrics['gaussian_noisy_restored_gauss']['ssim']}",
            f"Gauss + Median Filter\nPSNR: {metrics['gaussian_noisy_restored_median']['psnr']} dB | SSIM: {metrics['gaussian_noisy_restored_median']['ssim']}",
            f"S&P Corrupted\nPSNR: {metrics['sp_noisy']['psnr']} dB",
            f"S&P + Gaussian Filter\nPSNR: {metrics['sp_noisy_restored_gauss']['psnr']} dB | SSIM: {metrics['sp_noisy_restored_gauss']['ssim']}",
            f"S&P + Median Filter (Optimal)\nPSNR: {metrics['sp_noisy_restored_median']['psnr']} dB | SSIM: {metrics['sp_noisy_restored_median']['ssim']}",
        ],
        save_path=output_dir / "fig4_comprehensive_restoration_grid.png",
        rows=2,
        cols=3,
        figsize=(16, 10),
        main_title="Task 1A: Comprehensive Restoration & Filter Benchmark",
    )
    
    print(f" [DONE] Part A outputs saved to: {output_dir}")
    return metrics


def run_morphological_and_segmentation(
    image: np.ndarray, 
    output_dir: Path
) -> Dict[str, Any]:
    """Execute Part B: Morphological filtering, Canny edge detection, and Watershed."""
    ensure_dir(output_dir)
    print("\n--- Running Task 1 Part B: Morphological Operations & Segmentation ---")
    
    # 1. Grayscale conversion
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    save_image(output_dir / "08_grayscale.png", gray, is_rgb=False)
    
    # 2. Thresholding: Otsu and Adaptive
    # Slight Gaussian blur to reduce high-frequency noise before binarization
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    save_image(output_dir / "09_binary_otsu.png", binary_otsu, is_rgb=False)
    save_image(output_dir / "10_binary_adaptive.png", binary_adaptive, is_rgb=False)
    
    # 3. Morphological Operations (Erosion, Dilation, Opening, Closing)
    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    eroded = cv2.erode(binary_otsu, kernel_3x3, iterations=1)
    dilated = cv2.dilate(binary_otsu, kernel_3x3, iterations=1)
    opened = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel_3x3)
    closed = cv2.morphologyEx(binary_otsu, cv2.MORPH_CLOSE, kernel_3x3)
    
    save_image(output_dir / "11_morph_eroded.png", eroded, is_rgb=False)
    save_image(output_dir / "12_morph_dilated.png", dilated, is_rgb=False)
    save_image(output_dir / "13_morph_opened.png", opened, is_rgb=False)
    save_image(output_dir / "14_morph_closed.png", closed, is_rgb=False)

    # 4. Canny Edge Detection
    # Standard 1:3 lower:upper threshold ratio
    canny_edges_standard = cv2.Canny(blurred, 50, 150)
    canny_edges_tight = cv2.Canny(blurred, 100, 200)
    
    save_image(output_dir / "15_canny_edges_standard.png", canny_edges_standard, is_rgb=False)
    save_image(output_dir / "16_canny_edges_tight.png", canny_edges_tight, is_rgb=False)

    # 5. Watershed Segmentation with Distance Transform Markers
    # Noise removal with opening
    opening = cv2.morphologyEx(binary_otsu, cv2.MORPH_OPEN, kernel_3x3, iterations=2)
    
    # Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel_3x3, iterations=3)
    
    # Finding sure foreground area via Distance Transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Threshold distance transform to get core object peaks
    ret_fg, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown boundary region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that background is not 0, but 1
    markers = markers + 1
    # Mark the unknown region as 0
    markers[unknown == 255] = 0
    
    # Apply Watershed on 3-channel BGR/RGB image
    color_img = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    markers_watershed = cv2.watershed(color_img.copy(), markers)
    
    # Visualize watershed boundaries overlaid in bright red
    watershed_overlay = color_img.copy()
    watershed_overlay[markers_watershed == -1] = [255, 0, 0]  # Red boundary lines
    
    save_image(output_dir / "17_dist_transform.png", (dist_norm * 255).astype(np.uint8), is_rgb=False)
    save_image(output_dir / "18_watershed_markers.png", markers_watershed.astype(np.uint8), is_rgb=False)
    save_image(output_dir / "19_watershed_segmented_overlay.png", watershed_overlay, is_rgb=True)

    # Visualizations
    # Figure: Thresholding Comparison
    save_grid_figure(
        images=[gray, binary_otsu, binary_adaptive],
        titles=[
            "Grayscale Input",
            f"Otsu Binary (Threshold: {int(otsu_val)})",
            "Adaptive Gaussian Threshold",
        ],
        save_path=output_dir / "fig5_thresholding_comparison.png",
        rows=1,
        cols=3,
        figsize=(15, 5),
        main_title="Task 1B: Grayscale Thresholding & Binarization",
    )

    # Figure: Morphological Operations
    save_grid_figure(
        images=[binary_otsu, eroded, dilated, opened, closed],
        titles=[
            "Input Binary (Otsu)",
            "Erosion (Shrinks bright foreground)",
            "Dilation (Expands bright foreground)",
            "Opening (Erosion -> Dilation: Removes noise)",
            "Closing (Dilation -> Erosion: Fills holes)",
        ],
        save_path=output_dir / "fig6_morphological_transforms.png",
        rows=1,
        cols=5,
        figsize=(20, 4.5),
        main_title="Task 1B: Morphological Processing Comparison",
    )

    # Figure: Canny Edge Detection
    save_grid_figure(
        images=[image, gray, canny_edges_standard, canny_edges_tight],
        titles=[
            "Original Color",
            "Grayscale (Smoothed)",
            "Canny Edges (Thresholds: 50, 150)",
            "Canny Edges (Thresholds: 100, 200)",
        ],
        save_path=output_dir / "fig7_canny_edge_detection.png",
        rows=1,
        cols=4,
        figsize=(18, 5),
        main_title="Task 1B: Canny Edge Detection Comparison",
    )

    # Figure: Watershed Segmentation Pipeline
    save_grid_figure(
        images=[
            binary_otsu,
            (dist_norm * 255).astype(np.uint8),
            sure_fg,
            watershed_overlay,
        ],
        titles=[
            "1. Binary Foreground Mask",
            "2. Euclidean Distance Transform",
            "3. Sure Foreground Markers",
            "4. Watershed Boundaries Overlaid (Red)",
        ],
        save_path=output_dir / "fig8_watershed_segmentation_pipeline.png",
        rows=1,
        cols=4,
        figsize=(18, 5),
        main_title="Task 1B: Marker-Controlled Watershed Segmentation Pipeline",
    )

    results = {
        "otsu_threshold": int(otsu_val),
        "detected_marker_regions": int(num_markers) - 1,
    }
    print(f" [DONE] Part B outputs saved to: {output_dir}")
    return results


def run_task1(
    image_path: Optional[Path] = None, 
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Execute complete Task 1: Restoration and Segmentation."""
    project_root = Path(__file__).resolve().parent.parent
    if image_path is None:
        image_path = project_root / "dataset" / "original_photo.jpg"
    if output_dir is None:
        output_dir = project_root / "outputs" / "task1_restoration_segmentation"
        
    print("=" * 70)
    print("      TASK 1: IMAGE RESTORATION AND SEGMENTATION EXECUTION")
    print("=" * 70)
    print(f"Input image : {image_path}")
    print(f"Output path : {output_dir}")
    
    if not image_path.exists():
        msg = (
            f"ERROR: Dataset image not found at '{image_path}'.\n"
            "Please place your personal photo at 'dataset/original_photo.jpg'.\n"
            "Alternatively, if you have 'dataset/original_video.mp4', a frame can be extracted."
        )
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    image = load_image(image_path)
    if image is None:
        msg = f"ERROR: Could not decode image at '{image_path}'."
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    # If image is unusually large (> 1920 on longest side), resize proportionally for faster academic compute
    h, w = image.shape[:2]
    if max(h, w) > 1920:
        scale = 1920.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[*] Resized high-res image from ({w}x{h}) to ({new_w}x{new_h}) for compute efficiency.")

    # Execute Part A
    restoration_metrics = run_image_restoration(image, output_dir)
    
    # Execute Part B
    segmentation_metrics = run_morphological_and_segmentation(image, output_dir)
    
    print("\n" + "=" * 70)
    print(" Task 1 completed successfully! Check outputs/task1_restoration_segmentation/")
    print("=" * 70)
    
    return {
        "status": "success",
        "restoration": restoration_metrics,
        "segmentation": segmentation_metrics,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    run_task1()
