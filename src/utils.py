"""Utility functions for CCA 3 Computer Vision Assignment.
Provides image I/O, noise injection, metric computation (PSNR/SSIM),
and publication-quality Matplotlib visualizer helpers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensure a directory exists and return it as a Path object."""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_image(image_path: Union[str, Path], as_gray: bool = False) -> Optional[np.ndarray]:
    """Load an image from disk using OpenCV, returning RGB or Grayscale numpy array.
    
    Args:
        image_path: Path to the image file.
        as_gray: If True, return single-channel grayscale (uint8).
        
    Returns:
        np.ndarray image in RGB (H, W, 3) or Grayscale (H, W), or None on failure.
    """
    path_str = str(image_path)
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    bgr_or_gray = cv2.imread(path_str, flag)
    
    if bgr_or_gray is None:
        return None
        
    if as_gray:
        return bgr_or_gray
    return cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2RGB)


def save_image(save_path: Union[str, Path], image: np.ndarray, is_rgb: bool = True) -> bool:
    """Save a numpy image array to disk safely.
    
    Args:
        save_path: Target path to save the image.
        image: Image array (RGB, Grayscale, or binary).
        is_rgb: Set to True if image is RGB, so it gets converted to BGR for OpenCV.
    """
    p = Path(save_path)
    ensure_dir(p.parent)
    
    # Ensure correct dtype
    if image.dtype != np.uint8:
        if np.max(image) <= 1.0:
            to_save = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            to_save = image.clip(0, 255).astype(np.uint8)
    else:
        to_save = image

    if is_rgb and len(to_save.shape) == 3 and to_save.shape[2] == 3:
        bgr = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)
    else:
        bgr = to_save
        
    return bool(cv2.imwrite(str(p), bgr))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    PSNR = 10 * log10(MAX_I^2 / MSE)
    Higher values denote closer resemblance to the ground truth.
    """
    if original.shape != processed.shape:
        # Resize processed to match original if needed
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)
    mse = np.mean((orig_f - proc_f) ** 2)
    
    if mse == 0:
        return float("inf")
        
    max_val = 255.0
    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """Calculate Structural Similarity Index Measure (SSIM).
    
    Measures luminance, contrast, and structural degradation.
    Scale ranges from -1 to +1 (+1 indicates perfect fidelity).
    """
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    if original.ndim == 3:
        # For scikit-image >= 0.19, channel_axis is preferred
        val = ssim_fn(original, processed, channel_axis=2, data_range=255)
    else:
        val = ssim_fn(original, processed, data_range=255)
        
    return float(val)


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0, seed: int = 42) -> np.ndarray:
    """Add reproducible zero-mean additive white Gaussian noise to an image."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray, 
    salt_prob: float = 0.02, 
    pepper_prob: float = 0.02, 
    seed: int = 42
) -> np.ndarray:
    """Add reproducible impulse (salt-and-pepper) noise to an image."""
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    
    # Generate random matrix
    rand_matrix = rng.random(image.shape[:2])
    
    # Salt noise (white pixels = 255)
    salt_mask = rand_matrix < salt_prob
    noisy[salt_mask] = 255
    
    # Pepper noise (black pixels = 0)
    pepper_mask = (rand_matrix >= salt_prob) & (rand_matrix < (salt_prob + pepper_prob))
    noisy[pepper_mask] = 0
    
    return noisy


def save_grid_figure(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Union[str, Path],
    rows: int,
    cols: int,
    figsize: Tuple[int, int] = (15, 10),
    main_title: Optional[str] = None,
    cmaps: Optional[List[Optional[str]]] = None
) -> None:
    """Generate and save a publication-quality subplots comparison figure."""
    p = Path(save_path)
    ensure_dir(p.parent)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows * cols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx < len(images):
            img = images[idx]
            title = titles[idx] if idx < len(titles) else f"Image {idx+1}"
            cmap = cmaps[idx] if (cmaps and idx < len(cmaps)) else None
            
            if img.ndim == 2 and cmap is None:
                cmap = "gray"
                
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
