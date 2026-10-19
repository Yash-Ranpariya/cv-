"""Dataset validation module for CCA 3 Computer Vision Project.
Validates the presence, dimensions, formats, and integrity of:
  1. dataset/original_photo.jpg
  2. dataset/second_photo.jpg
  3. dataset/original_video.mp4
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2


def get_dataset_dir(base_dir: Optional[Path] = None) -> Path:
    """Return the dataset directory relative to project root."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "dataset"


def extract_frame_from_video(
    video_path: Path, 
    output_path: Path, 
    frame_number: int = 15
) -> bool:
    """Extract a single frame from the video to serve as a photograph.
    
    This fulfills the assignment option:
    'Using your own photograph or a suitable frame extracted from your video'
    """
    if not video_path.exists():
        return False
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(output_path), frame)
    return False


def validate_dataset(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Validate presence and properties of required input files.
    
    Returns a dictionary of statuses, specs, and guidance for each file.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
        
    dataset_dir = project_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    orig_photo_path = dataset_dir / "original_photo.jpg"
    second_photo_path = dataset_dir / "second_photo.jpg"
    orig_video_path = dataset_dir / "original_video.mp4"
    
    results: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "original_photo": {"path": orig_photo_path, "exists": False, "details": None, "error": None},
        "second_photo": {"path": second_photo_path, "exists": False, "details": None, "error": None},
        "original_video": {"path": orig_video_path, "exists": False, "details": None, "error": None},
        "all_valid": False,
        "ready_tasks": []
    }
    
    # 1. Validate original_photo.jpg
    if orig_photo_path.exists():
        img = cv2.imread(str(orig_photo_path))
        if img is not None:
            h, w, c = img.shape
            size_mb = orig_photo_path.stat().st_size / (1024 * 1024)
            results["original_photo"]["exists"] = True
            results["original_photo"]["details"] = {
                "resolution": f"{w}x{h}",
                "channels": c,
                "size_mb": round(size_mb, 2)
            }
        else:
            results["original_photo"]["error"] = "File exists but cannot be decoded by OpenCV (corrupt image)."
    else:
        results["original_photo"]["error"] = (
            f"Missing file: {orig_photo_path.name}. "
            "Please copy your personal photograph into the 'dataset' directory."
        )

    # 2. Validate second_photo.jpg
    if second_photo_path.exists():
        img2 = cv2.imread(str(second_photo_path))
        if img2 is not None:
            h, w, c = img2.shape
            size_mb = second_photo_path.stat().st_size / (1024 * 1024)
            results["second_photo"]["exists"] = True
            results["second_photo"]["details"] = {
                "resolution": f"{w}x{h}",
                "channels": c,
                "size_mb": round(size_mb, 2)
            }
        else:
            results["second_photo"]["error"] = "File exists but cannot be decoded by OpenCV (corrupt image)."
    else:
        results["second_photo"]["error"] = (
            f"Missing file: {second_photo_path.name}. "
            "Please capture a second photo of the same scene/object for Task 2."
        )

    # 3. Validate original_video.mp4
    if orig_video_path.exists():
        cap = cv2.VideoCapture(str(orig_video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = round(total_frames / fps, 2) if fps > 0 else 0
            size_mb = orig_video_path.stat().st_size / (1024 * 1024)
            cap.release()
            
            results["original_video"]["exists"] = True
            results["original_video"]["details"] = {
                "resolution": f"{width}x{height}",
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": duration,
                "size_mb": round(size_mb, 2)
            }
        else:
            results["original_video"]["error"] = "File exists but cannot be opened by cv2.VideoCapture."
    else:
        results["original_video"]["error"] = (
            f"Missing file: {orig_video_path.name}. "
            "Please record a 15-30 second personal video and place it in 'dataset'."
        )

    # Assess which tasks are ready
    if results["original_photo"]["exists"]:
        results["ready_tasks"].append("Task 1 (Restoration & Segmentation)")
        results["ready_tasks"].append("Task 3 (Object Recognition - Photo)")
    elif results["original_video"]["exists"]:
        results["ready_tasks"].append("Task 1 (Restoration - via extracted video frame)")
        
    if results["original_photo"]["exists"] and results["second_photo"]["exists"]:
        results["ready_tasks"].append("Task 2 (Feature Extraction & Matching)")
        
    if results["original_video"]["exists"]:
        results["ready_tasks"].append("Task 4 (Video Motion Estimation)")
        
    results["all_valid"] = (
        results["original_photo"]["exists"]
        and results["second_photo"]["exists"]
        and results["original_video"]["exists"]
    )
    
    return results


def print_dataset_status(results: Optional[Dict[str, Any]] = None) -> None:
    """Print an MCA student-friendly summary of dataset status."""
    if results is None:
        results = validate_dataset()
        
    print("=" * 70)
    print("         CCA 3 COMPUTER VISION - DATASET VERIFICATION")
    print("=" * 70)
    print(f"Directory: {results['dataset_dir']}\n")

    # Photo 1
    p1 = results["original_photo"]
    if p1["exists"]:
        d = p1["details"]
        print(f" [PASS] original_photo.jpg : {d['resolution']}, {d['size_mb']} MB, {d['channels']} channels")
    else:
        print(f" [MISSING] original_photo.jpg : {p1['error']}")

    # Photo 2
    p2 = results["second_photo"]
    if p2["exists"]:
        d = p2["details"]
        print(f" [PASS] second_photo.jpg   : {d['resolution']}, {d['size_mb']} MB, {d['channels']} channels")
    else:
        print(f" [MISSING] second_photo.jpg   : {p2['error']}")

    # Video
    v = results["original_video"]
    if v["exists"]:
        d = v["details"]
        print(f" [PASS] original_video.mp4 : {d['resolution']} @ {d['fps']} FPS, {d['duration_seconds']}s, {d['size_mb']} MB")
    else:
        print(f" [MISSING] original_video.mp4 : {v['error']}")

    print("-" * 70)
    if results["all_valid"]:
        print("ALL REQUIRED PERSONAL DATASET FILES DETECTED! Ready to run all tasks.")
    else:
        print("ACTION REQUIRED:")
        print("  1. Capture two photos of the same object/scene (e.g. your desk, laptop, books).")
        print("     Save them as 'original_photo.jpg' and 'second_photo.jpg' inside 'dataset/'.")
        print("  2. Record a 15-30s video of yourself or moving objects.")
        print("     Save it as 'original_video.mp4' inside 'dataset/'.")
    print("=" * 70)


if __name__ == "__main__":
    print_dataset_status()
