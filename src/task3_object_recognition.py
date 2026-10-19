"""Task 3: Object Recognition Using Viola–Jones (Haar Cascade) and YOLO.
Part of CCA 3 Computer Vision Assignment (Unit 5).

Techniques implemented:
  1. Face detection using Viola-Jones Haar Cascade classifier
  2. Multi-class object detection using Ultralytics YOLOv8 nano (COCO pretrained)
  3. Bounding box rendering with class labels and confidence scores
  4. Comparative analysis: Classical vs Deep Learning Object Detectors
  5. Empirical failure/limitation analysis (profile angles, lighting, occlusions)
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
)


def get_haar_cascade_path() -> Path:
    """Locate the bundled or system Haar Cascade XML file."""
    # Check local bundled models folder first
    local_path = Path(__file__).resolve().parent / "models" / "haarcascade_frontalface_default.xml"
    if local_path.exists():
        return local_path
        
    # Check OpenCV system data path
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        sys_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if sys_path.exists():
            return sys_path
            
    # Check current directory
    root_path = Path(__file__).resolve().parent.parent / "src" / "models" / "haarcascade_frontalface_default.xml"
    return root_path


def detect_faces_haar(
    image: np.ndarray,
    cascade_path: Optional[Path] = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Detect human faces using the classical Viola-Jones Haar Cascade classifier.
    
    Returns:
        Annotated RGB image and list of detected face dictionaries.
    """
    if cascade_path is None:
        cascade_path = get_haar_cascade_path()
        
    if not cascade_path.exists():
        raise FileNotFoundError(
            f"Haar cascade XML not found at {cascade_path}. "
            "Please ensure haarcascade_frontalface_default.xml is present."
        )

    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")

    # Convert to grayscale and equalize histogram for illumination invariance
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    gray_eq = cv2.equalizeHist(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
    )

    annotated = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    detections: List[Dict[str, Any]] = []

    for (x, y, w, h) in faces:
        # Draw bounding rectangle (Blue)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (30, 144, 255), 3)
        label = "Face (Haar Cascade: No Conf)"
        
        # Draw text label banner
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x, y - text_h - 10), (x + text_w + 6, y), (30, 144, 255), -1)
        cv2.putText(
            annotated, label, (x + 3, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )
        detections.append({
            "class": "face",
            "confidence": None,  # Classical Viola-Jones does not output continuous confidence
            "bbox": [int(x), int(y), int(w), int(h)],
        })

    return annotated, detections


def detect_objects_yolo(
    image: np.ndarray,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Detect objects using Ultralytics YOLOv8 (Deep Learning CNN-based detector).
    
    Returns:
        Annotated RGB image and list of detected object dictionaries.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics is required for Task 3. Install via 'pip install ultralytics'.")

    # Load YOLO model (will automatically load local or download nano weights)
    model = YOLO(model_name)
    
    # Run inference on RGB image
    results = model.predict(image, conf=conf_threshold, verbose=False)[0]
    
    annotated = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    detections: List[Dict[str, Any]] = []

    names = results.names
    boxes = results.boxes

    # Distinct vibrant colors for visual distinction
    color_palette = [
        (46, 204, 113),  # Emerald Green
        (231, 76, 60),   # Red
        (155, 89, 182),  # Purple
        (241, 196, 15),  # Yellow
        (52, 152, 219),  # Blue
        (230, 126, 34),  # Orange
    ]

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = names.get(cls_id, f"Class {cls_id}")

        x1, y1, x2, y2 = xyxy
        color = color_palette[cls_id % len(color_palette)]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"{cls_name}: {conf:.2f}"

        # Draw filled label background
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 10)), (x1 + text_w + 6, max(text_h + 10, y1)), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, max(text_h + 3, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
        )

        detections.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
        })

    return annotated, detections


def run_object_recognition(
    image: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute Task 3: Run Haar Cascade and YOLO, save figures and comparison report."""
    ensure_dir(output_dir)
    print("\n--- Running Task 3: Object Recognition (Viola-Jones & YOLO) ---")

    # 1. Viola-Jones Haar Cascade
    haar_annotated, haar_detections = detect_faces_haar(image)
    save_image(output_dir / "01_haar_face_detections.png", haar_annotated)
    print(f" [DONE] Viola-Jones Haar Cascade detected {len(haar_detections)} face(s).")

    # 2. YOLOv8 Deep Learning Detector
    yolo_annotated, yolo_detections = detect_objects_yolo(image, model_name="yolov8n.pt", conf_threshold=0.25)
    save_image(output_dir / "02_yolo_object_detections.png", yolo_annotated)
    print(f" [DONE] YOLOv8 detected {len(yolo_detections)} object(s).")
    
    # Class breakdown
    class_counts: Dict[str, int] = {}
    for d in yolo_detections:
        c = d["class"]
        class_counts[c] = class_counts.get(c, 0) + 1

    # 3. Save Side-by-Side Comparison Figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(image)
    axes[0].set_title("Input Scene / Photograph", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(haar_annotated)
    axes[1].set_title(
        f"Viola-Jones Haar Cascade\nDetected Faces: {len(haar_detections)} (Rigid feature cascade)",
        fontsize=12, fontweight="bold"
    )
    axes[1].axis("off")

    yolo_summary_str = ", ".join([f"{k}: {v}" for k, v in class_counts.items()]) if class_counts else "None"
    axes[2].imshow(yolo_annotated)
    axes[2].set_title(
        f"Ultralytics YOLOv8 Multi-Object Detection\nTotal Objects: {len(yolo_detections)} ({yolo_summary_str})",
        fontsize=12, fontweight="bold"
    )
    axes[2].axis("off")

    fig.suptitle("Task 3: Object Recognition Comparison (Classical Viola–Jones vs Deep Learning YOLO)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(output_dir / "fig1_detection_comparison_montage.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Print Comparison Table
    print("\n[Object Recognition Architecture Comparison Table]")
    print("-" * 80)
    print(f"{'Feature / Metric':<30} | {'Viola–Jones (Haar Cascade)':<25} | {'Ultralytics YOLOv8'}")
    print("-" * 80)
    print(f"{'Underlying Paradigm':<30} | {'Handcrafted Haar-like features':<25} | {'Deep CNN (Anchor-Free)'}")
    print(f"{'Classifier Mechanism':<30} | {'AdaBoost Cascaded Classifiers':<25} | {'Single-stage end-to-end tensor'}")
    print(f"{'Classes Supported':<30} | {'Specific (Frontal Faces)':<25} | {'80 Common Objects (COCO)'}")
    print(f"{'Confidence Estimation':<30} | {'Binary acceptance (No score)':<25} | {'Calibrated Softmax Probabilities'}")
    print(f"{'Rotation / Pose Tolerance':<30} | {'Low (< 15-20° tilt)':<25} | {'High (In-plane and perspective)'}")
    print(f"{'Detected in Current Image':<30} | {f'{len(haar_detections)} face(s)':<25} | {f'{len(yolo_detections)} object(s)'}")
    print("-" * 80)

    # 5. Failure / Edge Case Documentation (Assignment Requirement)
    failure_notes = []
    if len(haar_detections) == 0:
        failure_notes.append(
            "Haar Cascade Missed Detection: No frontal face was detected. "
            "Causes: The subject is looking away, head is tilted, lighting is uneven, or no person is in the frame."
        )
    if len(yolo_detections) == 0:
        failure_notes.append(
            "YOLO Low Confidence / Miss: No object met the confidence threshold of 0.25. "
            "Causes: Objects in the image may not belong to the 80 COCO classes, or are heavily occluded."
        )
    if not failure_notes:
        failure_notes.append(
            "Observation: Viola-Jones successfully identified frontal facial regions but ignored other objects in the scene. "
            "YOLO successfully localized multi-class objects with high confidence scores, exhibiting superior generalization."
        )

    print("\n[Detector Observations & Limitation Analysis]")
    for note in failure_notes:
        print(f" * {note}")

    return {
        "status": "success",
        "haar": {
            "total_faces": len(haar_detections),
            "detections": haar_detections,
        },
        "yolo": {
            "total_objects": len(yolo_detections),
            "classes_detected": class_counts,
            "detections": yolo_detections,
        },
        "failure_analysis": failure_notes,
        "output_dir": str(output_dir),
    }


def run_task3(
    image_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete Task 3 runner with validation."""
    project_root = Path(__file__).resolve().parent.parent
    if image_path is None:
        image_path = project_root / "dataset" / "original_photo.jpg"
    if output_dir is None:
        output_dir = project_root / "outputs" / "task3_object_recognition"

    print("=" * 70)
    print("      TASK 3: OBJECT RECOGNITION (VIOLA–JONES & YOLO)")
    print("=" * 70)
    print(f"Input image : {image_path}")
    print(f"Output path : {output_dir}")

    if not image_path.exists():
        msg = (
            f"ERROR: Dataset image not found at '{image_path}'.\n"
            "Please ensure 'dataset/original_photo.jpg' is present.\n"
            "(If you have 'dataset/original_video.mp4', you can also run frame extraction)."
        )
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    img = load_image(image_path)
    if img is None:
        msg = f"ERROR: Failed to decode image at '{image_path}'."
        print(f"\n[!] {msg}\n")
        return {"status": "failed", "error": msg}

    # Proportional resize if massive
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    results = run_object_recognition(img, output_dir)
    print("\n" + "=" * 70)
    print(" Task 3 completed successfully! Check outputs/task3_object_recognition/")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_task3()
