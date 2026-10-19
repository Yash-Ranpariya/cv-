"""CCA 3 Computer Vision Assignment — Master Entry Point.
Provides an interactive command-line interface and scriptable execution
for Units 3, 4, and 5 practical tasks.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.dataset_validator import validate_dataset, print_dataset_status, extract_frame_from_video
from src.task1_restoration_segmentation import run_task1
from src.task2_feature_extraction import run_task2
from src.task3_object_recognition import run_task3
from src.task4_motion_estimation import run_task4


PROJECT_ROOT = Path(__file__).resolve().parent


def print_banner() -> None:
    """Print the project execution header."""
    print("\n" + "=" * 75)
    print("      CCA 3 COMPUTER VISION — PRACTICAL CODING ASSIGNMENT (MCA)")
    print("      Units 3, 4 & 5 | OpenCV, NumPy, Matplotlib, skimage, YOLO")
    print("=" * 75)


def handle_task1() -> None:
    """Safely invoke Task 1 with user guidance."""
    print_banner()
    v = validate_dataset(PROJECT_ROOT)
    photo_path = PROJECT_ROOT / "dataset" / "original_photo.jpg"
    video_path = PROJECT_ROOT / "dataset" / "original_video.mp4"

    if not photo_path.exists():
        if video_path.exists():
            print("\n[Notice] 'original_photo.jpg' is missing, but 'original_video.mp4' was found.")
            print("Per assignment specification, extracting video frame 15 to use for Task 1...")
            extracted = extract_frame_from_video(video_path, photo_path, frame_number=15)
            if extracted:
                print(f"[Success] Extracted video frame saved to: {photo_path}")
            else:
                print("[Error] Failed to extract frame from video.")
                return
        else:
            print("\n[!] Input Missing for Task 1:")
            print("    Required file: dataset/original_photo.jpg")
            print("    Please take a photo with your mobile phone and save it to the 'dataset/' folder.")
            return

    run_task1(photo_path, PROJECT_ROOT / "outputs" / "task1_restoration_segmentation")


def handle_task2() -> None:
    """Safely invoke Task 2 with user guidance."""
    print_banner()
    photo1 = PROJECT_ROOT / "dataset" / "original_photo.jpg"
    photo2 = PROJECT_ROOT / "dataset" / "second_photo.jpg"

    missing = []
    if not photo1.exists():
        missing.append("dataset/original_photo.jpg")
    if not photo2.exists():
        missing.append("dataset/second_photo.jpg")

    if missing:
        print("\n[!] Inputs Missing for Task 2 (Feature Extraction & Matching):")
        for m in missing:
            print(f"    Missing: {m}")
        print("\n    Action needed:")
        print("    Capture two photos of the SAME object or scene from slightly different perspectives.")
        print("    Place them inside the 'dataset/' folder and re-run.")
        return

    run_task2(photo1, photo2, PROJECT_ROOT / "outputs" / "task2_feature_extraction")


def handle_task3() -> None:
    """Safely invoke Task 3 with user guidance."""
    print_banner()
    photo_path = PROJECT_ROOT / "dataset" / "original_photo.jpg"
    video_path = PROJECT_ROOT / "dataset" / "original_video.mp4"

    target_image = photo_path
    if not photo_path.exists():
        if video_path.exists():
            print("\n[Notice] 'original_photo.jpg' is missing. Extracting frame from video for detection...")
            temp_frame = PROJECT_ROOT / "dataset" / "video_extracted_frame.jpg"
            if extract_frame_from_video(video_path, temp_frame, frame_number=20):
                target_image = temp_frame
            else:
                print("[Error] Failed to extract frame from video.")
                return
        else:
            print("\n[!] Input Missing for Task 3:")
            print("    Required: dataset/original_photo.jpg (or dataset/original_video.mp4)")
            print("    Please place your photo or video in 'dataset/' and re-run.")
            return

    run_task3(target_image, PROJECT_ROOT / "outputs" / "task3_object_recognition")


def handle_task4() -> None:
    """Safely invoke Task 4 with user guidance."""
    print_banner()
    video_path = PROJECT_ROOT / "dataset" / "original_video.mp4"

    if not video_path.exists():
        print("\n[!] Input Missing for Task 4 (Video Motion Estimation):")
        print("    Required file: dataset/original_video.mp4")
        print("    Action needed:")
        print("    Record a 15–30 second video with moving objects/persons using a steady camera.")
        print("    Save the file as 'dataset/original_video.mp4' and re-run.")
        return

    run_task4(video_path, PROJECT_ROOT / "outputs" / "task4_motion_estimation")


def handle_all_tasks() -> None:
    """Execute all tasks sequentially without crashing if one task has missing input."""
    print_banner()
    print("\n>>> EXECUTING ALL TASKS (Graceful Pipeline) <<<\n")
    
    print("\n--- [Step 1/4] Running Task 1 ---")
    handle_task1()
    
    print("\n--- [Step 2/4] Running Task 2 ---")
    handle_task2()
    
    print("\n--- [Step 3/4] Running Task 3 ---")
    handle_task3()
    
    print("\n--- [Step 4/4] Running Task 4 ---")
    handle_task4()
    
    print("\n" + "=" * 75)
    print(" Pipeline execution finished! Check the outputs/ folder for all generated figures.")
    print("=" * 75)


def interactive_menu() -> None:
    """Run the interactive CLI selection menu."""
    while True:
        print_banner()
        print(" Select an option to execute:")
        print("  1. Run Task 1 — Image Restoration and Segmentation")
        print("  2. Run Task 2 — Feature Extraction and Image Matching")
        print("  3. Run Task 3 — Object Recognition (Viola–Jones & YOLO)")
        print("  4. Run Task 4 — Video Motion Estimation")
        print("  5. Run All Tasks")
        print("  6. Verify Dataset Status & Guidance")
        print("  7. Exit")
        print("-" * 75)

        choice = input("Enter choice (1-7): ").strip()
        
        if choice == "1":
            handle_task1()
        elif choice == "2":
            handle_task2()
        elif choice == "3":
            handle_task3()
        elif choice == "4":
            handle_task4()
        elif choice == "5":
            handle_all_tasks()
        elif choice == "6":
            print_dataset_status(validate_dataset(PROJECT_ROOT))
        elif choice == "7":
            print("\nExiting CCA 3 Computer Vision system. Goodbye!\n")
            break
        else:
            print(f"\n[!] Invalid option '{choice}'. Please enter a number between 1 and 7.")
            
        input("\nPress Enter to return to menu...")


def main() -> None:
    """Main CLI entry point with command-line arguments and interactive fallback."""
    parser = argparse.ArgumentParser(description="CCA 3 Computer Vision Assignment CLI")
    parser.add_argument("--task", type=str, choices=["1", "2", "3", "4", "all"], help="Execute specific task or all")
    parser.add_argument("--validate", action="store_true", help="Validate dataset files")
    args = parser.parse_args()

    if args.validate:
        print_dataset_status(validate_dataset(PROJECT_ROOT))
        return

    if args.task:
        if args.task == "1":
            handle_task1()
        elif args.task == "2":
            handle_task2()
        elif args.task == "3":
            handle_task3()
        elif args.task == "4":
            handle_task4()
        elif args.task == "all":
            handle_all_tasks()
        return

    # Fallback to interactive menu
    interactive_menu()


if __name__ == "__main__":
    main()
