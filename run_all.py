"""
run_all.py — Master pipeline runner
Formative 2: Multimodal Data Preprocessing

Runs all tasks in order:
  1. Data merge + EDA
  2. Image processing
  3. Audio processing
  4. Model training
  5. System simulation

Usage:
  python run_all.py               # Full pipeline
  python run_all.py --demo        # Skip training, run simulation only
"""

import os, sys, argparse

def section(title):
    print(f"\n\033[96m\033[1m{'='*60}\n  {title}\n{'='*60}\033[0m")

def run_task(script, label):
    section(label)
    result = os.system(f"python {script}")
    if result != 0:
        print(f"\033[91m[ERROR] {label} failed (exit code {result})\033[0m")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Skip training, run simulation only")
    parser.add_argument("--member", default="member_1", help="Member to simulate (default: member_1)")
    args = parser.parse_args()

    os.makedirs("features", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not args.demo:
        ok = run_task("scripts/task1_data_merge.py", "TASK 1: Data Merge + EDA")
        if not ok: sys.exit(1)

        ok = run_task("scripts/task2_image_processing.py", "TASK 2: Image Processing")
        if not ok: sys.exit(1)

        ok = run_task("scripts/task3_audio_processing.py", "TASK 3: Audio Processing")
        if not ok: sys.exit(1)

        ok = run_task("scripts/task4_model_training.py", "TASK 4: Model Training")
        if not ok: sys.exit(1)

    section("TASK 6: System Simulation — FULL DEMO")
    os.system(f"python scripts/task6_system_simulation.py --member {args.member}")

    section("TASK 6: System Simulation — UNAUTHORIZED ATTEMPT")
    os.system("python scripts/task6_system_simulation.py --unauthorized")

    print("\n\033[92m\033[1m Full pipeline complete.\033[0m")
    print("Outputs:")
    print("  features/merged_dataset.csv")
    print("  features/image_features.csv")
    print("  features/audio_features.csv")
    print("  features/eda_plots.png + augmentation plots")
    print("  models/face_model.pkl")
    print("  models/voice_model.pkl")
    print("  models/product_model.pkl")

if __name__ == "__main__":
    main()
