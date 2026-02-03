import cv2
import os
import glob

# ================= USER CONFIG =================
INPUT_ROOT = r"E:\Videos_Jan21-23_Camera-3\sliced_Videos"
OUTPUT_SUBFOLDER_NAME = "sliced_frames"

FRAME_INTERVAL_SEC = 0.4   # 0.5 = half second | 0.33 = one-third second
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")
# ===============================================

if not os.path.isdir(INPUT_ROOT):
    raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

subfolders = [
    os.path.join(INPUT_ROOT, name)
    for name in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, name))
]

if not subfolders:
    print("No subfolders found under input root.")
    raise SystemExit(0)

for folder in subfolders:
    output_root = os.path.join(folder, OUTPUT_SUBFOLDER_NAME)
    os.makedirs(output_root, exist_ok=True)

    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(folder, ext)))

    if not video_files:
        print(f"[SKIP] No video files in {folder}")
        continue

    print(f"[INFO] Processing folder: {folder} ({len(video_files)} videos)")

    for video_path in sorted(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_root, video_name)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            print(f"[WARN] Could not read FPS for {video_name}, skipping.")
            cap.release()
            continue

        frame_interval = max(1, int(fps * FRAME_INTERVAL_SEC))

        frame_count = 0
        saved_count = 0

        print(f"  [RUN] {video_name} | FPS={fps:.2f}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"  [DONE] Saved {saved_count} frames for {video_name}")

print("All folders processed successfully.")
