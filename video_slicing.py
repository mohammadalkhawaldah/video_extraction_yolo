import cv2
import os
import glob

# ================= USER CONFIG =================
INPUT_FOLDER = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\videos_to_slice"
OUTPUT_FOLDER = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\slicing_result"

FRAME_INTERVAL_SEC = 0.5   # 0.5 = نصف ثانية | 0.33 = ثلث ثانية
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")
# ===============================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_files = []
for ext in VIDEO_EXTENSIONS:
    video_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

if not video_files:
    print("❌ No video files found!")
    exit()

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(OUTPUT_FOLDER, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"⚠️ Could not read FPS for {video_name}, skipping.")
        continue

    frame_interval = max(1, int(fps * FRAME_INTERVAL_SEC))

    frame_count = 0
    saved_count = 0

    print(f"▶ Processing: {video_name} | FPS={fps:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_dir, f"frame_{saved_count:06d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✔ Saved {saved_count} frames for {video_name}")

print("✅ All videos processed successfully.")
