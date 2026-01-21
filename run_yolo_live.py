import os
import glob
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\best.pt"
VIDEO_FOLDER = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\test_videos"
OUTPUT_ROOT = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\outputs"

CONFIDENCE = 0.35
IOU = 0.5
DEVICE = "cpu"

RESIZE_TO = (640, 640)          # أهم تسريع
PROCESS_EVERY_N_FRAMES = 2      # حلّل فريم كل 2
SHOW = True
SAVE = False                    # أوقف الحفظ لزيادة السرعة

# ----------------------------
# Prep
# ----------------------------
os.makedirs(OUTPUT_ROOT, exist_ok=True)
run_name = "fast_cpu_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(OUTPUT_ROOT, run_name)

model = YOLO(MODEL_PATH)
print("✅ Model loaded (CPU fast mode)")

videos = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
print(f"📂 Found {len(videos)} video(s)")

cv2.namedWindow("YOLO Live (Fast CPU)", cv2.WINDOW_NORMAL)

# ----------------------------
# Process videos
# ----------------------------
for vid_path in videos:
    print("\n▶️ Processing:", os.path.basename(vid_path))
    cap = cv2.VideoCapture(vid_path)

    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Frame skipping
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        processed_frames += 1

        # Resize for speed
        frame_small = cv2.resize(frame, RESIZE_TO)

        # Inference
        results = model.predict(
            source=frame_small,
            conf=CONFIDENCE,
            iou=IOU,
            device=DEVICE,
            verbose=False
        )

        annotated = results[0].plot()

        # FPS calculation
        elapsed = time.time() - start_time
        fps = processed_frames / elapsed if elapsed > 0 else 0

        # Overlay FPS
        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if SHOW:
            cv2.imshow("YOLO Live (Fast CPU)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    total_time = time.time() - start_time
    avg_fps = processed_frames / total_time if total_time > 0 else 0

    print(f"✅ Done {os.path.basename(vid_path)}")
    print(f"   Processed frames: {processed_frames}")
    print(f"   Avg FPS: {avg_fps:.2f}")

cv2.destroyAllWindows()
print("\n🎯 All videos processed.")
