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

CONFIDENCE = 0.35
IOU = 0.5
DEVICE = "cpu"

RESIZE_TO = (640, 640)
PROCESS_EVERY_N_FRAMES = 2
SHOW = True

# ----------------------------
# Load model
# ----------------------------
model = YOLO(MODEL_PATH)
print("✅ Model loaded with TRACKING enabled")

videos = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
print(f"📂 Found {len(videos)} video(s)")

cv2.namedWindow("YOLO Tracking (ID)", cv2.WINDOW_NORMAL)

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

        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        processed_frames += 1

        frame_small = cv2.resize(frame, RESIZE_TO)

        # 🔥 TRACK instead of predict
        results = model.track(
            source=frame_small,
            conf=CONFIDENCE,
            iou=IOU,
            device=DEVICE,
            persist=True,      # مهم جدًا للحفاظ على ID
            verbose=False
        )

        annotated = results[0].plot()

        # FPS
        elapsed = time.time() - start_time
        fps = processed_frames / elapsed if elapsed > 0 else 0

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
            cv2.imshow("YOLO Tracking (ID)", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    print(f"✅ Finished: {os.path.basename(vid_path)}")

cv2.destroyAllWindows()
print("\n🎯 All videos processed with TRACK IDs.")
