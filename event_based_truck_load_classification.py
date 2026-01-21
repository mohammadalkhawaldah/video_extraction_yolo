import os
import glob
import cv2
import time
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\best.pt"
VIDEO_FOLDER = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\test_videos"

CONFIDENCE = 0.35
IOU = 0.5
DEVICE = "cpu"     # غيّرها إلى 0 إذا كان CUDA متوفرًا

# Performance (CPU)
RESIZE_TO = (640, 640)
PROCESS_EVERY_N_FRAMES = 2

# Event logic
EXIT_MISSED_FRAMES = 10   # عدد الفريمات بعد الاختفاء لاعتبار الخروج

# Display
SHOW = True

# =====================================================
# Helpers
# =====================================================
def sec_to_hhmmss_msec(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = (int(seconds) // 3600)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def frame_to_time(frame_index: int, fps: float) -> float:
    if fps <= 0:
        return 0.0
    return frame_index / fps

# =====================================================
# MAIN
# =====================================================
def main():
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded")
    print("🎯 Event-Based Truck Load Classification\n")

    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    if not video_files:
        raise FileNotFoundError("❌ No videos found")

    if SHOW:
        cv2.namedWindow("YOLO Event-Based Classification", cv2.WINDOW_NORMAL)

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # لكل Track ID نخزن بياناته
        track_data = defaultdict(lambda: {
            "scores": defaultdict(float),   # مجموع confidence لكل class
            "counts": defaultdict(int),     # عدد الفريمات لكل class
            "first_seen": None,              # أول فريم (processed)
            "last_seen": None,               # آخر فريم (processed)
            "finalized": False
        })

        processed_frame_idx = 0
        original_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            original_frame_idx += 1
            if original_frame_idx % PROCESS_EVERY_N_FRAMES != 0:
                continue

            processed_frame_idx += 1
            frame_small = cv2.resize(frame, RESIZE_TO)

            # Tracking
            results = model.track(
                source=frame_small,
                conf=CONFIDENCE,
                iou=IOU,
                device=DEVICE,
                persist=True,
                verbose=False
            )

            r = results[0]
            names = r.names

            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                clss = r.boxes.cls.int().tolist()
                confs = r.boxes.conf.tolist()

                for tid, cid, cf in zip(ids, clss, confs):
                    d = track_data[tid]
                    if d["finalized"]:
                        continue

                    d["scores"][cid] += float(cf)
                    d["counts"][cid] += 1

                    if d["first_seen"] is None:
                        d["first_seen"] = processed_frame_idx
                    d["last_seen"] = processed_frame_idx

            # Check exit condition (event-based)
            for tid, d in list(track_data.items()):
                if d["finalized"] or d["last_seen"] is None:
                    continue

                if processed_frame_idx - d["last_seen"] > EXIT_MISSED_FRAMES:
                    best_class_id = max(
                        d["scores"],
                        key=lambda c: (d["scores"][c], d["counts"][c])
                    )

                    event_time_sec = frame_to_time(
                        d["last_seen"] * PROCESS_EVERY_N_FRAMES, fps
                    )
                    event_time_str = sec_to_hhmmss_msec(event_time_sec)

                    print(
                        f"🚚 EVENT | track_id={tid} | class={names[int(best_class_id)]} "
                        f"| timestamp={event_time_str}"
                    )

                    d["finalized"] = True

            # Display
            if SHOW:
                annotated = r.plot()
                cv2.imshow("YOLO Event-Based Classification", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        # Finalize remaining IDs at end of video
        for tid, d in track_data.items():
            if d["finalized"] or not d["scores"]:
                continue

            best_class_id = max(
                d["scores"],
                key=lambda c: (d["scores"][c], d["counts"][c])
            )

            event_time_sec = frame_to_time(
                d["last_seen"] * PROCESS_EVERY_N_FRAMES, fps
            )
            event_time_str = sec_to_hhmmss_msec(event_time_sec)

            print(
                f"🚚 EVENT | track_id={tid} | class={names[int(best_class_id)]} "
                f"| timestamp={event_time_str}"
            )

        cap.release()

    if SHOW:
        cv2.destroyAllWindows()

    print("\n✅ Processing completed")

# =====================================================
if __name__ == "__main__":
    main()

