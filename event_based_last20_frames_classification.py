import os
import glob
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\best.pt"
VIDEO_FOLDER = r"C:\Users\moham\OneDrive\Documents\clips_extraction_by_yolo\test_videos"

CONFIDENCE = 0.35
IOU = 0.5
DEVICE = "cpu"          # غيّرها إلى 0 إذا CUDA متوفر

# Performance
RESIZE_TO = (640, 640)
PROCESS_EVERY_N_FRAMES = 2

# Event logic
WINDOW_SIZE = 20                # نأخذ فقط آخر 20 فريم
MIN_FRAMES_PER_CLASS = 5        # ❗ شرطك الأساسي
EXIT_MISSED_FRAMES = 10         # الخروج من المشهد

SHOW = True

# =====================================================
# Helpers
# =====================================================
def sec_to_hhmmss_msec(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = (int(seconds) // 3600)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def frame_to_time(frame_index: int, fps: float) -> float:
    return frame_index / fps if fps > 0 else 0.0

# =====================================================
# MAIN
# =====================================================
def main():
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded")
    print("🎯 Event-Based Classification (Last 20 Frames + Min 5 Frames Rule)\n")

    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    if not video_files:
        raise FileNotFoundError("❌ No videos found")

    if SHOW:
        cv2.namedWindow("YOLO Event-Based (Stable Classes)", cv2.WINDOW_NORMAL)

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Track data لكل ID
        track_data = defaultdict(lambda: {
            "history": deque(maxlen=WINDOW_SIZE),  # (class_id, conf)
            "last_seen": None,
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

            # تحديث آخر 20 فريم
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                clss = r.boxes.cls.int().tolist()
                confs = r.boxes.conf.tolist()

                for tid, cid, cf in zip(ids, clss, confs):
                    d = track_data[tid]
                    if d["finalized"]:
                        continue

                    d["history"].append((cid, float(cf)))
                    d["last_seen"] = processed_frame_idx

            # تحقق من الخروج من المشهد
            for tid, d in list(track_data.items()):
                if d["finalized"] or d["last_seen"] is None:
                    continue

                if processed_frame_idx - d["last_seen"] > EXIT_MISSED_FRAMES:
                    scores = defaultdict(float)
                    counts = defaultdict(int)

                    # تجميع فقط آخر 20 فريم
                    for cid, cf in d["history"]:
                        scores[cid] += cf
                        counts[cid] += 1

                    # ❗ تجاهل أي كلاس أقل من 5 فريمات
                    valid_classes = [
                        cid for cid in scores
                        if counts[cid] >= MIN_FRAMES_PER_CLASS
                    ]

                    if valid_classes:
                        best_class_id = max(
                            valid_classes,
                            key=lambda c: (scores[c], counts[c])
                        )

                        event_time_sec = frame_to_time(
                            d["last_seen"] * PROCESS_EVERY_N_FRAMES, fps
                        )
                        event_time_str = sec_to_hhmmss_msec(event_time_sec)

                        print(
                            f"🚚 EVENT | track_id={tid} | "
                            f"class={names[int(best_class_id)]} | "
                            f"timestamp={event_time_str}"
                        )

                    d["finalized"] = True

            # عرض الفيديو
            if SHOW:
                annotated = r.plot()
                cv2.imshow("YOLO Event-Based (Stable Classes)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        # إنهاء أي ID متبقٍ عند نهاية الفيديو
        for tid, d in track_data.items():
            if d["finalized"] or not d["history"]:
                continue

            scores = defaultdict(float)
            counts = defaultdict(int)

            for cid, cf in d["history"]:
                scores[cid] += cf
                counts[cid] += 1

            valid_classes = [
                cid for cid in scores
                if counts[cid] >= MIN_FRAMES_PER_CLASS
            ]

            if valid_classes:
                best_class_id = max(
                    valid_classes,
                    key=lambda c: (scores[c], counts[c])
                )

                event_time_sec = frame_to_time(
                    d["last_seen"] * PROCESS_EVERY_N_FRAMES, fps
                )
                event_time_str = sec_to_hhmmss_msec(event_time_sec)

                print(
                    f"🚚 EVENT | track_id={tid} | "
                    f"class={names[int(best_class_id)]} | "
                    f"timestamp={event_time_str}"
                )

        cap.release()

    if SHOW:
        cv2.destroyAllWindows()

    print("\n✅ Processing completed")

# =====================================================
if __name__ == "__main__":
    main()
