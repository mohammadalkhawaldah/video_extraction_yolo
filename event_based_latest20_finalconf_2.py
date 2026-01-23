# =====================================================
# event_based_latest20_finalconf_2.py
# - Same logic as your current script (LLM + event gating unchanged)
# - Fix: load_class is computed by associating NON-truck detections to the Truck track
#        using Criterion A: (load bbox center inside truck bbox) IN THE SAME FRAME.
# - No second YOLO run.
# =====================================================

import base64
import glob
import json
import os
import re
import time
from pathlib import Path
import cv2
from collections import defaultdict, deque
from ultralytics import YOLO
from openai import OpenAI

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
WINDOW_SIZE = 20                # آخر 20 فريم فقط
MIN_FRAMES_PER_CLASS = 5        # تجاهل أي كلاس < 5 فريمات (للـ Truck gating كما هو)
FINAL_CONF_THRESHOLD = 0.55     # ⭐ Final Average Confidence Threshold
EXIT_MISSED_FRAMES = 10         # الخروج من المشهد

SHOW = True

# Target class
TARGET_CLASS_NAME = "Truck"  # update if your model uses a different label
TARGET_CLASS_ID = None       # set to an int to match by class id instead

# LLM (load level estimation)
USE_LLM = True
LLM_MODEL = "gpt-5.2-chat-latest"
LLM_MAX_RETRIES = 2
LLM_BBOX_PADDING = 0.08   # 8% padding around bbox
LLM_TOP_EXTRA = 0.15      # extra padding upward (bed wall context)
LLM_MIN_CROP_SIZE = 480   # min edge length before upscaling
LLM_UPSCALE = 1.5         # upscale factor for small crops
LLM_MULTI_CROPS = 3      # number of top bboxes to sample per track (set 1 for single)
LLM_MAX_TOTAL_MS = 2000  # stop extra crops if total LLM time exceeds this (0 = no limit)

# Debug
DEBUG_CLASS_SCORES = True
DEBUG_ASSOCIATION = True  # prints load_top derived from association (set False to reduce console spam)

SYSTEM_PROMPT = (
    "You estimate dump-truck load fill level from a single image. "
    "Load type may be cement bags, blocks, fine sand, bars, stones, or empty. "
    "Focus on load level, not material type. "
    "Use only visual inspection. Compare load height to truck bed wall height. "
    "Assume a standard rectangular dump-truck container. "
    "If edges or load surface are unclear, lower confidence. "
    "Be conservative and realistic. "
    "Return strict JSON only."
)

USER_PROMPT = (
    "Estimate load fill level in the truck bed.\n"
    "Return JSON with:\n"
    "- fill_percent (integer 0-100)\n"
    "- confidence (float 0.0-1.0)\n"
    "- category (one of: 0%, 25%, 50%, 75%, 90%, 100%)\n"
    "- short_reasoning (<= 20 words)\n"
    "Use only the image."
)

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

def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def strict_json_from_text(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data

def crop_with_padding(frame, bbox_xyxy, padding: float, top_extra: float):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y - int(bh * top_extra))
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def encode_image_b64(image) -> str:
    h, w = image.shape[:2]
    if min(h, w) < LLM_MIN_CROP_SIZE:
        new_w = int(w * LLM_UPSCALE)
        new_h = int(h * LLM_UPSCALE)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode crop")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def estimate_fill_llm(client: OpenAI, image_b64: str) -> dict:
    last_error = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            prompt_suffix = ""
            if attempt > 0:
                prompt_suffix = (
                    "\nReturn ONLY the JSON object on a single line. "
                    "Do not add any extra text."
                )
            response = client.responses.create(
                model=LLM_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": USER_PROMPT + prompt_suffix},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        ],
                    },
                ],
            )
            return strict_json_from_text(response.output_text)
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_MAX_RETRIES:
                break
    raise RuntimeError(f"LLM failed: {last_error}")

def categorize_fill(percent: float) -> str:
    if percent <= 5:
        return "empty"
    if percent <= 35:
        return "25%"
    if percent <= 60:
        return "50%"
    if percent <= 82:
        return "75%"
    if percent <= 95:
        return "90%"
    return "full"

def is_target_class(class_id: int, names: dict) -> bool:
    if TARGET_CLASS_ID is not None:
        return int(class_id) == int(TARGET_CLASS_ID)
    name = names.get(int(class_id), "").lower()
    return TARGET_CLASS_NAME.lower() in name

def get_load_class(scores: dict, counts: dict, names: dict) -> str:
    load_candidates = [cid for cid in scores if not is_target_class(cid, names)]
    if not load_candidates:
        return "unknown"
    best_id = max(load_candidates, key=lambda c: (scores[c], counts[c]))
    return names.get(int(best_id), "unknown")

def top_classes(scores: dict, counts: dict, names: dict, top_n: int = 2):
    items = []
    for cid in scores:
        items.append(
            (
                names.get(int(cid), "unknown"),
                counts[cid],
                round(scores[cid] / max(counts[cid], 1), 3),
            )
        )
    items.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return items[:top_n]

def center_of_bbox(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def point_in_bbox(px, py, b):
    x1, y1, x2, y2 = b
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)

# =====================================================
# MAIN
# =====================================================
def main():
    load_dotenv(Path(".env"))
    llm_client = OpenAI() if USE_LLM else None

    model = YOLO(MODEL_PATH)
    print("✅ Model loaded")
    print(f"Classes: {model.names}")
    print("🎯 Event-Based Classification (Last 20 + Min 5 + Final Conf)\n")

    video_files = glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4"))
    if not video_files:
        raise FileNotFoundError("❌ No videos found")

    if SHOW:
        cv2.namedWindow("YOLO Event-Based (Stable)", cv2.WINDOW_NORMAL)

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Track data لكل ID
        track_data = defaultdict(lambda: {
            "history": deque(maxlen=WINDOW_SIZE),       # (class_id, conf) for the track itself
            "load_history": deque(maxlen=WINDOW_SIZE),  # (class_id, conf) associated to TRUCK tid ✅
            "last_seen": None,
            "finalized": False,
            "last_bbox": None,
            "last_frame": None,
            "best_bbox": None,
            "best_frame": None,
            "best_area": 0,
            "top_bboxes": [],  # list of (area, bbox, frame)
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

            # Update histories + collect detections for SAME-FRAME association
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().tolist()
                clss = r.boxes.cls.int().tolist()
                confs = r.boxes.conf.tolist()
                xyxys = r.boxes.xyxy.tolist()

                scale_x = frame.shape[1] / RESIZE_TO[0]
                scale_y = frame.shape[0] / RESIZE_TO[1]

                frame_dets = []  # (tid, cid, conf, bbox_o)

                for tid, cid, cf, bbox in zip(ids, clss, confs, xyxys):
                    tid = int(tid)
                    cid = int(cid)
                    cf = float(cf)

                    d = track_data[tid]
                    if d["finalized"]:
                        continue

                    # per-track history (unchanged)
                    d["history"].append((cid, cf))
                    d["last_seen"] = processed_frame_idx
                    d["last_frame"] = frame  # reference is fine

                    # scale bbox to original frame
                    x1, y1, x2, y2 = bbox
                    x1o = int(x1 * scale_x)
                    y1o = int(y1 * scale_y)
                    x2o = int(x2 * scale_x)
                    y2o = int(y2 * scale_y)
                    x1o = max(0, min(frame.shape[1] - 1, x1o))
                    x2o = max(0, min(frame.shape[1] - 1, x2o))
                    y1o = max(0, min(frame.shape[0] - 1, y1o))
                    y2o = max(0, min(frame.shape[0] - 1, y2o))
                    bbox_o = (x1o, y1o, x2o, y2o)
                    d["last_bbox"] = bbox_o

                    # keep best/top crops for LLM (unchanged)
                    area = max(0, x2o - x1o) * max(0, y2o - y1o)
                    if area > d["best_area"]:
                        d["best_area"] = area
                        d["best_bbox"] = bbox_o
                        d["best_frame"] = frame.copy()
                    if area > 0:
                        d["top_bboxes"].append((area, bbox_o, frame.copy()))
                        d["top_bboxes"] = sorted(
                            d["top_bboxes"], key=lambda x: x[0], reverse=True
                        )[: max(LLM_MULTI_CROPS, 1)]

                    frame_dets.append((tid, cid, cf, bbox_o))

                # -----------------------------------------------------
                # SAME-FRAME Association (Criterion A):
                # link non-truck detections to the truck bbox (by center inside truck bbox)
                # and store them in truck track_data[truck_tid]["load_history"]
                # -----------------------------------------------------
                truck_boxes = {}     # truck_tid -> bbox_o
                nontruck_dets = []   # (cid, conf, bbox_o)

                for tid_i, cid_i, cf_i, bbox_i in frame_dets:
                    if is_target_class(cid_i, names):
                        truck_boxes[tid_i] = bbox_i
                    else:
                        nontruck_dets.append((cid_i, cf_i, bbox_i))

                if truck_boxes and nontruck_dets:
                    for cid_i, cf_i, bbox_i in nontruck_dets:
                        cx, cy = center_of_bbox(bbox_i)

                        linked_tid = None
                        for ttid, tb in truck_boxes.items():
                            if point_in_bbox(cx, cy, tb):
                                linked_tid = ttid
                                break

                        if linked_tid is not None:
                            track_data[linked_tid]["load_history"].append((cid_i, cf_i))

            # تحقق من الخروج من المشهد
            for tid, d in list(track_data.items()):
                if d["finalized"] or d["last_seen"] is None:
                    continue

                if processed_frame_idx - d["last_seen"] > EXIT_MISSED_FRAMES:
                    scores = defaultdict(float)
                    counts = defaultdict(int)

                    # تجميع آخر 20 فريم (truck track history)
                    for cid, cf in d["history"]:
                        scores[cid] += cf
                        counts[cid] += 1

                    # Truck gating unchanged
                    valid_classes = [
                        cid for cid in scores
                        if counts[cid] >= MIN_FRAMES_PER_CLASS
                        and is_target_class(cid, names)
                    ]

                    if valid_classes:
                        best_class_id = max(valid_classes, key=lambda c: (scores[c], counts[c]))
                        avg_conf = scores[best_class_id] / counts[best_class_id]

                        # NEW: load class from associated load_history (same-frame association)
                        load_scores = defaultdict(float)
                        load_counts = defaultdict(int)
                        for cid_l, cf_l in d["load_history"]:
                            load_scores[cid_l] += cf_l
                            load_counts[cid_l] += 1

                        load_class = get_load_class(load_scores, load_counts, names)

                        debug_info = ""
                        assoc_info = ""
                        if DEBUG_CLASS_SCORES:
                            top2 = top_classes(scores, counts, names, top_n=2)
                            debug_info = f" | top={top2}"
                        if DEBUG_ASSOCIATION:
                            load_top = top_classes(load_scores, load_counts, names, top_n=3) if load_scores else []
                            assoc_info = f" | load_top={load_top}"

                        # ⭐ Final confidence threshold (unchanged)
                        if avg_conf >= FINAL_CONF_THRESHOLD:
                            event_time_sec = frame_to_time(d["last_seen"] * PROCESS_EVERY_N_FRAMES, fps)
                            event_time_str = sec_to_hhmmss_msec(event_time_sec)

                            llm_payload = None
                            llm_ms = None

                            # LLM logic (unchanged)
                            if USE_LLM and d["last_bbox"] is not None and llm_client:
                                crops = []
                                if d["top_bboxes"]:
                                    crops = d["top_bboxes"][: max(LLM_MULTI_CROPS, 1)]
                                else:
                                    crops = [(d["best_area"], d["best_bbox"], d["best_frame"])]

                                results_llm = []
                                total_ms = 0.0
                                for _, bbox, source_frame in crops:
                                    if source_frame is None or bbox is None:
                                        continue
                                    crop = crop_with_padding(source_frame, bbox, LLM_BBOX_PADDING, LLM_TOP_EXTRA)
                                    if crop is None:
                                        continue
                                    try:
                                        start = time.perf_counter()
                                        image_b64 = encode_image_b64(crop)
                                        payload = estimate_fill_llm(llm_client, image_b64)
                                        elapsed_ms = (time.perf_counter() - start) * 1000
                                        total_ms += elapsed_ms
                                        results_llm.append(payload)
                                    except Exception as exc:
                                        print(f"[WARN] LLM error for track {tid}: {exc}")
                                    if LLM_MAX_TOTAL_MS > 0 and total_ms >= LLM_MAX_TOTAL_MS:
                                        break

                                if results_llm:
                                    fills = [float(rr.get("fill_percent", 0)) for rr in results_llm]
                                    confs_llm = [float(rr.get("confidence", 0)) for rr in results_llm]
                                    avg_fill = round(sum(fills) / len(fills))
                                    avg_conf_llm = round(sum(confs_llm) / len(confs_llm), 3) if confs_llm else 0.0
                                    best_reason = max(results_llm, key=lambda rr: float(rr.get("confidence", 0))).get("short_reasoning", "")
                                    llm_payload = {
                                        "fill_percent": int(avg_fill),
                                        "confidence": avg_conf_llm,
                                        "category": categorize_fill(avg_fill),
                                        "short_reasoning": best_reason,
                                    }
                                    llm_ms = total_ms

                            if llm_payload:
                                print(
                                    f"EVENT {event_time_str} | "
                                    f"track_id={tid} | "
                                    f"class={names[int(best_class_id)]} | "
                                    f"load_class={load_class} | "
                                    f"load={llm_payload} | "
                                    f"llm_ms={llm_ms:.0f}"
                                    f"{debug_info}{assoc_info}"
                                )
                            else:
                                print(
                                    f"EVENT {event_time_str} | "
                                    f"track_id={tid} | "
                                    f"class={names[int(best_class_id)]} | "
                                    f"load_class={load_class}"
                                    f"{debug_info}{assoc_info}"
                                )

                    d["finalized"] = True

            # عرض الفيديو
            if SHOW:
                annotated = r.plot()
                cv2.imshow("YOLO Event-Based (Stable)", annotated)
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
                and is_target_class(cid, names)
            ]

            if valid_classes:
                best_class_id = max(valid_classes, key=lambda c: (scores[c], counts[c]))
                avg_conf = scores[best_class_id] / counts[best_class_id]

                # NEW: load class from associated load_history
                load_scores = defaultdict(float)
                load_counts = defaultdict(int)
                for cid_l, cf_l in d["load_history"]:
                    load_scores[cid_l] += cf_l
                    load_counts[cid_l] += 1
                load_class = get_load_class(load_scores, load_counts, names)

                debug_info = ""
                assoc_info = ""
                if DEBUG_CLASS_SCORES:
                    top2 = top_classes(scores, counts, names, top_n=2)
                    debug_info = f" | top={top2}"
                if DEBUG_ASSOCIATION:
                    load_top = top_classes(load_scores, load_counts, names, top_n=3) if load_scores else []
                    assoc_info = f" | load_top={load_top}"

                if avg_conf >= FINAL_CONF_THRESHOLD:
                    event_time_sec = frame_to_time(d["last_seen"] * PROCESS_EVERY_N_FRAMES, fps)
                    event_time_str = sec_to_hhmmss_msec(event_time_sec)

                    llm_payload = None
                    llm_ms = None
                    if USE_LLM and d["last_bbox"] is not None and llm_client:
                        crops = []
                        if d["top_bboxes"]:
                            crops = d["top_bboxes"][: max(LLM_MULTI_CROPS, 1)]
                        else:
                            crops = [(d["best_area"], d["best_bbox"], d["best_frame"])]

                        results_llm = []
                        total_ms = 0.0
                        for _, bbox, source_frame in crops:
                            if source_frame is None or bbox is None:
                                continue
                            crop = crop_with_padding(source_frame, bbox, LLM_BBOX_PADDING, LLM_TOP_EXTRA)
                            if crop is None:
                                continue
                            try:
                                start = time.perf_counter()
                                image_b64 = encode_image_b64(crop)
                                payload = estimate_fill_llm(llm_client, image_b64)
                                elapsed_ms = (time.perf_counter() - start) * 1000
                                total_ms += elapsed_ms
                                results_llm.append(payload)
                            except Exception as exc:
                                print(f"[WARN] LLM error for track {tid}: {exc}")
                            if LLM_MAX_TOTAL_MS > 0 and total_ms >= LLM_MAX_TOTAL_MS:
                                break

                        if results_llm:
                            fills = [float(rr.get("fill_percent", 0)) for rr in results_llm]
                            confs_llm = [float(rr.get("confidence", 0)) for rr in results_llm]
                            avg_fill = round(sum(fills) / len(fills))
                            avg_conf_llm = round(sum(confs_llm) / len(confs_llm), 3) if confs_llm else 0.0
                            best_reason = max(results_llm, key=lambda rr: float(rr.get("confidence", 0))).get("short_reasoning", "")
                            llm_payload = {
                                "fill_percent": int(avg_fill),
                                "confidence": avg_conf_llm,
                                "category": categorize_fill(avg_fill),
                                "short_reasoning": best_reason,
                            }
                            llm_ms = total_ms

                    if llm_payload:
                        print(
                            f"EVENT {event_time_str} | "
                            f"track_id={tid} | "
                            f"class={names[int(best_class_id)]} | "
                            f"load_class={load_class} | "
                            f"load={llm_payload} | "
                            f"llm_ms={llm_ms:.0f}"
                            f"{debug_info}{assoc_info}"
                        )
                    else:
                        print(
                            f"EVENT {event_time_str} | "
                            f"track_id={tid} | "
                            f"class={names[int(best_class_id)]} | "
                            f"load_class={load_class}"
                            f"{debug_info}{assoc_info}"
                        )

        cap.release()

    if SHOW:
        cv2.destroyAllWindows()

    print("\n✅ Processing completed")

# =====================================================
if __name__ == "__main__":
    main()
