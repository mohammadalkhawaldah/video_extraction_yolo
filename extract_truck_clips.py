from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract truck-only clips from videos using YOLO + ffmpeg."
    )
    parser.add_argument("--input", required=True, help="Input video file or directory.")
    parser.add_argument("--output", required=True, help="Output folder.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights path.")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--stride", type=int, default=8, help="Process every Nth frame.")
    parser.add_argument(
        "--start_hits",
        type=int,
        default=2,
        help="Consecutive detections required to start an event.",
    )
    parser.add_argument(
        "--end_misses",
        type=int,
        default=6,
        help="Consecutive misses required to end an event.",
    )
    parser.add_argument("--pre_sec", type=float, default=5.0, help="Pre-roll seconds.")
    parser.add_argument("--post_sec", type=float, default=5.0, help="Post-roll seconds.")
    parser.add_argument(
        "--min_event_sec", type=float, default=3.0, help="Minimum event length."
    )
    parser.add_argument(
        "--merge_gap_sec",
        type=float,
        default=1.5,
        help="Merge events if gap is below this threshold.",
    )
    parser.add_argument(
        "--confirm_stride",
        type=int,
        default=0,
        help="Confirm pass stride (0 = auto: stride//2).",
    )
    parser.add_argument(
        "--confirm_min_hits",
        type=int,
        default=3,
        help="Minimum truck detections inside event to keep it.",
    )
    parser.add_argument("--device", default="cpu", help="Device to run inference on.")
    parser.add_argument(
        "--accurate_cut",
        action="store_true",
        help="Re-encode clips for accurate cut points.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print events and ffmpeg commands without cutting.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=200,
        help="Print progress every N processed frames.",
    )
    parser.add_argument(
        "--extensions",
        default="mp4,mov,mkv",
        help="Comma-separated list of video extensions.",
    )
    return parser.parse_args()


def check_ffmpeg() -> None:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Ensure it is installed and on PATH.") from exc
    if result.returncode != 0:
        raise RuntimeError("ffmpeg is not available. Check your installation.")


def safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "video"


def discover_videos(input_path: Path, extensions: Iterable[str]) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    exts = {f".{ext.lower().lstrip('.')}" for ext in extensions}
    videos = [p for p in input_path.rglob("*") if p.suffix.lower() in exts]
    return sorted(videos)


def has_truck_detection(result, model_names) -> bool:
    if result.boxes is None or len(result.boxes) == 0:
        return False
    cls_ids = result.boxes.cls
    if cls_ids is None or len(cls_ids) == 0:
        return False
    for cls_id in cls_ids:
        name = model_names.get(int(cls_id), "")
        if name == "truck":
            return True
    return False


def format_ffmpeg_cmd(cmd: List[str]) -> str:
    def quote(arg: str) -> str:
        if " " in arg or "(" in arg or ")" in arg:
            return f"\"{arg}\""
        return arg

    return " ".join(quote(arg) for arg in cmd)


def build_ffmpeg_cmd(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    accurate_cut: bool,
) -> List[str]:
    start_str = f"{start_sec:.3f}"
    dur_str = f"{duration_sec:.3f}"
    if accurate_cut:
        return [
            "ffmpeg",
            "-y",
            "-ss",
            start_str,
            "-i",
            str(input_path),
            "-t",
            dur_str,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            str(output_path),
        ]
    return [
        "ffmpeg",
        "-y",
        "-ss",
        start_str,
        "-i",
        str(input_path),
        "-t",
        dur_str,
        "-an",
        "-c",
        "copy",
        str(output_path),
    ]


def build_ffmpeg_fallback_cmd(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
) -> List[str]:
    start_str = f"{start_sec:.3f}"
    dur_str = f"{duration_sec:.3f}"
    return [
        "ffmpeg",
        "-y",
        "-ss",
        start_str,
        "-i",
        str(input_path),
        "-t",
        dur_str,
        "-an",
        "-c:v",
        "copy",
        str(output_path),
    ]


def merge_events(events: List[Tuple[int, int]], gap_frames: int) -> List[Tuple[int, int]]:
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e[0])
    merged: List[Tuple[int, int]] = [events_sorted[0]]
    for start, end in events_sorted[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap_frames:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def confirm_events(
    video_path: Path,
    events: List[Tuple[int, int]],
    model: YOLO,
    args: argparse.Namespace,
) -> List[Tuple[int, int]]:
    if not events or args.confirm_min_hits <= 0:
        return events

    confirm_stride = args.confirm_stride if args.confirm_stride > 0 else max(1, args.stride // 2)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Confirm pass could not open {video_path}. Skipping confirm.")
        return events

    kept: List[Tuple[int, int]] = []
    for start, end in events:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        hits = 0
        consecutive_failures = 0
        while frame_idx <= end:
            if (frame_idx - start) % confirm_stride == 0:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                    result = model.predict(
                        frame,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        device=args.device,
                        verbose=False,
                    )[0]
                    if has_truck_detection(result, model.names):
                        hits += 1
                        if hits >= args.confirm_min_hits:
                            break
            else:
                ret = cap.grab()
                if not ret:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0

            if consecutive_failures >= 50:
                print(f"[WARN] Confirm pass read failures in {video_path}.")
                break
            frame_idx += 1

        if hits >= args.confirm_min_hits:
            kept.append((start, end))

    cap.release()
    return kept


def process_video(
    video_path: Path, output_dir: Path, model: YOLO, args: argparse.Namespace
) -> List[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    pre_frames = int(args.pre_sec * fps)
    post_frames = int(args.post_sec * fps)
    gap_frames = int(args.merge_gap_sec * fps)

    in_event = False
    consec_hits = 0
    consec_misses = 0
    candidate_start_frame: Optional[int] = None
    event_start_frame = 0
    last_truck_frame = -1

    events_raw: List[Tuple[int, int]] = []
    frame_idx = 0
    consecutive_failures = 0

    while True:
        if frame_idx % args.stride == 0:
            ret, frame = cap.read()
            if not ret:
                if total_frames > 0 and frame_idx >= total_frames - 1:
                    break
                consecutive_failures += 1
                frame_idx += 1
                if consecutive_failures >= 50:
                    print(f"[WARN] Too many frame read failures in {video_path}.")
                    break
                continue
            consecutive_failures = 0
        else:
            ret = cap.grab()
            if not ret:
                if total_frames > 0 and frame_idx >= total_frames - 1:
                    break
                consecutive_failures += 1
                frame_idx += 1
                if consecutive_failures >= 50:
                    print(f"[WARN] Too many frame read failures in {video_path}.")
                    break
                continue
            consecutive_failures = 0

        if args.log_every > 0 and frame_idx % args.log_every == 0 and frame_idx > 0:
            if total_frames > 0:
                pct = 100.0 * frame_idx / total_frames
                print(f"[INFO] {video_path.name}: {frame_idx}/{total_frames} ({pct:.1f}%)")
            else:
                print(f"[INFO] {video_path.name}: frame {frame_idx}")

        if frame_idx % args.stride == 0:
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            detected = has_truck_detection(result, model.names)

            if detected:
                if not in_event and consec_hits == 0:
                    candidate_start_frame = frame_idx
                consec_hits += 1
                consec_misses = 0
                last_truck_frame = frame_idx
                if not in_event and consec_hits >= args.start_hits:
                    in_event = True
                    event_start_frame = (
                        candidate_start_frame
                        if candidate_start_frame is not None
                        else frame_idx
                    )
            else:
                consec_misses += 1
                consec_hits = 0
                if in_event and consec_misses >= args.end_misses:
                    events_raw.append((event_start_frame, last_truck_frame))
                    in_event = False
                    candidate_start_frame = None

        frame_idx += 1

    if in_event and last_truck_frame >= 0:
        events_raw.append((event_start_frame, last_truck_frame))

    cap.release()

    adjusted_events: List[Tuple[int, int]] = []
    for start, end in events_raw:
        adj_start = max(0, start - pre_frames)
        adj_end = end + post_frames
        if total_frames > 0:
            adj_end = min(total_frames - 1, adj_end)
        if adj_end >= adj_start:
            adjusted_events.append((adj_start, adj_end))

    merged_events = merge_events(adjusted_events, gap_frames)

    filtered_events: List[Tuple[int, int]] = []
    for start, end in merged_events:
        duration_sec = (end - start) / fps
        if duration_sec >= args.min_event_sec:
            filtered_events.append((start, end))

    confirmed_events = confirm_events(video_path, filtered_events, model, args)

    events: List[dict] = []
    for idx, (start, end) in enumerate(confirmed_events, start=1):
        start_sec = start / fps
        end_sec = end / fps
        duration_sec = end_sec - start_sec
        events.append(
            {
                "index": idx,
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(duration_sec, 3),
            }
        )

    base_name = safe_stem(video_path.stem)
    video_out_dir = output_dir / base_name
    clips_dir = video_out_dir / "clips"
    video_out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    json_path = video_out_dir / "truck_events.json"
    report_path = video_out_dir / "report.txt"

    summary = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "stride": args.stride,
        "start_hits": args.start_hits,
        "end_misses": args.end_misses,
        "pre_sec": args.pre_sec,
        "post_sec": args.post_sec,
        "min_event_sec": args.min_event_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "confirm_stride": args.confirm_stride if args.confirm_stride > 0 else max(1, args.stride // 2),
        "confirm_min_hits": args.confirm_min_hits,
        "events": events,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"Video: {video_path}")
    lines.append(f"FPS: {fps}")
    lines.append(f"Frames: {total_frames}")
    lines.append(f"Resolution: {width}x{height}")
    lines.append(f"Stride: {args.stride}")
    lines.append(
        f"Confirm stride: "
        f"{args.confirm_stride if args.confirm_stride > 0 else max(1, args.stride // 2)}"
    )
    lines.append(f"Confirm min hits: {args.confirm_min_hits}")
    lines.append(f"Events: {len(events)}")
    lines.append("")
    for event in events:
        lines.append(
            f"[{event['index']:04d}] {event['start_sec']:.3f}s -> "
            f"{event['end_sec']:.3f}s ({event['duration_sec']:.3f}s)"
        )
    lines.append("")
    lines.append("ffmpeg commands:")

    if args.dry_run:
        print(f"[DRY RUN] {video_path.name}: {len(events)} event(s)")
        for event in events:
            print(
                f"  [{event['index']:04d}] {event['start_sec']:.3f}s -> "
                f"{event['end_sec']:.3f}s ({event['duration_sec']:.3f}s)"
            )

    for event in events:
        clip_name = f"{base_name}_truck_{event['index']:04d}.mp4"
        clip_path = clips_dir / clip_name
        cmd = build_ffmpeg_cmd(
            video_path,
            clip_path,
            float(event["start_sec"]),
            float(event["duration_sec"]),
            args.accurate_cut,
        )
        lines.append(format_ffmpeg_cmd(cmd))
        if args.dry_run:
            print(format_ffmpeg_cmd(cmd))
        else:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0 and not args.accurate_cut:
                fallback_cmd = build_ffmpeg_fallback_cmd(
                    video_path,
                    clip_path,
                    float(event["start_sec"]),
                    float(event["duration_sec"]),
                )
                lines.append(format_ffmpeg_cmd(fallback_cmd))
                print(
                    f"[WARN] ffmpeg copy failed; retrying with audio re-encode for "
                    f"{clip_path.name}"
                )
                subprocess.run(fallback_cmd, check=False)

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return events


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    check_ffmpeg()

    extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
    videos = discover_videos(input_path, extensions)
    if not videos:
        print(f"[INFO] No videos found in {input_path}")
        return

    print(f"[INFO] Found {len(videos)} video(s). Loading model...")
    model = YOLO(args.model)

    for video in videos:
        print(f"[INFO] Processing {video}")
        try:
            events = process_video(video, output_dir, model, args)
        except Exception as exc:
            print(f"[ERROR] {video}: {exc}")
            continue
        if not events:
            print(f"[INFO] No truck events found for {video.name}")


if __name__ == "__main__":
    main()
