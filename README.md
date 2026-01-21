# Truck Clip Extraction (YOLO + ffmpeg)

Extracts only the portions of long gate videos where a **truck** appears. The scan is done at lower inference resolution for speed; clips are cut from the original file to preserve quality.

## Setup (Windows 11, CPU-only)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ensure `ffmpeg` is installed and available on `PATH`.

## Usage

Single file:

```powershell
python .\extract_truck_clips.py --input "D:\videos\gate01.mp4" --output "D:\clips_out"
```

Directory (recursive scan for .mp4/.mov/.mkv):

```powershell
python .\extract_truck_clips.py --input "D:\videos" --output "D:\clips_out"
```

## Tips for CPU speed

- Increase stride (e.g., `--stride 10`)
- Reduce inference size (e.g., `--imgsz 384`)
- Slightly raise confidence (e.g., `--conf 0.45`)

## Accurate vs fast cutting

Default mode uses fast stream copy (`-c copy`) which may start a bit before the exact frame due to keyframes. If you need accurate start points, enable:

```powershell
python .\extract_truck_clips.py --input "D:\videos\gate01.mp4" --output "D:\clips_out" --accurate_cut
```

## Outputs

For each input video:

```
<output>/<video_base>/
  truck_events.json
  report.txt
  clips/
    <video_base>_truck_0001.mp4
    <video_base>_truck_0002.mp4
```
