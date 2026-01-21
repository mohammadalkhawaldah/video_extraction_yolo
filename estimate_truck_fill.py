from __future__ import annotations

import argparse
import base64
import json
import re
import time
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI


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
    "- category (one of: empty, 25%, 50%, 75%, 90%, full)\n"
    "- short_reasoning (<= 20 words)\n"
    "Use only the image."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate dump truck sand fill percentage from an image."
    )
    parser.add_argument(
        "--image",
        help="Path to a single truck image (local file path).",
    )
    parser.add_argument(
        "--dir",
        help="Directory of truck images (defaults to ./truck_images).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI vision model (Responses API compatible).",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Retry count if JSON parsing fails.",
    )
    return parser.parse_args()


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


def image_to_data_url(image_path: Path) -> str:
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in {"jpg", "jpeg"} else "image/png"
    return f"data:{mime};base64,{encoded}"


def strict_json_from_text(text: str) -> Dict[str, Any]:
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


def iter_images(image_path: str | None, dir_path: str | None) -> List[Path]:
    if image_path:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return [path]
    root = Path(dir_path) if dir_path else Path("truck_images")
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    allowed = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in root.iterdir() if p.suffix.lower() in allowed])


def estimate_fill(
    client: OpenAI, image_path: Path, args: argparse.Namespace
) -> Dict[str, Any]:
    image_url = image_to_data_url(image_path)
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            prompt_suffix = ""
            if attempt > 0:
                prompt_suffix = (
                    "\nReturn ONLY the JSON object on a single line. "
                    "Do not add any extra text."
                )
            response = client.responses.create(
                model=args.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": USER_PROMPT + prompt_suffix},
                            {"type": "input_image", "image_url": image_url},
                        ],
                    },
                ],
            )
            text = response.output_text.strip()
            return strict_json_from_text(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= args.max_retries:
                break
    raise RuntimeError(f"Failed to get valid JSON for {image_path.name}: {last_error}")


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))

    try:
        images = iter_images(args.image, args.dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if not images:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    rows: List[Dict[str, Any]] = []
    for image_path in images:
        try:
            start_time = time.perf_counter()
            result = estimate_fill(client, image_path, args)
            elapsed = time.perf_counter() - start_time
        except Exception as exc:  # noqa: BLE001
            print(str(exc), file=sys.stderr)
            continue
        payload = {"image": str(image_path), **result}
        print(json.dumps(payload, ensure_ascii=True))
        rows.append(
            {
                "image": image_path.name,
                "category": result.get("category", ""),
                "fill_percent": result.get("fill_percent", ""),
                "confidence": result.get("confidence", ""),
                "time_sec": f"{elapsed:.2f}",
            }
        )

    if rows:
        headers = ["image", "category", "fill_percent", "confidence", "time_sec"]
        widths = {h: len(h) for h in headers}
        for row in rows:
            for h in headers:
                widths[h] = max(widths[h], len(str(row.get(h, ""))))
        header_line = " | ".join(h.ljust(widths[h]) for h in headers)
        sep_line = "-+-".join("-" * widths[h] for h in headers)
        print(header_line)
        print(sep_line)
        for row in rows:
            print(" | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


if __name__ == "__main__":
    main()
