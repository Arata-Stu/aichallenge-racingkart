#!/usr/bin/env python3
"""Export raw ROS image frames and optionally upload them to a remote dataset."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def image_msg_to_rgb(msg: Any) -> np.ndarray:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    if encoding in ("bgr8", "rgb8"):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
        if encoding == "bgr8":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif encoding in ("bgra8", "rgba8"):
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 4)
        code = cv2.COLOR_BGRA2RGB if encoding == "bgra8" else cv2.COLOR_RGBA2RGB
        image = cv2.cvtColor(image, code)
    elif encoding == "mono8":
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    return image


def write_image(
    rgb: np.ndarray,
    path: Path,
    image_format: str,
    jpeg_quality: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if image_format == "png":
        success = cv2.imwrite(str(path), bgr)
    else:
        success = cv2.imwrite(
            str(path),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
    if not success:
        raise RuntimeError(f"Failed to write image: {path}")


def safe_stem(value: str) -> str:
    sanitized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    ).strip("._-")
    return sanitized[:96] or "sequence"


@dataclass
class ImageRecord:
    path: Path
    name: str
    sequence: str
    timestamp_ns: int
    frame_index: int

    def as_json(self, root: Path) -> dict[str, Any]:
        return {
            "file": str(self.path.relative_to(root)),
            "name": self.name,
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "frame_index": self.frame_index,
        }


def export_sequence(
    sequence_dir: Path,
    out_dir: Path,
    image_topic: str,
    frame_stride: int,
    max_frames: int,
    image_format: str,
    jpeg_quality: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    sequence_stem = safe_stem(sequence_dir.name)
    extension = "png" if image_format == "png" else "jpg"
    print(f"[EXPORT] {sequence_dir}")
    raw_index = 0
    with AnyReader([sequence_dir]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == image_topic
        ]
        if not connections:
            print(f"[WARN] image topic not found: {image_topic}")
            return records
        for connection, timestamp, raw in reader.messages(connections=connections):
            if connection.msgtype != "sensor_msgs/msg/Image":
                continue
            if raw_index % frame_stride != 0:
                raw_index += 1
                continue
            if max_frames and len(records) >= max_frames:
                break
            try:
                rgb = image_msg_to_rgb(reader.deserialize(raw, connection.msgtype))
            except ValueError as exc:
                print(f"[WARN] skip frame {raw_index}: {exc}")
                raw_index += 1
                continue
            filename = f"{sequence_stem}_{timestamp}_{raw_index:06d}.{extension}"
            image_path = out_dir / sequence_stem / filename
            write_image(rgb, image_path, image_format, jpeg_quality)
            records.append(
                ImageRecord(
                    path=image_path,
                    name=filename,
                    sequence=sequence_dir.name,
                    timestamp_ns=int(timestamp),
                    frame_index=raw_index,
                )
            )
            raw_index += 1
    print(f"[EXPORT] {sequence_dir.name}: {len(records)} images")
    return records


def multipart_body(field_name: str, image_path: Path) -> tuple[bytes, str]:
    boundary = f"----learning-studio-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{image_path.name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return header + image_path.read_bytes() + footer, boundary


def upload_image(
    image: ImageRecord,
    project_id: str,
    api_key: str,
    split: str,
    batch_name: str,
    tags: list[str],
    sequence_number: int,
    sequence_size: int,
    retries: int,
) -> None:
    params: list[tuple[str, str | int]] = [
        ("api_key", api_key),
        ("name", image.name),
        ("split", split),
        ("sequence_number", sequence_number),
        ("sequence_size", sequence_size),
    ]
    if batch_name:
        params.append(("batch", batch_name))
    for tag in tags:
        params.append(("tag", tag))
    url = (
        f"https://api.roboflow.com/dataset/{quote(project_id, safe='')}/upload?"
        f"{urlencode(params)}"
    )
    body, boundary = multipart_body("file", image.path)
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read().decode("utf-8", errors="replace")
            print(f"[UPLOAD] {image.name}: {payload}")
            return
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if attempt >= retries:
                raise RuntimeError(
                    f"Remote upload failed for {image.name}: {exc.code} {detail}"
                ) from exc
        except Exception as exc:
            if attempt >= retries:
                raise RuntimeError(f"Remote upload failed for {image.name}: {exc}") from exc
        time.sleep(min(2**attempt, 8))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export/upload raw images.")
    parser.add_argument("--seq-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--image-topic", default="/sensing/camera/image_raw")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--batch-name", default="")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--api-key-env", default="PRIVATE_UPLOAD_API_KEY")
    parser.add_argument("--upload-retries", type=int, default=2)
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.max_frames_per_sequence < 0:
        raise ValueError("--max-frames-per-sequence must be >= 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_records: list[ImageRecord] = []
    for sequence_dir in args.seq_dirs:
        all_records.extend(
            export_sequence(
                sequence_dir.resolve(),
                args.outdir,
                args.image_topic,
                args.frame_stride,
                args.max_frames_per_sequence,
                args.format,
                args.jpeg_quality,
            )
        )
    if not all_records:
        raise RuntimeError("No images were exported.")

    manifest = {
        "created_at": utc_stamp(),
        "image_topic": args.image_topic,
        "frame_stride": args.frame_stride,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "format": args.format,
        "jpeg_quality": args.jpeg_quality,
        "images_dir": str(args.outdir.resolve()),
        "image_count": len(all_records),
        "sequences": [str(path.resolve()) for path in args.seq_dirs],
        "images": [record.as_json(args.outdir) for record in all_records],
        "upload": {
            "enabled": bool(args.upload),
            "project_id": args.project_id,
            "split": args.split,
            "batch_name": args.batch_name,
            "tags": args.tag,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Exported {len(all_records)} images: {args.outdir}")
    print(f"[DONE] Manifest: {args.manifest}")

    if not args.upload:
        return
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set.")
    if not args.project_id:
        raise RuntimeError("--project-id is required when --upload is enabled.")

    print(
        f"[UPLOAD] Remote project={args.project_id} "
        f"split={args.split} count={len(all_records)}"
    )
    for index, record in enumerate(all_records, start=1):
        upload_image(
            record,
            args.project_id,
            api_key,
            args.split,
            args.batch_name,
            args.tag,
            index,
            len(all_records),
            args.upload_retries,
        )
    print("[DONE] Remote upload completed.")


if __name__ == "__main__":
    main()
