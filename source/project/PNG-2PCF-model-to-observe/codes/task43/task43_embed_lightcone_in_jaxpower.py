#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Task4.3 lightcone 几何图矢量嵌入 jaxpower corner 的右上空白区。

这个脚本使用 PyMuPDF 的 ``show_pdf_page``，因此源 lightcone PDF 不会先被
栅格化。默认原位更新会议版 ``task43_jaxpower.pdf``，同时刷新对应的
provenance JSON；也可以用 ``--output`` 先生成临时候选文件做视觉检查。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz


PROJECT_ROOT = Path("/pscratch/sd/l/lzy/PNG-2PCF-model-to-observe")
DEFAULT_TARGET = PROJECT_ROOT / "plots/7.13meeting/task43_jaxpower.pdf"
DEFAULT_INSET = (
    PROJECT_ROOT / "plots/7.13meeting/task43_abacus_box_lightcone_geometry.pdf"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs/task43_outputs/summary/task43_meeting_compact_contours.json"
)

# 用目标页宽高的比例定义空白第一象限：实际嵌入时保持源页 1:1 比例。
# 稍微向上下扩展嵌入框，吃掉 contour 页面在第一象限留下的多余白边；
# 右下 b1 marginal 顶边仍保留安全间隔。
INSET_RECT_FRACTIONS = (0.540, 0.005, 0.990, 0.555)
# 裁掉源 PDF 四周未承载信息的白边，使可见 lightcone 内容在相同空白区内
# 放大 15%，同时不侵入右下 b1 posterior。
VISIBLE_CONTENT_SCALE = 1.15
SOURCE_CROP_FRACTION = (1.0 - 1.0 / VISIBLE_CONTENT_SCALE) / 2.0


def parse_args() -> argparse.Namespace:
    """解析输入、输出和 provenance 路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--inset", type=Path, default=DEFAULT_INSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--variant-key", default="jaxpower")
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="不刷新 provenance；用于 /tmp 候选图视觉检查。",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """流式计算文件哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed_pdf_page(input_pdf: Path, inset_pdf: Path, output_pdf: Path) -> dict[str, Any]:
    """把 inset 第 1 页作为 PDF Form XObject 覆盖到目标页右上角。"""
    if not input_pdf.is_file() or not inset_pdf.is_file():
        raise FileNotFoundError(f"Missing input PDF: {input_pdf} or {inset_pdf}")

    target = fitz.open(input_pdf)
    source = fitz.open(inset_pdf)
    try:
        if target.page_count != 1 or source.page_count < 1:
            raise ValueError("Expected a one-page target and a non-empty inset PDF")
        page = target[0]
        width = float(page.rect.width)
        height = float(page.rect.height)
        x0, y0, x1, y1 = INSET_RECT_FRACTIONS
        destination = fitz.Rect(x0 * width, y0 * height, x1 * width, y1 * height)
        source_page = source[0]
        source_width = float(source_page.rect.width)
        source_height = float(source_page.rect.height)
        source_clip = fitz.Rect(
            SOURCE_CROP_FRACTION * source_width,
            SOURCE_CROP_FRACTION * source_height,
            (1.0 - SOURCE_CROP_FRACTION) * source_width,
            (1.0 - SOURCE_CROP_FRACTION) * source_height,
        )
        xref = page.show_pdf_page(
            destination,
            source,
            0,
            keep_proportion=True,
            overlay=True,
            clip=source_clip,
        )

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        if output_pdf.resolve() == input_pdf.resolve():
            temporary = output_pdf.with_name(f".{output_pdf.stem}.lightcone.tmp.pdf")
        else:
            temporary = output_pdf
        target.save(temporary, garbage=4, deflate=True, clean=True)
    finally:
        source.close()
        target.close()

    if output_pdf.resolve() == input_pdf.resolve():
        os.replace(temporary, output_pdf)

    return {
        "destination_rect_pt": [
            float(destination.x0),
            float(destination.y0),
            float(destination.x1),
            float(destination.y1),
        ],
        "destination_rect_fractions": list(INSET_RECT_FRACTIONS),
        "source_clip_rect_pt": [
            float(source_clip.x0),
            float(source_clip.y0),
            float(source_clip.x1),
            float(source_clip.y1),
        ],
        "source_crop_fraction_each_side": float(SOURCE_CROP_FRACTION),
        "visible_content_scale": VISIBLE_CONTENT_SCALE,
        "pdf_xobject_xref": int(xref),
        "keep_proportion": True,
        "vector_embedding": True,
    }


def update_manifest(
    manifest_path: Path,
    output_pdf: Path,
    inset_pdf: Path,
    placement: dict[str, Any],
    *,
    variant_key: str = "jaxpower",
) -> None:
    """刷新 jaxpower variant 的最终 PDF 哈希和 inset provenance。"""
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    variants = manifest.get("variants", [])
    variant = next((item for item in variants if item.get("key") == variant_key), None)
    if variant is None:
        raise KeyError(f"Could not find variant {variant_key!r} in manifest")
    variant["style_audit"]["lightcone_inset"] = {
        **placement,
        "source_pdf": str(inset_pdf),
        "source_pdf_sha256": sha256(inset_pdf),
    }
    variant["output"]["pdf"] = str(output_pdf)
    variant["output"]["pdf_sha256"] = sha256(output_pdf)
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    """完成矢量嵌入，并在正式输出模式下更新 provenance。"""
    args = parse_args()
    placement = embed_pdf_page(args.input, args.inset, args.output)
    if not args.skip_manifest:
        update_manifest(
            args.manifest,
            args.output,
            args.inset,
            placement,
            variant_key=args.variant_key,
        )
    print(f"saved: {args.output}")
    print(f"inset rect (pt): {placement['destination_rect_pt']}")


if __name__ == "__main__":
    main()
