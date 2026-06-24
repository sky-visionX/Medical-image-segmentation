#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量将 PNG 从 RGBA/带透明度 转为 RGB（递归处理）。
适用目录结构：datasetlabelprostate/sequences/case_00001/im1.png ...
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image


def has_transparency(im: Image.Image) -> bool:
    """判断图片是否带透明度信息（RGBA/LA/P带transparency）"""
    if im.mode in ("RGBA", "LA"):
        return True
    if im.mode == "P" and "transparency" in im.info:
        return True
    return False


def rgba_to_rgb(im: Image.Image, bg=(0, 0, 0)) -> Image.Image:
    """
    将任意模式图片转为 RGB。
    - 若带透明度：先与背景色合成，再转 RGB
    - 若不带透明度：直接 convert('RGB')
    """
    if has_transparency(im):
        # 统一转 RGBA 再合成背景
        im_rgba = im.convert("RGBA")
        background = Image.new("RGBA", im_rgba.size, (bg[0], bg[1], bg[2], 255))
        composed = Image.alpha_composite(background, im_rgba).convert("RGB")
        return composed
    else:
        # 灰度/其他模式也强制转 RGB
        return im.convert("RGB")


def convert_one_file(src: Path, dst: Path, bg=(0, 0, 0), backup: bool = False, inplace: bool = False) -> bool:
    """
    转换单张图。返回是否做了实际转换/写入。
    """
    try:
        with Image.open(src) as im:
            # 如果已经是 RGB 且没有透明度，可以选择跳过（更快）
            if im.mode == "RGB" and not has_transparency(im):
                if inplace:
                    return False
                else:
                    # 非原地模式：仍然复制一份到目标目录，保持目录结构一致
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    return True

            out = rgba_to_rgb(im, bg=bg)

        # 写文件前确保目录存在
        dst.parent.mkdir(parents=True, exist_ok=True)

        if inplace and backup:
            bak = src.with_suffix(src.suffix + ".bak")  # 例如 im1.png.bak
            if not bak.exists():
                shutil.copy2(src, bak)

        out.save(dst)
        return True

    except Exception as e:
        print(f"[ERROR] 处理失败: {src} -> {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="批量将 RGBA PNG 强制转成 RGB（递归）")
    parser.add_argument("--root", type=str, required=True,
                        help="输入根目录，例如 /path/to/datasetlabelprostate/sequences 或 /path/to/datasetlabelprostate")
    parser.add_argument("--subdir", type=str, default="sequences",
                        help="当 root 指向 datasetlabelprostate 根目录时，实际处理的子目录名（默认 sequences）")
    parser.add_argument("--out_root", type=str, default=None,
                        help="输出根目录（不原地时必须指定或自动生成 root_rgb）")
    parser.add_argument("--inplace", action="store_true",
                        help="原地覆盖写回（会把 PNG 变成 RGB）")
    parser.add_argument("--backup", action="store_true",
                        help="原地覆盖前备份为 *.png.bak（建议开启）")
    parser.add_argument("--bg", type=str, default="0,0,0",
                        help="背景色 RGB，例：0,0,0(黑) 或 255,255,255(白)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    # 允许你传的是 datasetlabelprostate 根目录，也允许直接传 sequences
    if (root / args.subdir).exists():
        seq_root = root / args.subdir
    else:
        seq_root = root

    if not seq_root.exists():
        raise FileNotFoundError(f"找不到目录: {seq_root}")

    bg = tuple(int(x) for x in args.bg.split(","))
    if len(bg) != 3:
        raise ValueError("--bg 必须是 r,g,b 形式，例如 0,0,0")

    # 输出目录策略
    if args.inplace:
        out_root = seq_root
        if args.backup:
            print("[INFO] 原地覆盖 + 备份模式开启：将生成 *.png.bak")
        else:
            print("[WARN] 原地覆盖但未开启备份 --backup（建议开启）")
    else:
        if args.out_root is None:
            out_root = seq_root.parent / (seq_root.name + "_rgb")  # sequences_rgb
        else:
            out_root = Path(args.out_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 非原地模式：输出到 {out_root}")

    png_files = sorted(seq_root.rglob("*.png"))
    if not png_files:
        print(f"[WARN] 未找到 png：{seq_root}")
        return

    total = len(png_files)
    changed = 0
    copied = 0

    for idx, src in enumerate(png_files, 1):
        if args.inplace:
            dst = src
        else:
            # 保持相对路径结构
            rel = src.relative_to(seq_root)
            dst = out_root / rel

        did = convert_one_file(src, dst, bg=bg, backup=args.backup, inplace=args.inplace)
        if did:
            # 原地：表示确实写了RGB（或非原地复制）
            # 这里区分下统计：如果非原地且 src 本来RGB，会走 copy2，也算 copied
            if args.inplace:
                changed += 1
            else:
                # 简单点：统一记为 copied/converted 的数量
                copied += 1

        if idx % 500 == 0 or idx == total:
            print(f"[PROGRESS] {idx}/{total}")

    if args.inplace:
        print(f"[DONE] 原地处理完成：共写入 {changed} 张（含RGBA/非RGB会被强制转RGB；已是RGB可能跳过）")
    else:
        print(f"[DONE] 输出完成：共写入/复制 {copied} 张到 {out_root}")


if __name__ == "__main__":
    main()
