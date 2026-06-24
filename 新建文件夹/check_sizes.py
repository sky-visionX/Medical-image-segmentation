#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path
import cv2
from collections import defaultdict, Counter

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_image(p: Path):
    return p.suffix.lower() in IMG_EXTS

def read_size(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None  # 读图失败
    if img.ndim == 2:
        H, W = img.shape
        C = 1
    elif img.ndim == 3:
        H, W, C = img.shape
    else:
        return None
    return (C, H, W)

def main():
    ap = argparse.ArgumentParser(description="Check per-sequence image size consistency")
    ap.add_argument("--root", required=True, help="Dataset root, e.g. ./dataset2/vimeo_triplet")
    ap.add_argument("--depth", type=int, default=1,
                    help="Grouping depth: 1=group by parent, 2=grandparent, etc.")
    ap.add_argument("--min-files", type=int, default=1,
                    help="Only report groups with at least this many files (default: 1)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERR] Root not found: {root}")
        sys.exit(1)

    groups = defaultdict(list)
    total_imgs = 0
    failed_imgs = 0

    for p in root.rglob("*"):
        if not p.is_file() or not is_image(p):
            continue
        total_imgs += 1
        g = p.parent
        for _ in range(args.depth - 1):
            g = g.parent
        size = read_size(p)
        if size is None:
            failed_imgs += 1
        groups[str(g)].append((str(p), size))

    inconsistent_groups = []
    for gpath, items in groups.items():
        sizes = [it[1] for it in items if it[1] is not None]
        size_counter = Counter(sizes)
        is_consistent = (len(size_counter) == 1) and (None not in size_counter)
        if (not is_consistent) and (len(items) >= args.min_files):
            inconsistent_groups.append((gpath, items, size_counter))

    print("\n========== SUMMARY ==========")
    print(f"Scanned images  : {total_imgs}")
    print(f"Failed to read  : {failed_imgs}")
    print(f"Total groups    : {len(groups)}")
    print(f"Inconsistent    : {len(inconsistent_groups)}")
    print("=============================\n")

    if inconsistent_groups:
        print(">>> Inconsistent groups (showing up to first 20 groups):")
        for gi, (gpath, items, size_counter) in enumerate(inconsistent_groups[:20], start=1):
            print(f"\n[{gi}] Group: {gpath}")
            print("    Size distribution:", dict(size_counter))
            for fp, sz in items[:50]:  # 每组最多展示 50 个文件，避免刷屏
                print(f"    {fp} -> {sz}")
            if len(items) > 50:
                print(f"    ... ({len(items)-50} more files)")
    else:
        print("All groups look consistent ✅")

if __name__ == "__main__":
    main()
