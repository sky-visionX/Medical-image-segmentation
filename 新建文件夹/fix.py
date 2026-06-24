#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 tri_trainlist.txt / tri_testlist.txt 中的路径：
将
    /.../datasetheart/case_xxx
改为
    /.../datasetheart/sequences/case_xxx
"""

import argparse
import os
from pathlib import Path


def fix_path_line(line: str, dataset_name: str = "datasetlabelprostate") -> str:
    """
    修复单行路径。
    假设路径中包含 '/datasetheart/case_'，我们在 'datasetheart/' 后插入 'sequences/'
    """
    line = line.strip()
    if not line:
        return line

    # 查找 "/datasetheart/case_" 的位置（兼容不同 datasetheart 名）
    target = f"/{dataset_name}/case_"
    pos = line.find(target)
    if pos == -1:
        # 如果不匹配，原样返回（或可报错）
        print(f"[WARN] 路径不符合预期，跳过: {line}")
        return line

    # 插入 "sequences" 在 datasetheart/ 之后
    # 例如: .../datasetheart/case_001 → .../datasetheart/sequences/case_001
    fixed = line[: pos + len(dataset_name) + 1] + "/sequences" + line[pos + len(dataset_name) + 1 :]
    return fixed


def process_txt_file(file_path: str, dataset_name: str = "datasetlabelprostate", inplace: bool = True):
    """
    处理单个 .txt 文件
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"文件不存在: {file}")

    print(f"[INFO] 处理文件: {file}")

    with open(file, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        fixed = fix_path_line(line, dataset_name)
        new_lines.append(fixed + "\n")

    output_file = file if inplace else file.with_suffix(f".fixed{file.suffix}")
    with open(output_file, "w") as f:
        f.writelines(new_lines)

    print(f"[INFO] 修复完成 → {output_file}")


def main():
    parser = argparse.ArgumentParser(description="修复 list 文件中的 case 路径，插入 /sequences/")

    parser.add_argument(
        "files",
        nargs="+",
        help="要处理的 .txt 文件路径，例如 tri_trainlist.txt tri_testlist.txt"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="datasetlabelprostate",
        help="数据集目录名（默认 'datasetheart'）"
    )
    parser.add_argument(
        "--no-inplace",
        action="store_true",
        help="不覆盖原文件，而是生成 .fixed.txt"
    )

    args = parser.parse_args()

    for txt_file in args.files:
        process_txt_file(
            file_path=txt_file,
            dataset_name=args.dataset_name,
            inplace=not args.no_inplace
        )

    print("[DONE] 所有文件处理完成！")


if __name__ == "__main__":
    main()
