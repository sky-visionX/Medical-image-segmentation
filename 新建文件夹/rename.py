import os
import sys


def add_0000_suffix(images_tr_dir):
    if not os.path.isdir(images_tr_dir):
        print(f"❌ 错误: 目录不存在 {images_tr_dir}")
        sys.exit(1)

    renamed = 0
    for filename in os.listdir(images_tr_dir):
        # 只处理以 .nii.gz 结尾、且不包含 _0000 的文件
        if filename.endswith('.nii.gz') and '_0001.nii.gz' not in filename:
            # 插入 _0000 在 .nii.gz 之前
            name_without_ext = filename[:-7]  # 去掉 .nii.gz
            new_name = f"{name_without_ext}_0001.nii.gz"
            old_path = os.path.join(images_tr_dir, filename)
            new_path = os.path.join(images_tr_dir, new_name)

            if os.path.exists(new_path):
                print(f"⚠️ 跳过（目标已存在）: {new_name}")
                continue

            os.rename(old_path, new_path)
            print(f"✅ 重命名: {filename} → {new_name}")
            renamed += 1

    print(f"\n�� 共重命名 {renamed} 个文件。")


if __name__ == "__main__":
    # 默认路径（请根据你的任务修改）
    default_dir = "/home/dell/wzh/new/InterpAny-Clearer-main/1"

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = default_dir

    add_0000_suffix(target_dir)