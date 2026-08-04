#!/usr/bin/env python3
"""既存のdata/<camera>/dataset/<version>/{images,labels,pose_gt.json}をtrain/valに分割し、
dataset.yamlを更新する。

物体がほぼ静止したまま連続撮影される区間があるため、frame単位でシャッフル分割すると
ほぼ同一画像がtrain/val両方に混入してリーク(検証指標が意味を持たなくなる)する。
pose_gt.jsonの物体姿勢を丸めてグループ化し、同一姿勢とみなせるフレーム群を
1単位としてtrain/valどちらか一方にまとめて割り当てる。
"""

import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="data/<camera>/dataset/<version>のパス")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.dataset_dir, "pose_gt.json")) as f:
        pose_gt = json.load(f)

    groups: dict[tuple, list[str]] = {}
    for entry in pose_gt:
        key = tuple(round(v, 3) for v in entry["position_xyz"] + entry["orientation_xyzw"])
        groups.setdefault(key, []).append(entry["frame"])

    # 大きいグループから順に、目標比率により遠い側のバケットへ詰めることで、
    # グループサイズが不均一でもtrain/valの枚数比を目標に近づける。
    ordered_groups = sorted(groups.values(), key=len, reverse=True)
    rng = random.Random(args.seed)
    rng.shuffle(ordered_groups)

    total_frames = sum(len(frames) for frames in ordered_groups)
    val_target = total_frames * args.val_ratio

    train_frames, val_frames = [], []
    for frames in ordered_groups:
        if len(val_frames) < val_target:
            val_frames += frames
        else:
            train_frames += frames

    images_dir = os.path.join(args.dataset_dir, "images")
    train_txt = os.path.join(args.dataset_dir, "train.txt")
    val_txt = os.path.join(args.dataset_dir, "val.txt")
    with open(train_txt, "w") as f:
        f.write("\n".join(os.path.join(images_dir, f"{name}.png") for name in sorted(train_frames)) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(os.path.join(images_dir, f"{name}.png") for name in sorted(val_frames)) + "\n")

    yaml_path = os.path.join(args.dataset_dir, "dataset.yaml")
    with open(yaml_path) as f:
        content = f.read()
    content = content.replace("train: images\n", "train: train.txt\n").replace("val: images\n", "val: val.txt\n")
    with open(yaml_path, "w") as f:
        f.write(content)

    print(
        f"{len(ordered_groups)}グループ(姿勢) / train: {len(train_frames)}枚, "
        f"val: {len(val_frames)}枚 -> {yaml_path}を更新しました"
    )


if __name__ == "__main__":
    main()
