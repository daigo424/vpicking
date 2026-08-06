#!/usr/bin/env python3
"""既存のdata/<camera>/dataset/<version>/{images,labels,pose_gt.json}をtrain/valに分割し、
dataset.yamlを生成する。--dataset-dirを複数指定すると、由来の異なるデータセット
(例: 自然なピック軌道とjitter補完)を1つの学習データとして結合できる。

物体がほぼ静止したまま連続撮影される区間があるため、frame単位でシャッフル分割すると
ほぼ同一画像がtrain/val両方に混入してリーク(検証指標が意味を持たなくなる)する。
pose_gt.jsonの物体姿勢を丸めてグループ化し、同一姿勢とみなせるフレーム群を
1単位としてtrain/valどちらか一方にまとめて割り当てる。
"""

import argparse
import json
import os
import random
import shutil

from common.target_object_shape import dataset_yaml_content
from common.versioning import next_version_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        nargs="+",
        help="data/<camera>/dataset/<script>/<version>のパス(複数指定で結合)",
    )
    parser.add_argument(
        "--output-dir",
        help="train.txt/val.txt/dataset.yamlの出力先。省略時、単数指定なら--dataset-dir自身に出力し、"
        "複数指定なら各<script>を+で連結した名前の下に新バージョンを自動採番する",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _default_combined_output_dir(dataset_dirs: list[str]) -> str:
    # 各dataset_dirはdata/<camera>/dataset/<script>/<version>の形式を前提とする。
    # <script>部分を+で連結した名前のディレクトリの下に、結合結果として新バージョンを
    # 採番する(例: dataset/picking_session/v3 + dataset/jitter/v4
    # -> dataset/picking_session+jitter/v1)。
    scripts: list[str] = []
    for d in dataset_dirs:
        script = os.path.basename(os.path.dirname(d.rstrip("/")))
        if script not in scripts:
            scripts.append(script)
    dataset_root = os.path.dirname(os.path.dirname(dataset_dirs[0].rstrip("/")))
    combined_dir = os.path.join(dataset_root, "+".join(scripts))
    return os.path.join(combined_dir, next_version_dir(combined_dir))


def main():
    args = parse_args()
    if args.output_dir:
        output_dir = args.output_dir
    elif len(args.dataset_dir) == 1:
        output_dir = args.dataset_dir[0]
    else:
        output_dir = _default_combined_output_dir(args.dataset_dir)

    # 同じframe名(例: frame_0000)が由来の異なるdataset-dir間で重複しうるため、
    # グループ化・出力のどちらもframe名単体ではなく(dataset_dir, frame名)の組で扱う。
    groups: dict[tuple, list[tuple[str, str]]] = {}
    for dataset_dir in args.dataset_dir:
        with open(os.path.join(dataset_dir, "pose_gt.json")) as f:
            pose_gt = json.load(f)
        for entry in pose_gt:
            key = tuple(round(v, 3) for v in entry["position_xyz"] + entry["orientation_xyzw"])
            groups.setdefault(key, []).append((dataset_dir, entry["frame"]))

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

    os.makedirs(output_dir, exist_ok=True)

    # 結合出力(output_dirが元のdataset-dirと別の場所)の場合、object_3d_keypoints.jsonは
    # そこに存在しないため、後段のyolo_pose.py(--dataと同じディレクトリから読む)が
    # 見つけられなくなる。全dataset-dirで同じ物体形状のはずなので、先頭から1つコピーする。
    keypoints_dst = os.path.join(output_dir, "object_3d_keypoints.json")
    keypoints_src = os.path.join(args.dataset_dir[0], "object_3d_keypoints.json")
    if os.path.exists(keypoints_src) and not os.path.exists(keypoints_dst):
        shutil.copy(keypoints_src, keypoints_dst)

    train_txt = os.path.join(output_dir, "train.txt")
    val_txt = os.path.join(output_dir, "val.txt")
    with open(train_txt, "w") as f:
        f.write(
            "\n".join(os.path.join(d, "images", f"{name}.png") for d, name in sorted(train_frames)) + "\n"
        )
    with open(val_txt, "w") as f:
        f.write("\n".join(os.path.join(d, "images", f"{name}.png") for d, name in sorted(val_frames)) + "\n")

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    content = dataset_yaml_content(output_dir).replace("train: images\n", "train: train.txt\n").replace(
        "val: images\n", "val: val.txt\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)

    print(
        f"{len(ordered_groups)}グループ(姿勢) / train: {len(train_frames)}枚, "
        f"val: {len(val_frames)}枚 -> {yaml_path}を生成しました"
    )


if __name__ == "__main__":
    main()
