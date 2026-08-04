#!/usr/bin/env python3
"""cube_only.py/picking_session.pyが生成したデータセットでYOLO11-Poseを学習する。

学習結果一式(best.pt・object_3d_keypoints.json)は、実行のたびに自動採番される
data/<camera>/train-models/<VERSION>/(git管理外・ローカルのみ)に保存される。気に入った結果は
`make model-promote CAMERA=<camera> VER=<VERSION>`でdata/<camera>/models/<VERSION>/
(git管理下・push対象)へ昇格させる。
"""

import argparse
import os
import re
import shutil

from ultralytics import YOLO


def _next_version(*version_dirs: str) -> str:
    # data/<camera>/models/には既に手動で昇格させたバージョンが存在しうるため、
    # 両ディレクトリを合わせて最大値+1を採番し、番号の意味が競合しないようにする。
    numbers = []
    for d in version_dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = re.fullmatch(r"v(\d+)", name)
            if m and os.path.isdir(os.path.join(d, name)):
                numbers.append(int(m.group(1)))
    return f"v{max(numbers, default=0) + 1}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, choices=["overhead-camera", "wrist-camera"])
    parser.add_argument("--data", required=True, help="dataset.yamlのパス")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", default="yolo11n-pose.pt", help="学習の起点にする事前学習済みモデル")
    parser.add_argument(
        "--project",
        default=None,
        help="Ultralytics自身の作業ディレクトリ(学習ログ等の置き場、既定は/data/<camera>/created。"
        "最終的な結果はdata/<camera>/train-models/へ別途コピーされる)",
    )
    parser.add_argument("--name", default="model")
    return parser.parse_args()


def main():
    args = parse_args()
    camera_base = f"/data/{args.camera}"
    train_models_dir = os.path.join(camera_base, "train-models")
    models_dir = os.path.join(camera_base, "models")
    project = args.project or os.path.join(camera_base, "created")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        project=project,
        name=args.name,
        exist_ok=True,
    )
    # 事前学習済みモデルによっては埋め込みメタデータの影響でargs.nameが無視され、
    # 実際の保存先が<project>/<name>と一致しないことがあるため、trainerが実際に
    # 解決した保存先(save_dir)を使う(<project>/<name>を前提に組み立てない)。
    save_dir = str(model.trainer.save_dir)

    version = _next_version(train_models_dir, models_dir)
    version_dir = os.path.join(train_models_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    shutil.copy(os.path.join(save_dir, "weights", "best.pt"), os.path.join(version_dir, "best.pt"))
    keypoints_src = os.path.join(os.path.dirname(args.data), "object_3d_keypoints.json")
    if os.path.exists(keypoints_src):
        shutil.copy(keypoints_src, os.path.join(version_dir, "object_3d_keypoints.json"))
    print(f"学習結果を {train_models_dir}/{version}/ に保存しました")
    print(f"(気に入ればmake model-promote CAMERA={args.camera} VER={version}でdata/{args.camera}/models/へ昇格できます)")


if __name__ == "__main__":
    main()
