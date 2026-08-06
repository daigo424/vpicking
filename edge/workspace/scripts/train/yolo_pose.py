#!/usr/bin/env python3
"""overhead_camera/cube_only.py/camera/picking_session.pyが生成したデータセットでYOLO11-Poseを学習する。

学習結果一式(best.pt・object_3d_keypoints.json)は、--dataのパス(data/<camera>/dataset/<script>/<VERSION>/
dataset.yaml)から<script>を読み取り、実行のたびに自動採番される
data/<camera>/train-models/<script>/<VERSION>/(git管理外・ローカルのみ)に保存される。
どのデータセット収集方式で学習したモデルかを保存先の時点で追跡できるようにするため。
気に入った結果は`make model-promote CAMERA=<camera> VER=<script>/<VERSION>`で
data/<camera>/models/<VERSION>/(git管理下・push対象、こちらはscript別に分けない)へ昇格させる。
"""

import argparse
import os
import shutil

from common.versioning import next_version_dir
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, choices=["overhead-camera", "wrist-camera"])
    parser.add_argument("--data", required=True, help="dataset.yamlのパス")
    parser.add_argument("--epochs", type=int, default=100)
    # data/yolo/models/にgit管理下で固定しているベースモデルを既定にする(Ultralyticsの
    # リリース更新で同名でも中身が変わりうるため、都度自動ダウンロードに頼らない)。
    parser.add_argument("--model", default="/data/yolo/models/yolo11n-pose.pt", help="学習の起点にする事前学習済みモデル")
    parser.add_argument(
        "--project",
        default=None,
        help="Ultralytics自身の作業ディレクトリ(学習ログ等の置き場、既定は/tmp/vpicking/<camera>/created。"
        "再現性のため保持する必要はない一時生成物なのでdata/ではなくtmp/側に置く。"
        "最終的な結果はdata/<camera>/train-models/へ別途コピーされる)",
    )
    # Ref: ultralytics/cfg/__init__.py の get_cfg() 実装
    # ("if cfg.get('name') == 'model': cfg['name'] = str(cfg.get('model', '')).partition('.')[0]")
    # Ultralyticsは--nameに"model"という文字列を渡すと「モデルファイル自身のパスをnameに使う」
    # 特殊な意味に解釈する。nameがそのモデルの絶対パスに置き換わると、
    # get_save_dir()のPath(project)/nameで絶対パス側が優先され、--projectの指定が無視される。
    parser.add_argument("--name", default="run")
    return parser.parse_args()


def main():
    args = parse_args()
    camera_base = f"/data/{args.camera}"
    # --dataはdata/<camera>/dataset/<script>/<VERSION>/dataset.yamlの形式である前提で、
    # <script>部分を保存先(train-models/<script>/)にそのまま引き継ぐ。
    script = os.path.basename(os.path.dirname(os.path.dirname(args.data)))
    train_models_dir = os.path.join(camera_base, "train-models", script)
    models_dir = os.path.join(camera_base, "models")
    project = args.project or f"/tmp/vpicking/{args.camera}/created"

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        project=project,
        name=args.name,
        exist_ok=True,
    )
    # <project>/<name>から組み立てず、trainerが実際に解決した保存先(save_dir)を使う
    # (--nameの値によってはUltralytics側の特殊処理で実際の保存先が変わりうるため)。
    save_dir = str(model.trainer.save_dir)

    version = next_version_dir(train_models_dir, models_dir)
    version_dir = os.path.join(train_models_dir, version)
    os.makedirs(version_dir, exist_ok=True)
    shutil.copy(os.path.join(save_dir, "weights", "best.pt"), os.path.join(version_dir, "best.pt"))
    keypoints_src = os.path.join(os.path.dirname(args.data), "object_3d_keypoints.json")
    if os.path.exists(keypoints_src):
        shutil.copy(keypoints_src, os.path.join(version_dir, "object_3d_keypoints.json"))
    print(f"学習結果を {train_models_dir}/{version}/ に保存しました")
    print(
        f"(気に入ればmake model-promote CAMERA={args.camera} VER={script}/{version}で"
        f"data/{args.camera}/models/へ昇格できます)"
    )


if __name__ == "__main__":
    main()
