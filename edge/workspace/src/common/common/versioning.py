#!/usr/bin/env python3
"""data/<camera>/{dataset,train-models}/<script>/vN形式のバージョンディレクトリの採番。

split_dataset.py(結合データセットの出力先採番)とyolo_pose.py(学習結果の出力先採番)の
両方が同じロジックを必要とするため、ここに集約する(重複させると片方だけ直して
食い違う不具合を起こしやすい)。
"""

import os
import re


def next_version_dir(*version_dirs: str) -> str:
    # 複数ディレクトリ(例: train-models/<script>とmodels/<script>)を跨いで最大値+1を
    # 採番し、番号の意味が競合しないようにする用途もあるため、可変長引数で受け取る。
    numbers = []
    for d in version_dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = re.fullmatch(r"v(\d+)", name)
            if m and os.path.isdir(os.path.join(d, name)):
                numbers.append(int(m.group(1)))
    return f"v{max(numbers, default=0) + 1}"
