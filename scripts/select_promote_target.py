#!/usr/bin/env python3
"""make model-promoteでCAMERA・VERが未指定の時に、昇格対象を対話式で選ばせ、
"<camera> <script>/<version>"の形でstdoutに出力する。

select_train_target.sh(学習対象の選択)と同じ構造だが、data/<camera>/train-models/配下は
<script>/<version>とネストしているため、収集方式(<script>)とバージョンの2段階で選ばせる。
選択メニュー・プロンプトは標準エラーに出すため、呼び出し側で$(...)からstdoutだけ拾っても
選択画面は表示される。
"""

import argparse
import os
import sys


def _prompt_choice(label: str, options: list[str]) -> str:
    print(f"{label}を選んでください:", file=sys.stderr)
    for i, option in enumerate(options, start=1):
        print(f"  {i}) {option}", file=sys.stderr)
    while True:
        # input()に渡した引数(プロンプト文字列)は標準入力がリダイレクトされていなくても
        # 常にstdoutへ書かれる仕様のため、呼び出し側が$(...)でstdoutだけを拾うと
        # このプロンプト文字列まで結果に混ざる。プロンプトはstderrへ自前で出し、
        # input()自体には何も渡さない。
        print(f"{label}の番号: ", end="", flush=True, file=sys.stderr)
        choice = input()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("番号を選んでください", file=sys.stderr)


def _subdirs(parent: str) -> list[str]:
    if not os.path.isdir(parent):
        print(f"{parent} が見つかりません", file=sys.stderr)
        sys.exit(1)
    names = sorted(d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d)))
    if not names:
        print(f"{parent} にディレクトリが見つかりません", file=sys.stderr)
        sys.exit(1)
    return names


def _version_sort_key(version: str) -> int:
    return int(version[1:]) if version[1:].isdigit() else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camera", nargs="?", default="", help="overhead-camera/wrist-camera(省略時は対話選択)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = args.camera or _prompt_choice("昇格対象のカメラ", ["overhead-camera", "wrist-camera"])

    train_models_dir = f"data/{camera}/train-models"
    script = _prompt_choice("学習データの収集方式", _subdirs(train_models_dir))

    script_dir = os.path.join(train_models_dir, script)
    versions = sorted(_subdirs(script_dir), key=_version_sort_key)
    version = _prompt_choice("昇格するバージョン", versions)

    print(f"{camera} {script}/{version}")


if __name__ == "__main__":
    main()
