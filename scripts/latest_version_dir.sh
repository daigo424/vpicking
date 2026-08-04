#!/usr/bin/env bash
# 指定したディレクトリ配下のvN形式のサブディレクトリを走査し、最新(最大N)のバージョンを
# 標準出力に返す。next_version_dir.sh(次に採番すべき未使用の番号を返す)とは逆に、
# 既に存在する最新版を指す。vp-run-yolo-latest-train/vp-run-yolo-latest-modelsが、
# 対話選択(select_version.sh)を省略して自動的に最新モデルを使うために利用する。
set -euo pipefail

DIR="${1:?使い方: latest_version_dir.sh <parent_dir>}"

if [ ! -d "$DIR" ]; then
    echo "$DIR が見つかりません" >&2
    exit 1
fi

MAX_N=$(ls -1 "$DIR" 2>/dev/null | grep -E '^v[0-9]+$' | sed 's/^v//' | sort -n | tail -1)
if [ -z "${MAX_N:-}" ]; then
    echo "$DIR にバージョンディレクトリが見つかりません" >&2
    exit 1
fi
echo "v$MAX_N"
