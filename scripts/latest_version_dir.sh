#!/usr/bin/env bash
# 指定したディレクトリ配下のvN形式のサブディレクトリを走査し、最新(最終更新が最も新しい)の
# バージョンを標準出力に返す。data/<camera>/models/のようにvNを直下に持つ構造(深さ1)と、
# data/<camera>/train-models/<script>/vNのように収集方式で1段ネストした構造(深さ2)の
# どちらにも対応し、後者はscript/vNの形式で返す。next_version_dir.sh(次に採番すべき
# 未使用の番号を返す)とは逆に、既に存在する最新版を指す。vp-run-yolo-latest-train/
# vp-run-yolo-latest-modelsが、対話選択(select_version.sh)を省略して自動的に
# 最新モデルを使うために利用する。
set -euo pipefail

DIR="${1:?使い方: latest_version_dir.sh <parent_dir>}"

if [ ! -d "$DIR" ]; then
    echo "$DIR が見つかりません" >&2
    exit 1
fi

# 深さ1・2それぞれのvNディレクトリを最終更新時刻付きで列挙し、最新のものを1件選ぶ。
LATEST=$(find "$DIR" -mindepth 1 -maxdepth 2 -type d -regextype posix-extended -regex '.*/v[0-9]+' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -z "${LATEST:-}" ]; then
    echo "$DIR にバージョンディレクトリが見つかりません" >&2
    exit 1
fi
echo "${LATEST#"$DIR"/}"
