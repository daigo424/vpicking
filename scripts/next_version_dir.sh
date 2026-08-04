#!/usr/bin/env bash
# 指定したディレクトリ配下のvN形式のサブディレクトリを走査し、次に採番すべきバージョン
# (v{最大値+1}、無ければv1)を標準出力に返す。model-promote/データ収集スクリプトで
# 同じ採番ロジックが重複しないよう共通化している。
set -euo pipefail

DIR="${1:?使い方: next_version_dir.sh <parent_dir>}"

if [ ! -d "$DIR" ]; then
    echo "v1"
    exit 0
fi

MAX_N=$(ls -1 "$DIR" 2>/dev/null | grep -E '^v[0-9]+$' | sed 's/^v//' | sort -n | tail -1)
if [ -z "${MAX_N:-}" ]; then
    echo "v1"
else
    echo "v$((MAX_N + 1))"
fi
