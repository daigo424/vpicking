#!/usr/bin/env bash
# make dataset / make trainから呼ばれる、手法選択用のディスパッチャ。
# edge/workspace/scripts/<category>/配下の.pyを一覧して選ばせ、対応する
# pixiタスク(スクリプトのstem名と同名である前提)をmake colcon経由で実行する。
set -euo pipefail

CATEGORY="${1:?使い方: select_method.sh dataset|train}"
DIR="edge/workspace/scripts/$CATEGORY"

mapfile -t scripts < <(cd "$DIR" && ls -1 *.py)
if [ "${#scripts[@]}" -eq 0 ]; then
    echo "edge/workspace/scripts/$CATEGORY/ にスクリプトが見つかりません" >&2
    exit 1
fi

PS3="実行する${CATEGORY}スクリプトを選んでください: "
select chosen in "${scripts[@]}"; do
    if [ -n "${chosen:-}" ]; then
        break
    fi
    echo "番号を選んでください" >&2
done

if [ -z "${chosen:-}" ]; then
    echo "選択されませんでした(標準入力がない環境で実行された可能性があります)" >&2
    exit 1
fi
task_name="${chosen%.py}"
exec make colcon CMD_RUN="pixi run $task_name"
