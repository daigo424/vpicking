#!/usr/bin/env bash
# make trainでCAMERA・DATAが未指定の時に、学習対象のカメラとdata/<camera>/dataset/配下の
# バージョンを対話式で選ばせ、"<camera> <dataset.yamlの絶対パス>"の形でstdoutに出力する。
# selectのメニュー・プロンプトは標準エラーに出るため、呼び出し側で$(...)から
# stdoutだけを拾っても選択画面は表示される(select_version.shと同じ構造)。
set -euo pipefail

CAMERA="${1:-}"
if [ -z "$CAMERA" ]; then
    PS3="学習対象のカメラを選んでください: "
    select CAMERA in overhead-camera wrist-camera; do
        [ -n "${CAMERA:-}" ] && break
        echo "番号を選んでください" >&2
    done
fi

DIR="data/$CAMERA/dataset"
if [ ! -d "$DIR" ]; then
    echo "$DIR が見つかりません" >&2
    exit 1
fi
mapfile -t versions < <(cd "$DIR" && ls -1 -d */ 2>/dev/null | sed 's#/$##')
if [ "${#versions[@]}" -eq 0 ]; then
    echo "$DIR にバージョンディレクトリが見つかりません" >&2
    exit 1
fi

# 由来の異なるデータセットを結合したい場合は、事前にmake dataset-splitで
# 結合済みのdataset.yamlを作ってから、そのバージョンをここで選ぶ想定
# (このスクリプト自体は単一バージョンの選択のみ担当する)。
PS3="学習に使うデータセットバージョンを選んでください: "
select chosen in "${versions[@]}"; do
    [ -n "${chosen:-}" ] && break
    echo "番号を選んでください" >&2
done

if [ -z "${chosen:-}" ]; then
    echo "選択されませんでした(標準入力がない環境で実行された可能性があります)" >&2
    exit 1
fi
echo "$CAMERA /data/$CAMERA/dataset/$chosen/dataset.yaml"
