#!/usr/bin/env bash
# make vp-pose-estimation/vp-run-yoloでVERが未指定の時に、モデルの取得元
# (data/train-models/: ローカルのみ・push前の学習直後の結果 / data/models/: git管理下の
# push済みモデル)とバージョンを対話式で選ばせる。selectのメニュー・プロンプトは
# 標準エラーに出るため、呼び出し側で$(...)からstdoutだけを拾っても選択画面は表示される。
set -euo pipefail

PS3="モデルの取得元を選んでください: "
select source in "train-models(ローカルのみ・push前)" "models(git管理・push済み)"; do
    case "${REPLY:-}" in
        1) DIR="data/train-models"; PREFIX="train:"; break ;;
        2) DIR="data/models"; PREFIX=""; break ;;
        *) echo "番号を選んでください" >&2 ;;
    esac
done

if [ ! -d "$DIR" ]; then
    echo "$DIR が見つかりません" >&2
    exit 1
fi
mapfile -t versions < <(cd "$DIR" && ls -1 -d */ 2>/dev/null | sed 's#/$##')
if [ "${#versions[@]}" -eq 0 ]; then
    echo "$DIR にバージョンディレクトリが見つかりません" >&2
    exit 1
fi

PS3="使用するモデルバージョンを選んでください: "
select chosen in "${versions[@]}"; do
    if [ -n "${chosen:-}" ]; then
        break
    fi
    echo "番号を選んでください" >&2
done

if [ -z "${chosen:-}" ]; then
    echo "選択されませんでした(標準入力がない環境で実行された可能性があります)" >&2
    exit 1
fi
echo "${PREFIX}${chosen}"
