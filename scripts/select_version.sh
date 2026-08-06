#!/usr/bin/env bash
# make vp-pose-estimation-overhead/vp-run-yoloでVERが未指定の時に、モデルの取得元
# (data/<camera>/train-models/: ローカルのみ・push前の学習直後の結果 /
# data/<camera>/models/: git管理下のpush済みモデル)とバージョンを対話式で選ばせる。
# selectのメニュー・プロンプトは標準エラーに出るため、呼び出し側で$(...)から
# stdoutだけを拾っても選択画面は表示される。
set -euo pipefail

# 俯瞰(粗検出)・手先(精緻検出)のように複数モデルを選ばせる場面で、どちらの選択かを
# 区別できるよう、呼び出し側から任意のラベルを第1引数で渡せるようにする。
LABEL="${1:-}"
if [ -n "$LABEL" ]; then
    echo "=== ${LABEL}のモデルを選択します ===" >&2
fi
CAMERA_DIR="${2:?使い方: select_version.sh <label> <overhead-camera|wrist-camera>}"

PS3="モデルの取得元を選んでください: "
select source in "train-models(ローカルのみ・push前)" "models(git管理・push済み)"; do
    case "${REPLY:-}" in
        1)
            # train-models配下はdata/<camera>/train-models/<script>/<version>のネスト構造
            # (どのデータセット収集方式で学習したかを区別するため)なので、
            # <script>をここで先に選ばせてからバージョン一覧に進む。
            SCRIPT_ROOT="data/$CAMERA_DIR/train-models"
            if [ ! -d "$SCRIPT_ROOT" ]; then
                echo "$SCRIPT_ROOT が見つかりません" >&2
                exit 1
            fi
            mapfile -t scripts < <(cd "$SCRIPT_ROOT" && ls -1 -d */ 2>/dev/null | sed 's#/$##')
            if [ "${#scripts[@]}" -eq 0 ]; then
                echo "$SCRIPT_ROOT にデータセット収集方式ディレクトリが見つかりません" >&2
                exit 1
            fi
            PS3="データセット収集方式を選んでください: "
            select script in "${scripts[@]}"; do
                [ -n "${script:-}" ] && break
                echo "番号を選んでください" >&2
            done
            DIR="$SCRIPT_ROOT/$script"
            PREFIX="train:$script/"
            break
            ;;
        2) DIR="data/$CAMERA_DIR/models"; PREFIX=""; break ;;
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
