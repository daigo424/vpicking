#!/usr/bin/env bash
# pip配布のisaacsimパッケージには標準のisaac-sim.shが無いため、
# 各extscache配下の拡張機能が持つネイティブライブラリ(libomni.usd.so等)が
# 依存する同階層外のライブラリ(libusd_usd.so等)をLD_LIBRARY_PATH経由で解決させる。
# 設定しないと"cannot open shared object file"でomni.usd等の拡張が起動失敗する。
_isaacsim_root="$(python3 -c 'import isaacsim, os; print(os.path.dirname(isaacsim.__file__))' 2>/dev/null)"
if [ -n "$_isaacsim_root" ]; then
    # extscache配下は拡張機能ごとにbin/またはlib/のどちらかにネイティブライブラリを置く。
    # pixi envのlib直下(libgomp.so.1等の汎用ランタイム)も同様にRPATH外から参照されるため含める。
    _isaacsim_lib_dirs=""
    for d in "$_isaacsim_root"/extscache/*/bin "$_isaacsim_root"/extscache/*/bin/deps "$_isaacsim_root"/extscache/*/lib \
             "$_isaacsim_root"/kit "$_isaacsim_root"/kit/kernel/plugins \
             "$_isaacsim_root"/../../..; do
        [ -d "$d" ] && _isaacsim_lib_dirs="$_isaacsim_lib_dirs:$d"
    done
    export LD_LIBRARY_PATH="${_isaacsim_lib_dirs#:}:$LD_LIBRARY_PATH"
    unset _isaacsim_lib_dirs d
fi
unset _isaacsim_root
