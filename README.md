# vpicking

ビジョンピッキングに関する検証プロジェクトです。

## 開発環境セットアップ

VSCode + Dev Containers前提(C++補完・ビルド・テスト・ブレークポイントデバッグまでコンテナ内で完結)。

### 1. 前提

- リポジトリはWSL2ネイティブパス(`/home/<user>/...`)にクローンしていること(Windows側マウントは低速)
- VSCode拡張「Dev Containers」(`ms-vscode-remote.remote-containers`)導入済み

### 2. Docker Compose起動

初回は以下のコマンドを実行。

```bash
make build
```

以降は以下でDocker Compose起動。

```bash
make up
```

その後VSCodeでリポジトリを開き、コマンドパレット → `Dev Containers: Reopen in Container`。
コンテナは`make up`で既に起動済みのため、VSCodeはそのままアタッチするだけになる。

補完(clangd)は`colcon build`が生成する`compile_commands.json`を自動検出するため
追加設定不要。

### 3. ビルド

```bash
make colcon-build
```

### 4. Isaac Sim起動

```bash
make sim
```
