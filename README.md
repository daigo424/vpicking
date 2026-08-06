# vpicking

ビジョンピッキングに関する検証プロジェクトです。

| Action Graph | Isaac Sim |
|---|---|
| <img width="1067" height="786" alt="Screenshot from 2026-08-07 00-30-31" src="https://github.com/user-attachments/assets/9d03329a-56af-4dd3-b5ae-f71658df75b3" /> | <video src="https://github.com/user-attachments/assets/1dfa049b-a9f4-48a3-bb77-efc092450c55"></video> |

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

## 運用

### ピッキング実行

`make sim`起動後、別ターミナルで実行。

```bash
make vp-run-gt     # ground truth座標を使ったピッキング(切り分け用)
make vp-run-cv     # Depthカメラ + 古典的CVでのピッキング
make vp-run-yolo   # YOLO11-Pose(俯瞰+手先カメラ)でのピッキング(本番版)
```

`vp-run-yolo`はモデルバージョンを`VER_OVERHEAD`(俯瞰・粗検出)・`VER_WRIST`(手先・精緻検出)で指定する
(省略時は対話選択)。最新版を自動選択する`vp-run-yolo-latest-train`(学習直後・未push)・
`vp-run-yolo-latest-models`(push済み)もある。

### モデルの学習・更新

データは`data/<camera>/dataset/<script>/vN`のように、収集方式(`<script>`)ごとにバージョンを分けて
管理する(由来の異なるデータセットを混同しないため)。`<script>`は`picking_session`(自然なピック軌道)・
`jitter`(手先カメラ用、位置・大きさの多様性を補う)・`cube_only`(俯瞰カメラ用、Pandaアームを
含まない孤立した合成シーン)。

1. **データ収集**(実行のたびに新バージョンが`data/<camera>/dataset/<script>/`に採番される)

   ```bash
   make dataset-collect-overhead-picking-session
   make dataset-collect-wrist-picking-session
   make dataset-collect-wrist-jitter
   make dataset-collect-overhead-cube-only
   ```

2. **train/val分割**(`vN`は1.のコマンド実行時に表示される`出力先: .../vN`のN)

   ```bash
   # 単一バージョンのみ使う場合、分割結果は同じディレクトリ(DATASET_DIR)に書き込まれる
   make dataset-split DATASET_DIR=/data/overhead-camera/dataset/picking_session/v1

   # 複数バージョンをスペース区切りで指定すると結合できる。OUTPUT_DIRを省略すると
   # 各<script>を+で連結した名前(例: picking_session+jitter)の下に自動採番される
   make dataset-split DATASET_DIR="/data/wrist-camera/dataset/picking_session/v1 /data/wrist-camera/dataset/jitter/v1"
   ```

3. **学習**(`CAMERA`・`DATA`省略で対話選択、結果は`data/<camera>/train-models/<script>/`へローカル保存)

   ```bash
   make train CAMERA=overhead-camera DATA=/data/overhead-camera/dataset/picking_session/v1/dataset.yaml
   ```

4. **本番への昇格**(`data/<camera>/models/`へコピー、git管理下。`<script>`分けはしない)

   ```bash
   make model-promote CAMERA=overhead-camera VER=picking_session/v3
   ```

## 現状の最良モデルの作り方

俯瞰カメラは自然なピック軌道のみ、手先カメラは自然な軌道+jitterを結合するのが現状のベスト。
`v<N>`等は各コマンド実行時に表示される実際のバージョン番号に置き換える(詳細は上の「モデルの学習・更新」参照)。

```bash
# 俯瞰カメラ
make dataset-collect-overhead-picking-session
make dataset-split DATASET_DIR=/data/overhead-camera/dataset/picking_session/v<N>
make train CAMERA=overhead-camera DATA=/data/overhead-camera/dataset/picking_session/v<N>/dataset.yaml
make model-promote CAMERA=overhead-camera VER=picking_session/v<採番されたtrain-modelsのバージョン>

# 手先カメラ(自然な軌道 + jitterを結合)
make dataset-collect-wrist-picking-session
make dataset-collect-wrist-jitter
make dataset-split DATASET_DIR="/data/wrist-camera/dataset/picking_session/v<N> /data/wrist-camera/dataset/jitter/v<N>"
make train CAMERA=wrist-camera DATA=/data/wrist-camera/dataset/picking_session+jitter/v<N>/dataset.yaml
make model-promote CAMERA=wrist-camera VER=picking_session+jitter/v<採番されたtrain-modelsのバージョン>
```
