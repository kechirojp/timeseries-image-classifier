# 時系列画像分類器プロジェクト

このプロジェクトは、EfficientNet、NFNet（正規化自由ネットワーク）またはResNetを利用した転移学習による時系列画像分類器を実装しています。多クラス分類タスク向けに最適化されており、ステージ毎差分学習率ファインチューニングもしくは段階的な凍結解除によるファインチューニングを行います。シングルモーダル（画像のみ）とマルチモーダル（画像＋時系列特徴量）の両方に対応しています。

## 機能概要

- 事前学習済みEfficientNet-B4 または NFNet-F0（利用できない場合はResNet18）をベースとした転移学習
- シングルモーダルおよびマルチモーダル学習に対応
- 段階的に凍結解除する効率的なファインチューニング
- TensorBoardによる学習進捗の可視化
- 損失値とF1スコアの両方に基づくモデルチェックポイント保存
- チェックポイントからの学習再開機能 (`main.py` に統合)
- 柔軟な設定システム（環境別YAML設定ファイル）
- 詳細なモデル評価と可視化ツール
- LightGBMを使った特徴量重要度の算出と自動設定反映（`feature_importance.py`）
- Optunaによるハイパーパラメーターの最適化

## F1スコアによる最適化

このプロジェクトでは、モデルの評価と最適化にF1スコアを重視しています。F1スコアを選択した主な理由は以下の通りです：

- **クラス不均衡への対応**: データセットにクラス不均衡が存在する場合、F1スコアはこの問題に対して堅牢な評価指標となります
- **精度と再現率のバランス**: F1スコアは精度(Precision)と再現率(Recall)の調和平均であり、両方のメトリクスのバランスを取ることで、誤検知と見逃しの両方を最小化します
- **チェックポイント保存の基準**: 検証時のF1スコアが向上した場合にモデルチェックポイントを保存することで、単純な損失値だけでなく実際の予測性能が向上したモデルを選択できます
- **ハイパーパラメータ最適化**: Optunaによるハイパーパラメータ探索でも、F1スコアを最大化することを目標関数としています

特に以下の機能でF1スコアが重要な役割を果たしています：

1. **モデルチェックポイント保存**:
   ```
   checkpoints/{model_mode}/{model_architecture_name}/epoch={epoch:05d}-val_loss={val_loss:.4f}-val_f1={val_f1:.4f}.ckpt
   ```
   形式のファイル名で、F1スコアの向上を確認できます

2. **早期停止**:
   検証F1スコアが`early_stopping_patience`エポックの間改善しない場合、過学習を防ぐために学習を停止します

3. **特徴量重要度評価**:
   LightGBMによる特徴量重要度分析でも、F1スコアを最大化するハイパーパラメータを選定し、最適な特徴量セットを特定します

## 必要なパッケージ

- PyTorch
- PyTorch Lightning
- TorchVision
- TorchMetrics
- PyYAML
- TensorBoard
- scikit-learn（評価・可視化用）
- matplotlib（可視化用）
- LightGBM (特徴量重要度分析用)
- Optuna (特徴量重要度分析用)

## 使い方

### プロジェクトのセットアップ

1. リポジトリをクローンします
2. 依存パッケージをインストールします: `pip install -r requirements.txt`
3. 設定ファイルを編集します:
    - ローカル環境: `configs/config.yaml`
    - Google Colab環境: `configs/config_for_google_colab.yaml`
    - `model_mode` ('single' or 'multi') と `model_architecture_name` を設定してください。

### 学習の実行

以下のコマンドで学習を開始します：

```bash
python main.py
```

### チェックポイントからの学習再開

学習を再開するには、設定ファイル (`config.yaml` または `config_for_google_colab.yaml`) の `resume_from_checkpoint` キーに、再開したいチェックポイントのファイル名（例: `last.ckpt` や `epoch=... .ckpt`）を指定し、通常通り `main.py` を実行します。

```yaml
# 例: config.yaml 内
# --- デバッグ・その他設定 ---
# ...
resume_from_checkpoint: last.ckpt # または 'epoch=00051-val_loss=0.7755-val_f1=0.6688.ckpt' など
```

```bash
python main.py
```
`main.py` は指定されたチェックポイントを自動的に `checkpoints/{model_mode}/{model_architecture_name}/` ディレクトリから探し、学習を再開します。

### Google Colabでの実行

Google Colab環境でモデルの訓練を行うには、以下のノートブックを使用します：

```bash
feature_analysis/colab_runner_current.ipynb
```

このノートブックは以下の手順を自動化します：
- Google Driveのマウント
- 必要なライブラリのインストール
- 設定ファイル (`configs/config_for_google_colab.yaml`) の調整
- 訓練スクリプト (`main.py`) の実行
- チェックポイントからの学習再開 (設定ファイルで `resume_from_checkpoint` を指定)
- TensorBoardによる結果の可視化
- モデルの評価と予測の可視化

### データチェックモード

データセットの形状を確認するには、設定ファイルで `check_data: true` を設定してください。

### モデルの評価と可視化

学習済みモデルの詳細な評価と可視化を行うには、専用のスクリプトを使用します：

```bash
# モデルの評価 (適切な設定ファイルとチェックポイントを指定する必要がある場合があります)
# python src/evaluate.py --mode evaluate --config configs/config.yaml --checkpoint checkpoints/single/nfnet/best.ckpt

# 予測結果の可視化
# python src/evaluate.py --mode visualize ...

# 推論過程の可視化
# python src/evaluate.py --mode reasoning ...

# すべての評価と可視化を実行
# python src/evaluate.py --mode all ...
```
**注意:** `src/evaluate.py` は現在のプロジェクト構成に合わせて更新が必要な場合があります。

### 詳細な分析と可視化

より詳細なモデル分析と可視化を行うには、以下のスクリプトを使用します：

```bash
# 混同行列の表示
# python src/visualize.py --mode confusion ...

# 誤分類サンプルの可視化
# python src/visualize.py --mode misclassified ...

# 特徴重要度の分析
# python src/visualize.py --mode feature ...

# すべての分析を実行
# python src/visualize.py --mode all --save_dir ./analysis_results ...
```
**注意:** `src/visualize.py` は現在のプロジェクト構成に合わせて更新が必要な場合があります。

### マルチモーダルモデル用の特徴量重要度分析

マルチモーダルモデルのための最適な特徴量（時系列数値データなど）を選定するために、特徴量重要度の分析ツールを提供しています：

```bash
# 特徴量重要度の分析と上位特徴量のYAML設定ファイルへの反映
python feature_analysis/feature_analysis.py
```

このスクリプトは以下の処理を実行します：
- LightGBMを使用した特徴量重要度の分析
- 時系列データに適したウォークフォワード検証
- Optunaによるハイパーパラメータの最適化
- 重要度の高い上位の特徴量の抽出
- マルチモーダルモデル設定ファイル (`configs/config.yaml` または `configs/config_for_google_colab.yaml`) 内の `multi_modal_features` セクション（必要に応じて作成）への自動反映

より詳細な使用法については、以下のREADMEを参照してください：
```bash
feature_analysis/README.md
```

## 設定ファイルの説明

### 学習用設定ファイル (`config.yaml`/`config_for_google_colab.yaml`)

- `model_mode`: モデルのモード ('single' または 'multi')
- `model_architecture_name`: モデルのアーキテクチャ名 (例: 'nfnet', 'efficientnet', 'nfnet_transformer')。ログとチェックポイントのサブディレクトリ名に使用されます。
- `base_dir`: プロジェクトのベースディレクトリ (環境に応じて自動設定される場合あり)
- `data_dir`: データセットのディレクトリ
- `logs_dir`: TensorBoardログの **ベース** ディレクトリ (実際のログは `logs/{model_mode}/{model_architecture_name}/` に保存)
- `checkpoint_dir`: モデルチェックポイントの **ベース** ディレクトリ (実際のチェックポイントは `checkpoints/{model_mode}/{model_architecture_name}/` に保存)
- `max_epochs`: 学習するエポック数
- `batch_size`: バッチサイズ
- `precision`: 計算精度設定 ('16-mixed'など)
- `accumulate_grad_batches`: 勾配累積ステップ数
- `early_stopping_patience`: 早期終了の忍耐回数
- `use_progressive_unfreezing`: 段階的凍結解除を使用するかどうかのフラグ
- `auto_unfreeze`: 段階的凍結解除の設定 (use_progressive_unfreezing: true の場合)
  - `stage1_epoch`: 最初の凍結解除を行うエポック
  - `stage2_epoch`: 2番目の凍結解除を行うエポック
  - `stage3_epoch`: 3番目の凍結解除を行うエポック
- `lr_head`, `lr_backbone`, `lr_decay_rate`: 学習率関連の設定
- `scheduler`: 学習率スケジューラの種類
- `use_agc`, `agc_clip_factor`, `agc_eps`: NFNet用のAdaptive Gradient Clipping設定
- `aux_loss_weight`: 補助損失の重み
- `weight_decay`: オプティマイザの重み減衰
- `num_folds`, `fold`: クロスバリデーション用の設定
- `seed`: 乱数シード
- `image_size`: 入力画像のサイズ
- `model`: モデル固有の設定 (例: `type`, `drop_path_rate`)
- `datasets`: 使用するデータセットリスト
- `dataset_a_dir`, `dataset_b_dir`, `dataset_c_dir`: 各データセットのディレクトリ
- `resume_from_checkpoint`: 学習を再開するチェックポイントファイル名 (指定しない場合は `false` またはキー自体を削除)
- `multi_modal_features`: (model_mode: multi の場合) 使用する追加特徴量のリスト (feature_analysis.py で自動設定される可能性あり)

### 特徴量重要度関連の設定ファイル

- 特徴量重要度分析 (`feature_analysis.py`) は、上記の学習用設定ファイル (`config.yaml` または `config_for_google_colab.yaml`) を直接参照・更新します。

## ステージ毎差分学習率ファインチューニング

このプロジェクトでは、ニューラルネットワークの異なる層に対して異なる学習率を適用する「ステージ毎差分学習率ファインチューニング」を実装しています。この手法には以下の特徴があります：

- **層の深さに基づく学習率設定**: 浅い層（初期層）から深い層（後期層）にかけて、異なる学習率を適用します
- **減衰率による制御**: 各ステージの学習率は、前のステージの学習率に対して減衰率（`lr_decay_rate`）を適用して設定
- **最適な知識転移**: 事前学習済みモデルの一般的な特徴を保持しながら、タスク固有の特徴を学習

具体的な設計：
1. **分類器ヘッド**: 最も高い学習率（`lr_head`）を適用し、タスク固有の出力を素早く学習
2. **最深層（Layer 4）**: 基本の学習率（`lr_backbone`）を適用
3. **中間層（Layer 3）**: 基本学習率 × 減衰率
4. **低レベル層（Layer 2）**: 基本学習率 × 減衰率²
5. **最初の層（Layer 1）**: 基本学習率 × 減衰率³

この方法は、以下のような利点があります：
- **転移学習の効率化**: 低レベルの特徴（エッジや色の検出など）は多くのタスクで共通しているため、低い学習率で微調整
- **過学習の防止**: 深い層では高い学習率を、浅い層では低い学習率を適用することで、モデル全体の過学習を抑制
- **学習の安定化**: 層ごとに適切な学習率を設定することで、勾配爆発や消失を防止し、より安定した学習を実現

設定ファイルで以下のパラメータを調整できます：
- `lr_head`: 分類器ヘッドの学習率（デフォルト: 3e-4）
- `lr_backbone`: バックボーンネットワークの基本学習率（デフォルト: 3e-5）
- `lr_decay_rate`: 層間の学習率減衰率（デフォルト: 0.1）

## 段階的ファインチューニング

- **Stage 1 (`stage1_epoch`エポック目)**: 最終ステージ（Layer 4）を解凍
- **Stage 2 (`stage2_epoch`エポック目)**: 中間ステージ（Layer 3）を解凍
- **Stage 3 (`stage3_epoch`エポック目)**: 低レベルステージ（Layer 2）を解凍

## モデル構造

このプロジェクトでは以下の構造を持つ分類モデルを実装しています：

1. **特徴抽出部**: 事前学習済みのNFNet-F0またはResNet18 (シングルモーダル) / 画像特徴抽出部 + 他の特徴量処理部 (マルチモーダル)
2. **推論ヘッド**: 特徴から中間表現を生成
3. **分類器**: 特徴と中間表現を組み合わせて最終分類を行う

## チェックポイント

学習中に以下の種類のチェックポイントが `checkpoints/{model_mode}/{model_architecture_name}/` ディレクトリに保存されます：

1. **F1スコアベース**: 検証F1スコアが最大のモデル (`epoch={epoch:05d}-val_loss={val_loss:.4f}-val_f1={val_f1:.4f}.ckpt`)
2. **最終エポック**: 最後のエポックのモデル (`last.ckpt`)

## TensorBoardの使用方法

### TensorBoardの起動

以下のコマンドでTensorBoardを起動し、学習の進捗状況を可視化できます。`{model_mode}` と `{model_architecture_name}` は、設定ファイルで指定した値に置き換えてください。

```bash
# 例: シングルモーダル、NFNetの場合 (ローカル)
tensorboard --logdir="./logs/single/nfnet"

# 例: マルチモーダル、NFNet+Transformerの場合 (Colab)
# tensorboard --logdir="/content/drive/MyDrive/Time_Series_Classifier/logs/multi/nfnet_transformer"
```

または相対パスを使用する場合：

```bash
# プロジェクトルートディレクトリに移動してから実行
cd /path/to/your/Time_Series_Classifier
# 例: シングルモーダル、NFNetの場合
tensorboard --logdir=logs/single/nfnet
```

### TensorBoardで確認できる情報

- **スカラー**: 学習・検証損失、F1スコア、学習率の推移
- **画像**: （設定によっては）入力データやモデルの注目領域の可視化
- **グラフ**: モデルのネットワーク構造
- **分布**: モデルの重みやバイアスの分布
- **ヒストグラム**: 勾配やアクティベーションの分布

### 重要なメトリクス

- `train_loss`, `val_loss`: 学習・検証損失
- `train_f1`, `val_f1`: 学習・検証F1スコア
- `learning_rate`: 学習率の変化

## トラブルシューティング

- **GPUメモリエラー**: `batch_size`を小さくするか、`accumulate_grad_batches` を増やしてみてください。`precision: '16-mixed'` の使用も有効です。
- **NFNetがロードできない**: TorchVisionのバージョンが古い可能性があります。最新版に更新するか、自動的にResNet18へのフォールバックが行われます。
- **学習が収束しない**: 学習率 (`lr_head`, `lr_backbone`) や `weight_decay` を調整してみてください。
- **Windows環境でのエラー**: `num_workers`を0に設定してください（設定ファイルでデフォルト0になっています）。
- **チェックポイントが見つからない**: `resume_from_checkpoint` で指定したファイル名が正しいか、また `checkpoints/{model_mode}/{model_architecture_name}/` ディレクトリ内に存在するか確認してください。

## 参考文献

- [NFNets論文](https://arxiv.org/abs/2102.06171)
- [PyTorch Lightning ドキュメント](https://pytorch-lightning.readthedocs.io/)
- [TensorBoard ドキュメント](https://www.tensorflow.org/tensorboard/)

## データセットのフォルダ構成

このプロジェクトでは、時系列画像分類のための特定のフォルダ構成を使用します。

### 基本的なフォルダ構造

```
プロジェクトルート/
├── data/
│   ├── dataset_a_15m_winsize40/
│   │   ├── train/
│   │   │   ├── Class_A/
│   │   │   │   ├── dataset_a_15m_20240101_0900_label_0.png
│   │   │   │   ├── dataset_a_15m_20240101_0915_label_0.png
│   │   │   │   └── ...
│   │   │   ├── Class_B/
│   │   │   │   ├── dataset_a_15m_20240101_1000_label_1.png
│   │   │   │   └── ...
│   │   │   └── Class_C/
│   │   │       ├── dataset_a_15m_20240101_1100_label_2.png
│   │   │       └── ...
│   │   ├── val/
│   │   │   ├── Class_A/
│   │   │   ├── Class_B/
│   │   │   └── Class_C/
│   │   └── test/
│   │       ├── Class_A/
│   │       ├── Class_B/
│   │       └── Class_C/
│   ├── dataset_b_15m_winsize40/
│   │   └── (同様の構造)
│   ├── dataset_c_15m_winsize40/
│   │   └── (同様の構造)
│   └── fix_labeled_data_dataset_a_15m.csv  # ラベルデータ (マルチモーダル用)
```

### ファイル命名規則

#### 画像ファイル
画像ファイルは以下の命名規則に従います：
```
{dataset_name}_{timeframe}_{YYYYMMDD}_{HHMM}_label_{class_id}.png
```

**例:**
- `dataset_a_15m_20240101_0900_label_0.png` → Class_A（ラベル0）
- `dataset_a_15m_20240101_0915_label_1.png` → Class_B（ラベル1）
- `dataset_a_15m_20240101_0930_label_2.png` → Class_C（ラベル2）

#### 時系列データファイル（マルチモーダル用）
```
{dataset_name}_{timeframe}_{YYYYMMDD}{HHMM}.csv
```

**例:**
- `dataset_a_15m_202412301431.csv` → 2024年12月30日14:31のデータ

### 設定ファイルでのパス指定

#### ローカル環境 (`configs/config.yaml`)
```yaml
# データディレクトリ設定
data_dir: "./data"

# 各データセットのディレクトリ
dataset_a_dir: "./data/dataset_a_15m_winsize40"
dataset_b_dir: "./data/dataset_b_15m_winsize40"
dataset_c_dir: "./data/dataset_c_15m_winsize40"

# 使用するデータセット
datasets: ["dataset_a", "dataset_b", "dataset_c"]
```

#### Google Colab環境 (`configs/config_for_google_colab.yaml`)
```yaml
# データディレクトリ設定
data_dir: /content/drive/MyDrive/Time_Series_Classifier/data

# 各データセットのディレクトリ
dataset_a_dir: /content/drive/MyDrive/Time_Series_Classifier/data/dataset_a_15m_winsize40
dataset_b_dir: /content/drive/MyDrive/Time_Series_Classifier/data/dataset_b_15m_winsize40
dataset_c_dir: /content/drive/MyDrive/Time_Series_Classifier/data/dataset_c_15m_winsize40
```

### マルチモーダル用の時系列データ

マルチモーダルモードを使用する場合、画像と併せて数値特徴量を使用できます：

```yaml
# マルチモーダル設定
model_mode: "multi"

# 時系列データ設定
timeseries:
  data_path: "./data/fix_labeled_data_dataset_a_15m.csv"
  feature_columns: ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "feature_6"]
  window_size: 40
```

### クラス設定

```yaml
# クラス設定
num_classes: 3
class_names: ["Class_A", "Class_B", "Class_C"]
```

### データセットの準備手順

1. **画像データの準備**:
   - 時系列チャートやグラフ画像を準備
   - 上記の命名規則に従ってファイル名を設定
   - クラス別にフォルダに分類

2. **時系列数値データの準備**（マルチモーダル用）:
   - CSVファイルで数値特徴量を準備
   - 日時インデックスと画像ファイルのタイムスタンプを対応させる

3. **フォルダ構造の作成**:
   - train/val/test分割を実施
   - 各分割内でクラス別フォルダを作成

4. **設定ファイルの更新**:
   - データパスを実際のフォルダ構造に合わせて設定

この構成により、シングルモーダル（画像のみ）とマルチモーダル（画像+数値特徴量）の両方の学習が可能になります。
