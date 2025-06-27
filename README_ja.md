# 時系列画像分類器

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-792EE5?style=flat&logo=PyTorch%20Lightning&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

PyTorch Lightningを使用したEfficientNet/NFNetによる時系列画像分類のプロダクション対応深層学習プロジェクトです。選択式ファインチューニング技術による転移学習を実装し、多クラス分類タスクに対応。シングルモーダル（画像のみ）とマルチモーダル（画像＋時系列特徴量）の両方の学習アプローチを単一の設定ファイルでサポートします。現在はEfficientNet-B4をメインアーキテクチャとして採用。

[日本語版README](README_ja.md) | [English README](README.md)

## 機能概要

- **統合アーキテクチャ**: 単一の設定ファイルで`model_mode`切り替えによるシングルモーダル/マルチモーダル学習
- **モデル選択可能な転移学習**: 事前学習済みEfficientNet-B4（メイン）、NFNet-F0、または ResNet18（フォールバック）による設定可能なファインチューニングアプローチ
- **マルチモーダルサポート**: シングルモーダル（画像のみ）およびマルチモーダル（画像＋数値時系列特徴量）対応
- **選択式ファインチューニング**: ステージ毎差分学習率または段階的凍結解除の選択が可能
- **F1スコア最適化**: 包括的なF1スコアベース評価と早期停止
- **プロダクション対応**: チェックポイントからの学習再開、柔軟なYAML設定システム
- **学習過程の可視化**: TensorBoardによる学習メトリクスの表示
- **特徴量エンジニアリング**: LightGBMベースの特徴量重要度分析と自動設定更新
- **ハイパーパラメータ最適化**: 自動ハイパーパラメータチューニングのためのOptuna統合
- **クロスプラットフォーム**: ローカル開発およびGoogle Colab環境サポート

## アーキテクチャモード

### モード設定
プロジェクトは単一の設定ファイルで両モードをサポート：

```yaml
# シングルモーダルモード（画像のみ）
model_mode: single

# マルチモーダルモード（画像 + 時系列特徴量）
model_mode: multi
```

### シングルモーダルモード（model_mode: single）
- **入力**: 画像データのみ
- **モデル**: EfficientNet/NFNetによる画像分類
- **用途**: 画像チャートパターンのみを使用した予測

### マルチモーダルモード（model_mode: multi）  
- **入力**: 画像データ + 時系列数値特徴量
- **モデル**: 
  - 画像特徴量抽出器（EfficientNet/NFNet）
  - 時系列特徴量処理（Transformer + 分類器）
  - 特徴量融合層
- **用途**: チャートパターンと数値特徴量を組み合わせた高精度予測

## F1スコアによる最適化の理由

このプロジェクトでは、モデルの評価と最適化にF1スコアを重視しています：

- **クラス不均衡への堅牢性**: F1スコアは不均衡データセットに対して堅牢な評価を提供
- **精度と再現率のバランス**: 精度と再現率を調和的にバランスし、偽陽性と偽陰性の両方を最小化
- **性能ベースチェックポイント**: 検証F1スコアの改善に基づいてモデルを保存し、実際の予測性能向上を保証
- **ハイパーパラメータ最適化**: Optuna最適化はF1スコア最大化を目標として最適なモデル選択を実現

### クラス不均衡への対処戦略

データセット内のクラス不均衡に対処するため、以下の戦略を採用：

- **均等サンプリング**: 最小クラスのサンプル数に合わせて、全クラスのサンプル数を統一
- **分割セット間の一貫性**: 学習・検証・テストセット全体で統一されたクラス分布を維持

F1スコアの主要応用：
1. **モデルチェックポイント**: `epoch={epoch:05d}-val_loss={val_loss:.4f}-val_f1={val_f1:.4f}.ckpt`
2. **早期停止**: 検証F1スコアの改善が停止した際の過学習防止
3. **特徴量重要度**: LightGBM分析による最大F1スコアのための特徴量選択最適化

## 必要なパッケージ

- PyTorch
- PyTorch Lightning
- TorchVision
- TorchMetrics
- PyYAML
- TensorBoard
- scikit-learn（評価・可視化用）
- matplotlib（可視化用）
- LightGBM（特徴量重要度分析用）
- Optuna（ハイパーパラメータ最適化用）

## クイックスタート

### セットアップ

1. リポジトリをクローンします
2. **オプションA: ローカルセットアップ**
   - PyTorch（CUDA対応版）をインストール: `pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121`
   - その他の依存パッケージをインストール: `pip install -r requirements.txt`
   - Optuna統合をインストール: `pip install optuna-integration[pytorch_lightning]`
3. **オプションB: Dockerセットアップ**
   - Docker Hubから取得: `docker pull kechiro/timeseries-image-classifier:latest`
   - またはローカルビルド: `./build-docker.sh`
4. 設定ファイルを編集:
   - ローカル環境: `configs/config.yaml`
   - Google Colab環境: `configs/config_for_google_colab.yaml`
   - `model_mode`（'single' または 'multi'）と `model_architecture_name` を設定

### 学習の実行

**ローカル実行:**
```bash
python main.py
```

**Docker実行:**
```bash
# Docker Composeで実行（推奨）
docker-compose up

# または直接実行
docker run --gpus all -it kechiro/timeseries-image-classifier:latest
```

### チェックポイントからの学習再開

学習を再開するには、設定ファイル（`config.yaml` または `config_for_google_colab.yaml`）でチェックポイントファイル名を指定します：

```yaml
# config.yaml 内
resume_from_checkpoint: last.ckpt  # または 'epoch=00051-val_loss=0.7755-val_f1=0.6688.ckpt' （例）
```

その後実行：
```bash
python main.py
```

### Google Colabでの学習

Google Colab環境での学習には、提供されているノートブックを使用します：

```bash
feature_analysis/colab_runner_current.ipynb
```

このノートブックは以下を自動化します：
- Google Driveのマウント
- 必要なライブラリのインストール
- 設定ファイル（`configs/config_for_google_colab.yaml`）の調整
- 学習スクリプト（`main.py`）の実行
- チェックポイントからの学習再開
- TensorBoardによる可視化
- モデル評価と予測の可視化

## データセット構造

プロジェクトは`data/`ディレクトリ内に画像データセットと時系列データを格納します。サンプルデータが含まれているため、すぐに動作確認が可能です。

### ディレクトリ構成

```
data/
├── README.md                              # データ構造の詳細説明
├── fix_labeled_data_timeseries_15m.csv   # 正解ラベルファイル（サンプル）
├── timeseries_15m_202412301431.csv       # 特徴量ファイル（サンプル）
├── dataset_a_15m_winsize40/              # データセットA（画像データ）
│   ├── README.md                         # 画像データ要件
│   ├── train/                            # 訓練用データ
│   │   ├── class_0/                      # クラス0画像（ラベル0）
│   │   ├── class_1/                      # クラス1画像（ラベル1）
│   │   └── class_2/                      # クラス2画像（ラベル2）
│   └── test/                             # テスト用データ
│       ├── class_0/                      # クラス0画像（ラベル0）
│       ├── class_1/                      # クラス1画像（ラベル1）
│       └── class_2/                      # クラス2画像（ラベル2）
├── dataset_b_15m_winsize40/              # データセットB（同様の構造）
│   ├── README.md
│   ├── train/
│   │   ├── class_0/
│   │   ├── class_1/
│   │   └── class_2/
│   └── test/
│       ├── class_0/
│       ├── class_1/
│       └── class_2/
└── dataset_c_15m_winsize40/              # データセットC（同様の構造）
    ├── README.md
    ├── train/
    │   ├── class_0/
    │   ├── class_1/
    │   └── class_2/
    └── test/
        ├── class_0/
        ├── class_1/
        └── class_2/
```

### サンプルデータ

リポジトリには以下のサンプルデータが含まれています：

1. **`fix_labeled_data_timeseries_15m.csv`**: マルチモーダル学習用の正解ラベルデータ
   - timestamp, action, dataset_id カラム
   - アクション分類（0: class_0, 1: class_1, 2: class_2）

2. **`timeseries_15m_202412301431.csv`**: マルチモーダル学習用の数値特徴量データ
   - timestamp, feature_1〜6, dataset_id カラム
   - 6次元の時系列特徴量

3. **`dataset_*/README.md`**: 各データセットディレクトリの画像要件説明
   - 画像形式要件（PNG/JPG、一貫したサイズ）
   - ディレクトリ構造
   - クラス別の説明

### マルチモーダル学習の動作フロー

1. **ラベル取得**: 画像ファイルのディレクトリ構造（`class_0/`, `class_1/`, `class_2/`）からラベルを取得
2. **タイムスタンプ抽出**: 画像ファイル名から日時情報を抽出
3. **特徴量マッチング**: 抽出したタイムスタンプで特徴量CSVから対応する時系列データを取得
4. **マルチモーダル入力**: 画像データ + 時系列特徴量データを組み合わせて学習

### 実データの準備

実際の学習には、以下のデータを準備してください：

1. **画像データ**: 各データセットの`train/`と`test/`ディレクトリに画像を配置
   - ファイル名は`{dataset_name}_15m_{YYYYMMDD}{HHMM}_label_{class_id}.png`形式
2. **特徴量データ**: マルチモーダル学習用の6次元特徴量CSVファイルを`data/`ディレクトリに配置
   - ファイル名は`timeseries_15m_YYYYMMDDHHMM.csv`形式
3. **タイムスタンプ整合性**: 画像ファイル名のタイムスタンプと特徴量CSVのタイムスタンプが一致している必要があります

## 高度な機能

### データ検証

データセット形状の検証を有効にする：

```yaml
check_data: true
```

### 特徴量重要度分析

マルチモーダルモデルの特徴量選択を最適化：

```bash
python feature_analysis/feature_analysis.py
```

このスクリプトは以下を実行：
- LightGBMベースの特徴量重要度分析
- 時系列データ用ウォークフォワード検証
- Optunaハイパーパラメータ最適化
- 上位特徴量の抽出と自動設定更新

詳細な使用方法：`feature_analysis/README.md`

## 設定

### メイン学習設定（`config.yaml`/`config_for_google_colab.yaml`）

主要パラメータ：
- `model_mode`: 'single' または 'multi'
- `model_architecture_name`: アーキテクチャ名（例：'nfnet', 'efficientnet'）
- `max_epochs`: 学習エポック数
- `batch_size`: バッチサイズ
- `precision`: 計算精度（'16-mixed'推奨）
- `early_stopping_patience`: 早期停止の忍耐回数
- `use_progressive_unfreezing`: 段階的凍結解除の有効化
- `lr_head`, `lr_backbone`, `lr_decay_rate`: 学習率設定
- `datasets`: 使用するデータセットリスト
- `resume_from_checkpoint`: 再開用チェックポイントファイル

### 段階的ファインチューニング

ステージ毎差分学習率の実装：

- **分類器ヘッド**: タスク固有出力のための最高学習率（`lr_head`）
- **レイヤー4（最深層）**: ベース学習率（`lr_backbone`）
- **レイヤー3**: ベースLR × 減衰率
- **レイヤー2**: ベースLR × 減衰率²
- **レイヤー1**: ベースLR × 減衰率³

利点：
- **転移学習効率**: 一般的特徴には低い学習率、タスク固有には高い学習率
- **過学習防止**: ネットワーク深度全体でバランスの取れた学習
- **学習安定性**: 勾配爆発・消失の防止

### 段階的凍結解除スケジュール

- **ステージ1（`stage1_epoch`）**: レイヤー4を凍結解除
- **ステージ2（`stage2_epoch`）**: レイヤー3を凍結解除
- **ステージ3（`stage3_epoch`）**: レイヤー2を凍結解除

## モデル構造

分類モデルの構成：

1. **特徴抽出部**: 事前学習済みEfficientNet-B4/NFNet-F0/ResNet18（シングルモーダル）または画像＋数値特徴量の組み合わせ（マルチモーダル）
2. **分類ヘッド**: 特徴量から中間表現を生成し、最終的なクラス分類を行う多層ニューラルネットワーク

## チェックポイント

チェックポイントは `checkpoints/{model_mode}/{model_architecture_name}/` に保存：

1. **F1スコアベース**: 最高検証F1スコアモデル
2. **最新エポック**: 最終エポックモデル（`last.ckpt`）

## TensorBoard可視化

### TensorBoard起動

```bash
# 例：シングルモーダルNFNet（ローカル）
tensorboard --logdir="./logs/single/nfnet"

# 例：マルチモーダルNFNet+Transformer（Colab）
# tensorboard --logdir="/content/drive/MyDrive/Time_Series_Classifier/logs/multi/nfnet_transformer"
```

### 利用可能メトリクス

- **スカラー**: 学習・検証損失、F1スコア、学習率推移
- **画像**: 入力データとモデル注目領域の可視化（設定による）
- **グラフ**: モデルネットワーク構造
- **分布**: モデル重みとバイアス
- **ヒストグラム**: 勾配とアクティベーション分布

## トラブルシューティング

- **GPUメモリエラー**: `batch_size`を削減または`accumulate_grad_batches`を増加。`precision: '16-mixed'`を使用
- **NFNet読み込みエラー**: TorchVisionを更新、または自動的にResNet18フォールバックが発生
- **学習収束問題**: 学習率（`lr_head`, `lr_backbone`）または`weight_decay`を調整
- **Windows環境**: 設定で`num_workers: 0`を設定（デフォルト設定）
- **チェックポイント未発見**: チェックポイントファイル名とパスを`checkpoints/{model_mode}/{model_architecture_name}/`で確認

## 参考文献

- [NFNets論文](https://arxiv.org/abs/2102.06171)
- [PyTorch Lightning ドキュメント](https://pytorch-lightning.readthedocs.io/)
- [TensorBoard ドキュメント](https://www.tensorflow.org/tensorboard/)

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 貢献

貢献を歓迎します！プルリクエストをお気軽に提出してください。

## 引用

研究でこのプロジェクトを使用する場合は、以下の引用をご検討ください：

```bibtex
@software{timeseries_image_classifier,
  title={Time-Series Image Classifier},
  author={kechirojp},
  year={2025},
  url={https://github.com/kechirojp/timeseries-image-classifier}
}
```


