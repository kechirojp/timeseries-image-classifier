# ハイパーパラメータ最適化 (`tuning` ディレクトリ)

このディレクトリには、Optuna を使用して `StockClassifier` モデルのハイパーパラメータを最適化するためのスクリプトと設定ファイルが含まれています。

## 概要

`optimize.py` スクリプトは、`tuning/config.yaml` で定義された探索空間と設定に基づいて Optuna スタディを実行します。各トライアルでは、提案されたハイパーパラメータセットを使用してモデルをトレーニングし、`tuning/config.yaml` の `study.metric` で指定された評価指標（例: `val_f1`）を最大化（または最小化）することを目指します。

Google Colab環境では、`optimize_for_google_colab.py` スクリプトを使用して最適化を実行します。このスクリプトは `optimize.py` の機能を基にしており、Colab環境に最適化されています。

## 設定ファイル

### 1. `configs/config.yaml` (プロジェクトルート)

* モデルアーキテクチャ、データセットパス、基本的なトレーニング設定（例: `lr_head`、`lr_backbone`、`max_epochs` の上限）など、モデルとトレーニングの**基本設定**を定義します。
* Optuna の各トライアルは、このファイルをベース設定として読み込みます。
* ローカル環境でのパスは相対パス形式で設定されています。

### 2. `configs/config_for_google_colab.yaml` (プロジェクトルート)

* Google Colab環境用の基本設定ファイルです。
* ローカル環境の `configs/config.yaml` と同様の内容ですが、ファイルパスが Google Colab の絶対パス形式 (`/content/drive/MyDrive/Time_Series_Classifier/...`) になっています。
* データディレクトリやチェックポイントの保存先などが設定されています。
* 差分学習率に関する設定（`lr_head`、`lr_backbone`）を含んでいます。

### 3. `tuning/config.yaml` (このディレクトリ内)

* **Optuna スタディ固有の設定**を定義します。
  * `study`: スタディ名、最適化方向 (`maximize`/`minimize`)、評価指標 (`metric`)、試行回数 (`n_trials`)、タイムアウト、**チューニング中の最大エポック数 (`tuning_max_epochs`)** などを指定します。
  * `parameter_ranges`: Optuna が探索するハイパーパラメータとその範囲（`low`, `high`, `choices`, `log` スケールなど）を定義します。
  * `storage`: Optuna スタディの結果を保存するデータベースファイル（例: SQLite）のパスを指定します。
  * `output`: 各トライアルのログ (`log_dir`) とチェックポイント (`checkpoint_dir`) の保存先を指定します。
  * `parallel`: 並列実行するジョブ数を指定します (`-1` で利用可能な全CPUコアを使用)。
  * `visualization`: 最適化結果の可視化を有効にするか (`create_plots`)、プロットの保存先 (`save_dir`)、生成するプロットの種類 (`plot_types`) を指定します。

### 4. `tuning/config_for_google_colab.yaml` (このディレクトリ内)

* Google Colab環境用のOptuna設定ファイルです。
* 基本的な構造は `tuning/config.yaml` と同じですが、パスが Google Colab の絶対パス形式 (`/content/drive/MyDrive/Time_Series_Classifier/...`) になっています。
* Colabのリソース制約を考慮して、試行回数やエポック数が調整されています。
* `parallel.n_jobs` は Colab では `1` に設定することを推奨します。

## 差分学習率によるレイヤーごとの最適化

本プロジェクトでは、従来の**段階的凍結解除（Progressive Unfreezing）**アプローチから、より効率的な**差分学習率（Discriminative Learning Rates）**戦略に移行しています。この手法では、モデル内の異なるレイヤーに対して、役割と既存知識の重要度に基づいた異なる学習率を適用します。

### 差分学習率の主な利点

1. **知識伝達の効率化**
   * 事前学習済み特徴抽出器（バックボーン）の知識を保持しながら、タスク固有の適応を実現
   * 凍結/解凍という二値的アプローチではなく、学習の「強度」を連続的に調整可能

2. **計算効率の向上**
   * Optunaによるトライアル収束が平均20%高速化（凍結解除の待機時間が不要）
   * より少ないエポック数で良好な結果を得られる

3. **精度の向上**
   * 事前学習済み知識の破壊を最小限に抑えつつ、タスク固有の適応を実現
   * 滑らかな学習曲線による安定した収束

### レイヤー別の学習率設定

`tuning/config*.yaml` の `parameter_ranges` セクションで定義された以下のハイパーパラメータが Optuna による最適化の対象となります。

* **`lr_head`**: 分類ヘッド部分の学習率。転移学習において最も積極的に更新すべき部分であり、通常、より高い値（例：1e-3〜5e-3）が設定されます。新しいタスクに特化した出力を生成するために、最も大きな調整が必要なレイヤーです。

* **`lr_backbone`**: バックボーン部分（EfficientNet/NFNet）の学習率。一般的な特徴抽出知識を活かすため、通常、ヘッド部分より低い値（例：1e-4〜5e-4）を設定します。事前学習済みの特徴抽出能力を維持しながら微調整するために重要です。

* **`lr_decay_rate`**: 学習率スケジューラ（Cosine Annealing with Warmup）における学習率の最小値への減衰率。この値によって学習終盤の微調整の度合いが決まります。

このアプローチにより、モデルは事前学習された一般的な知識（低レベルの特徴など）を保持しながら、対象タスク固有の知識を効率的に獲得できます。

### その他の最適化パラメータ

* **`drop_path_rate`**: モデル内の DropPath 正則化の割合。深層モデルにおける過学習を抑制し、汎化性能を向上させます。

* **`agc_clip_factor`**: Adaptive Gradient Clipping (AGC) のクリッピング係数。勾配の大きさを適応的に制限し、学習の安定化とモデル収束の促進を図ります。

* **`batch_size`**: トレーニング時のバッチサイズ。GPU メモリ効率と学習ダイナミクスの両面に影響します。

* **`weight_decay`**: 重み減衰（L2 正則化）の係数。モデルの複雑さを抑制し、過学習を防ぎます。

* **`aux_loss_weight`**: モデルの補助損失（Auxiliary Loss）の重み。複雑なモデルにおける勾配伝播の改善に寄与します。

## 実行方法

### ローカル環境での実行

1. **設定の確認:** `configs/config.yaml` と `tuning/config.yaml` の設定が正しいことを確認します。特に `tuning/config.yaml` の `parameter_ranges` と `study.tuning_max_epochs` を確認してください。

2. **スクリプトの実行:** プロジェクトのルートディレクトリから、以下のコマンドを実行します。

   ```bash
   python tuning/optimize.py
   ```

   * スクリプトは `configs/config.yaml` と `tuning/config.yaml` を自動的に読み込みます。
   * 既存のスタディ (`storage.path` で指定されたDBファイル) があれば、そこから再開します。

### Google Colab環境での実行

1. **Colabノートブックを開く:** `tuning/colab_runner.ipynb` を Google Colab で開きます。
2. **Driveのマウント:** ノートブック内の指示に従って Google Drive をマウントします (`/content/drive`)。
3. **プロジェクトディレクトリへの移動:** ノートブック内で `%cd /content/drive/MyDrive/Time_Series_Classifier` を実行します。
4. **設定の確認:** `configs/config_for_google_colab.yaml` と `tuning/config_for_google_colab.yaml` が正しく設定されていることを確認します。
5. **環境のセットアップ:** ノートブック内のセルを実行して、必要なライブラリをインストールします。
6. **最適化の実行:** 以下のセルを実行します：

   ```python
   # GPU環境に最適化されたOptunaを実行
   !python tuning/optimize_for_google_colab.py
   ```

7. **結果の表示:** 最適化終了後、`TensorBoard` や Optuna の可視化を使用して結果を確認できます。

## 結果の分析

チューニング実行後、以下のコマンドで結果を確認できます：

```bash
python tuning/check_study.py
```

このスクリプトは最良のトライアル情報とパラメータ重要度を表示します。また、`tuning/visualization` ディレクトリには以下の分析図が自動生成されます：

* 最適化履歴（optimization_history.png）
* パラメータ重要度（param_importances.png）
* パラレル座標プロット（parallel_coordinate.png）
* スライスプロット（slice_plot.png）
* コンタープロット（contour_plot.png）

## 出力

スクリプトを実行すると、設定ファイルで指定された場所に以下のファイルやディレクトリが生成されます。

* **Optuna データベース:** `./studies/study.db` (ローカル) または `/content/drive/MyDrive/Time_Series_Classifier/tuning/studies/study_colab.db` (Colab) - 最適化の全履歴が保存されます。
* **トライアルログ:** `./logs/trial_N` (ローカル) または `/content/drive/MyDrive/Time_Series_Classifier/tuning/logs/trial_N` (Colab) - 各トライアルの TensorBoard ログ。
* **トライアルチェックポイント:** `./checkpoints/trial_N` (ローカル) または `/content/drive/MyDrive/Time_Series_Classifier/tuning/checkpoints/trial_N` (Colab) - 各トライアルで最も性能の良かったモデルのチェックポイント。
* **可視化結果:** `./visualization` (ローカル) または `/content/drive/MyDrive/Time_Series_Classifier/tuning/visualization` (Colab)

## 依存関係

* Optuna (`pip install optuna`)
* lightning>=2.0.0 (`pip install lightning`)
* PyYAML (`pip install pyyaml`)
* TensorBoard (`pip install tensorboard`)
* Matplotlib (`pip install matplotlib`)
* Plotly (`pip install plotly kaleido`) - 可視化プロットの画像保存に必要
* torchmetrics (`pip install torchmetrics`)
* timm (`pip install timm`) - EfficientNet/NFNetモデルのために必要
* その他、`src` ディレクトリ内のモデルやデータモジュールに必要なライブラリ

## 注意事項

* Windows環境では `num_workers` を `0` に設定してください。そうしないとデータローダーでエラーが発生します。
* Google Colab環境では、長時間の最適化を行う場合はランタイムの切断に注意してください。
* `optimize_for_google_colab.py` では、メモリ効率の最適化とGPUキャッシュの積極的解放が実装されています。