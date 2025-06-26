# 特徴量分析・分類モデルパイプライン

このディレクトリには、特徴量分析、LightGBMモデルの学習、および分類のためのしきい値最適化を行うスクリプト群が含まれています。

## 各スクリプトの目的

### 1. `feature_analysis.py`

-   **目的**: 特徴量データからLightGBMモデルを学習し、特徴量重要度を評価します。
-   **主な機能**:
    -   データの前処理（対数変換、標準化オプションなど）。
    -   Optunaを用いたハイパーパラメータの最適化。
        -   時系列性を考慮した交差検証 (`TimeSeriesSplit`) を使用します。
        -   目的関数としてマクロ平均F1スコアを最大化します。
    -   最適化されたパラメータで最終モデルを学習。
    -   テストデータでのモデル評価（混同行列、分類レポート、特徴量重要度など）。
    -   学習済みモデル (`lgbm_model.txt`)、検証/テストデータ (`validation_data_*.csv`, `test_data_*.csv`)、Optunaの最適化履歴グラフなどを `outputs` ディレクトリに保存。

### 2. `threshold_optimizer.py`

-   **目的**: `feature_analysis.py` で学習されたモデルと検証データを使用し、クラス分類（Class_A/Class_B/Class_C）のための最適な確率しきい値を探索します。
-   **主な機能**:
    -   学習済みモデル (`lgbm_model.txt`) と検証データをロード。
    -   Optunaを使用し、分類のための複数の確率しきい値を最適化します。
    -   最適化の目的関数として、特定のクラスの評価メトリック（F1スコアなど）を最大化します。
    -   最適化されたしきい値 (`optimal_class_thresholds_*.json`) と、評価結果のサマリー図 (`threshold_optimization_summary_*.png`) を `outputs` ディレクトリに保存。

### 3. `classification_example.py`

-   **目的**: 学習済みのモデルとしきい値を使って、新しいデータポイントに対して分類を実行する例を示します。
-   **主な機能**:
    -   `models` と `outputs` ディレクトリから、学習済みモデルと最新のしきい値ファイルをロードします。
    -   ダミーデータを生成し、それに対する分類結果（Class 0, 1, or 2）をコンソールに出力します。
    -   実際のデータを使った分類処理のテンプレートとして利用できます。

## 使い方

### 前提条件

- Python 3.7以上
- 必要なパッケージ: `pandas`, `numpy`, `matplotlib`, `lightgbm`, `optuna`, `scikit-learn`, `seaborn`, `pyyaml`

### 実行方法

**推奨される実行順序:** `feature_analysis.py` -> `threshold_optimizer.py`

```bash
# feature_analysis ディレクトリに移動
cd /path/to/your_project/feature_analysis

# 特徴量分析とモデル学習を実行
python feature_analysis.py

# 分類しきい値の最適化を実行
python threshold_optimizer.py

# 学習済みモデルを使った分類の例を実行
python classification_example.py
```

### 設定のカスタマイズ

各スクリプト内の定数やパラメータを調整することで、挙動をカスタマイズできます:

- **`feature_analysis.py`**:
    - `N_TRIALS_OPTUNA`: ハイパーパラメータ最適化の試行回数。
    - `train_ratio`, `valid_ratio`: データ分割比率。
- **`threshold_optimizer.py`**:
    - `OPTIMIZATION_METRIC`: 最適化に使用するメトリック ('f1' など)。
    - `n_trials`: しきい値最適化の試行回数。