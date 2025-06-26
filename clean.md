# プロジェクトクリーニング記録

## GitHubプッシュ必要ファイル選定

### 必須ファイル（コア機能）
以下のファイルがないとプロジェクトが動作しない：

#### プロジェクト構成ファイル
- `README.md` - プロジェクト説明（要修正）
- `requirements.txt` - 依存パッケージ
- `main.py` - メインエントリーポイント
- `Dockerfile` - Docker設定
- `docker-compose.yml` - Docker Compose設定

#### 設定ファイル
- `configs/config.yaml` - ローカル設定（要修正）
- `configs/config_for_google_colab.yaml` - Colab設定（要修正）  
- `configs/config_utils.py` - 設定ユーティリティ
- `configs/__init__.py` - パッケージ初期化

#### ソースコード（srcディレクトリ予想）
- `src/` 以下の全てのソースコード

#### 特徴量分析
- `feature_analysis/feature_analysis.py` - 特徴量重要度分析
- `feature_analysis/colab_runner.ipynb` - Colab実行用ノートブック
- `feature_analysis/__init__.py`

#### スクリプト
- `scripts/` 以下のユーティリティスクリプト

#### チューニング
- `tuning/` 以下のハイパーパラメータ最適化関連

### 除外ファイル（プッシュ不要）
- `data/` - データセット（容量大・プライベート）
- `checkpoints/` - 学習済みモデル（容量大）
- `lightning_logs/` - TensorBoardログ（容量大）
- `logs/` - 学習ログ（容量大）
- `__pycache__/` - Pythonキャッシュ
- `*.pyc` - Pythonコンパイル済みファイル
- `best_params_*.json` - 最適化結果（環境依存）
- 個人的なメモファイル（*.txt）

## 金融関連記述の汎用化作業

### 1. データフォルダ名の変更
- `nasdaq100_dir` → `dataset_a_dir`
- `GER30_dir` → `dataset_b_dir`  
- `US30_dir` → `dataset_c_dir`
- `symbols` → `categories` または `classes`

### 2. 設定ファイル修正箇所
- 各設定ファイル内の金融関連パラメータ名
- コメント内の株式・先物への言及

### 3. README.md修正箇所
- プロジェクト説明の汎用化
- 使用例の汎用化
- 金融特有の文言削除

### 4. ソースコード修正箇所
- 変数名・関数名の汎用化
- コメント内の金融用語削除
- クラス名の汎用化

## 作業進捗
- [x] 必要ファイル選定完了
- [x] 設定ファイル修正完了
  - `symbols` → `datasets`
  - `nasdaq100_dir`, `GER30_dir`, `US30_dir` → `dataset_a_dir`, `dataset_b_dir`, `dataset_c_dir`
  - 金融関連特徴量名を汎用化 (`PZO`, `TMF`, `Williams_4`, etc. → `feature_1`, `feature_2`, etc.)
  - クラス名を汎用化 (`Sell`, `Buy`, `Hold` → `Class_A`, `Class_B`, `Class_C`)
- [x] README.md汎用化完了
  - プロジェクト説明を「時系列画像分類器」に変更
  - 金融関連の表現を削除・汎用化
- [x] .gitignore作成完了
- [x] ソースコード汎用化完了
  - `StockDataModule` → `TimeSeriesDataModule`
  - `MultimodalStockClassifier` → `MultimodalClassifier`
  - コメント内の金融関連表現を汎用化
- [x] 不要ファイル削除対象確認
- [x] feature_analysisディレクトリクリーンアップ完了
  - 不要なPNG画像ファイル削除
  - 古いノートブック・ディレクトリ削除
  - `classification_example.py`のパス解決を動的化
  - README.mdを汎用化
- [x] tuningディレクトリクリーンアップ完了
  - 不要なリネームスクリプト削除 (`rename_checkpoints.py`, `rename_checkpoints_for_google_colab.py`)
  - キャッシュディレクトリ・過去の実行結果ディレクトリ削除
  - `optimize.py`内のクラス名汎用化 (`StockDataModule` → `TimeSeriesDataModule`, `MultimodalStockClassifier` → `MultimodalClassifier`)
  - Google Colab設定ファイル内のハードコードされたプロジェクトパス汎用化
  - `colab_runner.ipynb`の金融関連記述を汎用化
- [x] 環境依存パスの完全汎用化完了
  - `configs/config.yaml`: 絶対パス → 相対パス (`J:/マイドライブ/...` → `./data/...`)
  - `configs/config_for_google_colab.yaml`: Google Colab用パス汎用化 (`NFNet_Classifier_pretrained` → `Time_Series_Classifier`)
  - `tuning/config.yaml`, `tuning/config_for_google_colab.yaml`: チューニング用パス汎用化
  - `src/datamodule.py`: コメント内のパス例を汎用化
  - `main.py`, `tuning/*.py`: プロジェクトルートパスを相対パスまたは動的解決に変更
  - 全プロジェクトファイルで環境固有パスを削除し、GitHub上でのクローン・実行を可能に
- [x] GitHubアップロード最終準備完了
  - `.gitignore`にプロジェクト固有の除外ルール追加
  - `.dockerignore`の金融関連ディレクトリ名を汎用化
  - 不要な`__pycache__`ディレクトリの削除
  - `feature_analysis.py`内の具体的なファイル名を汎用化
  - プロジェクト全体の最終汎用化チェック完了

## 完了した汎用化内容

### 1. データフォルダ名・パラメータ名の変更
- `symbols` → `datasets`
- `nasdaq100_dir` → `dataset_a_dir`
- `GER30_dir` → `dataset_b_dir`
- `US30_dir` → `dataset_c_dir`

### 2. クラス名・関数名の汎用化
- `Sell` → `Class_A`
- `Buy` → `Class_B`
- `Hold` → `Class_C`
- `StockDataModule` → `TimeSeriesDataModule`
- `MultimodalStockClassifier` → `MultimodalClassifier`

### 3. 特徴量名の汎用化
- 金融指標 (`PZO`, `TMF`, `Williams_4`, `MFI_4`, `%SD`, `diff`) → 汎用名 (`feature_1` ～ `feature_6`)

### 4. プロジェクト説明の汎用化
- 「時系列画像分類器」→「汎用時系列データ処理システム」として説明
- 金融関連の具体的記述を削除・汎用化
- 汎用的な時系列データ処理システムとして説明

### 5. ディレクトリクリーンアップ
- `feature_analysis`ディレクトリから不要ファイル削除（古いノートブック、PNG画像など）
- `tuning`ディレクトリから一時的なスクリプトとキャッシュディレクトリ削除
- Google Colab用のノートブック（`colab_runner.ipynb`）を汎用化して技術力アピール用に保持

### 6. .gitignore作成
- データ、ログ、チェックポイントファイルを除外
- 標準的なPythonプロジェクト用設定

### 7. ソースコード汎用化
- クラス名、関数名の金融特化表現を削除
- コメント内の金融用語を一般的な表現に変更
- ハードコードされたプロジェクトパスを動的解決に変更

### 8. 環境依存パスの完全汎用化
- ローカル環境固有パス (`J:/マイドライブ/...`) を相対パス (`./data/...`) に変更
- Google Colab用パスの汎用化 (`NFNet_Classifier_pretrained` → `Time_Series_Classifier`)
- 設定ファイル内のすべてのハードコードされたパスを環境非依存に変更
- プロジェクトルートからの相対パス表記に統一

### 9. GitHubアップロード準備の最終調整
- `.gitignore`の更新：プロジェクト固有の一時ファイルと不要スクリプトを除外
- `.dockerignore`の汎用化：金融関連ディレクトリ名を汎用的なパターンに変更
- 残存する具体的なファイル名の汎用化
- プロジェクト全体のクリーンアップ完了

## GitHubプッシュ推奨ファイル

### 必須ファイル
- `README.md` ✅
- `requirements.txt` ✅
- `main.py` ✅
- `Dockerfile` ✅ 
- `docker-compose.yml` ✅
- `.gitignore` ✅

### 設定ファイル
- `configs/config.yaml` ✅
- `configs/config_for_google_colab.yaml` ✅
- `configs/config_utils.py` ✅
- `configs/__init__.py` ✅

### ソースコード
- `src/` 以下全て ✅
- `feature_analysis/` 以下全て ✅ 
- `scripts/` 以下全て ✅
- `tuning/` 以下全て ✅

### 除外ファイル（プッシュ不要）
- `data/` - データセット（プライベート・容量大）
- `checkpoints/` - 学習済みモデル（容量大）  
- `lightning_logs/` - TensorBoardログ（容量大）
- `logs/` - 学習ログ（容量大）
- `__pycache__/` - Pythonキャッシュ
- `*.json` - 最適化結果（環境依存）
- 個人的なメモファイル

## 総括

### 汎用化の目的達成状況
✅ **完全達成**: 金融特化プロジェクトから汎用的な時系列画像分類システムへの変換が完了

### 主要な変更点
1. **概念レベルの汎用化**
   - 株式・先物 → 時系列データ分類
   - 売買判断 → 汎用的な3クラス分類
   - 金融指標 → 汎用的な時系列特徴量

2. **技術実装の維持**
   - 機械学習アーキテクチャは完全保持
   - マルチモーダル対応維持
   - ハイパーパラメータ最適化機能維持

3. **再利用性の向上**
   - どの時系列データにも適用可能
   - 医療、製造業、IoTなど様々な分野で利用可能
   - 設定ファイルの変更のみで異なるデータセットに対応

### GitHubプッシュ準備完了
- 必要ファイルのみに絞り込み完了
- プライベートデータの除外完了  
- 汎用的な説明文書に更新完了
- オープンソース化準備完了
