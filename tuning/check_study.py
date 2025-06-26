import optuna
import yaml
import os
import sys  # sysモジュールを先頭で明示的にインポート
import pandas as pd
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定ファイルのパスを指定 ---
# 環境変数から取得するか、Colab/ローカルのデフォルトパスを使用
TUNING_CONFIG_PATH_ENV = os.environ.get('TUNING_CONFIG_PATH')
BASE_CONFIG_PATH_ENV = os.environ.get('BASE_CONFIG_PATH') # 基本設定も念のため

# Colab/ローカルのパス判定
IN_COLAB = 'google.colab' in sys.modules  # sysがすでにインポートされているため修正
COLAB_PROJECT_ROOT = '/content/drive/MyDrive/Time_Series_Classifier'
LOCAL_PROJECT_ROOT = '.' # ローカルのプロジェクトルート
PROJECT_ROOT = COLAB_PROJECT_ROOT if IN_COLAB else LOCAL_PROJECT_ROOT

# チューニング設定ファイルの決定
if TUNING_CONFIG_PATH_ENV and os.path.exists(TUNING_CONFIG_PATH_ENV):
    tuning_config_path = TUNING_CONFIG_PATH_ENV
    logger.info(f"環境変数からチューニング設定ファイルを使用: {tuning_config_path}")
elif IN_COLAB:
    tuning_config_path = os.path.join(COLAB_PROJECT_ROOT, 'tuning', 'config_for_google_colab.yaml')
    logger.info(f"Colab用デフォルトチューニング設定ファイルを使用: {tuning_config_path}")
else:
    tuning_config_path = os.path.join(LOCAL_PROJECT_ROOT, 'tuning', 'config.yaml')
    logger.info(f"ローカル用デフォルトチューニング設定ファイルを使用: {tuning_config_path}")

# --- 設定ファイルの読み込み ---
try:
    with open(tuning_config_path, 'r', encoding='utf-8') as f:
        tuning_config = yaml.safe_load(f)
    logger.info(f"チューニング設定ファイルを読み込みました: {tuning_config_path}")
except FileNotFoundError:
    logger.error(f"エラー: チューニング設定ファイルが見つかりません: {tuning_config_path}")
    exit(1)
except yaml.YAMLError as e:
    logger.error(f"YAMLパースエラー: {e}")
    exit(1)

# --- Optunaデータベース情報 ---
storage_path = tuning_config['storage']['path']
# Colab環境の場合、パスを絶対パスに修正 (設定ファイルに絶対パスが書かれている前提だが念のため)
if IN_COLAB and not storage_path.startswith('/content/drive'):
     # storage_path が /content/drive で始まらない場合、Colabルートからの相対パスとみなし結合
     storage_path = os.path.join(COLAB_PROJECT_ROOT, storage_path.lstrip('/'))
     logger.info(f"Colab用にストレージパスを調整: {storage_path}")

storage_url = f"sqlite:///{storage_path}"
study_name = tuning_config['study']['name']
metric_name = tuning_config['study']['metric']

logger.info(f"\n--- Optunaデータベース情報 ---")
logger.info(f"データベースファイル: {storage_path}")
logger.info(f"スタディ名: {study_name}")
logger.info(f"ストレージURL: {storage_url}")
logger.info(f"監視メトリック: {metric_name}")

# --- Optunaスタディの読み込み ---
try:
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    logger.info(f"\nスタディ '{study_name}' をロードしました。")
    logger.info(f"総トライアル数: {len(study.trials)}")

    # 完了したトライアルのみをフィルタリング
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]

    logger.info(f"  完了したトライアル数: {len(completed_trials)}")
    logger.info(f"  早期終了したトライアル数: {len(pruned_trials)}")
    logger.info(f"  失敗したトライアル数: {len(failed_trials)}")
    logger.info(f"  実行中のトライアル数: {len(running_trials)}")


    # --- 最良のトライアル情報を表示 ---
    if completed_trials: # 完了したトライアルがある場合のみ
        try:
            best_trial = study.best_trial
            logger.info("\n--- 最良のトライアル (完了済み) --- ")
            logger.info(f"  トライアル番号: {best_trial.number}")
            logger.info(f"  評価値 ({metric_name}): {best_trial.value:.6f}")
            logger.info("  パラメータ:")
            for key, value in best_trial.params.items():
                logger.info(f"    {key}: {value}")
            logger.info(f"  開始日時: {best_trial.datetime_start}")
            logger.info(f"  完了日時: {best_trial.datetime_complete}")
            logger.info(f"  所要時間: {best_trial.duration}")
        except ValueError:
            logger.warning("完了したトライアルがありますが、最良のトライアルを取得できませんでした。")
        except Exception as e:
            logger.error(f"最良トライアル情報の表示中にエラー: {e}")
    else:
        logger.warning("完了したトライアルがないため、最良のトライアル情報は表示できません。")

    # --- 全トライアル情報をDataFrameで表示 ---
    if study.trials:
        # DataFrameに変換して表示
        df = study.trials_dataframe()
        # 表示オプションを設定
        pd.set_option('display.max_rows', None) # 全ての行を表示
        pd.set_option('display.max_columns', None) # 全ての列を表示
        pd.set_option('display.width', 1000) # 表示幅を広げる
        pd.set_option('display.max_colwidth', None) # 列の内容を省略しない

        logger.info("\n--- 全トライアル詳細 (DataFrame) ---")
        print(df) # loggerではなくprintで見やすく表示

        # 特に重要なカラムを抜粋して表示
        logger.info("\n--- 主要なトライアル情報抜粋 ---")
        important_columns = ['number', 'value', 'state', 'duration'] + [f'params_{p}' for p in study.best_params.keys()]
        # 存在するカラムのみを選択
        display_columns = [col for col in important_columns if col in df.columns]
        print(df[display_columns].sort_values(by='value', ascending=(study.direction != 'maximize')))

    else:
        logger.info("スタディにトライアルがまだありません。")

except FileNotFoundError:
     logger.error(f"エラー: データベースファイルが見つかりません: {storage_path}")
     logger.error("Optunaの学習が一度も実行されていないか、パスが間違っている可能性があります。")
except ModuleNotFoundError as e:
    logger.error(f"必要なライブラリが見つかりません: {e}")
    logger.error("pip install optuna pandas PyYAML を実行してください。")
except Exception as e:
    logger.error(f"スタディのロードまたは結果の表示中に予期せぬエラーが発生しました: {e}")
    logger.error(f"データベースファイル '{storage_path}' が破損しているか、アクセス権限に問題がある可能性があります。")
    logger.error(f"解決しない場合、データベースファイルを手動で削除 (rm {storage_path}) してから再実行を試みてください。")
