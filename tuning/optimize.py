import os
import sys
import yaml
import optuna

# Optunaの統合モジュールをインポート
try:
    # 新しい場所からインポートを試みる
    from optuna_integration import PyTorchLightningPruningCallback
except ImportError:
    # 従来の場所からのインポートをフォールバックとして試みる
    try:
        from optuna.integration import PyTorchLightningPruningCallback
    except ImportError:
        logger.error("PyTorchLightningPruningCallbackがどちらの場所からもインポートできませんでした。")
        logger.error("pip install optuna-integration を実行してください。")
        sys.exit(1)
        
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback # Callback をインポート
from lightning.pytorch.loggers import TensorBoardLogger
# import matplotlib.pyplot as plt # 未使用のためコメントアウト
import json
from datetime import datetime
import gc # メモリキャッシュクリア用
import logging
import math # isnanチェック用

# ロギング設定 - シンプルかつ情報量の多いログを出力
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- プロジェクトルートをモジュールパスに追加 ---
# optimize.pyの絶対パスからプロジェクトルートを取得
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
logger.info(f"プロジェクトルートをパスに追加: {project_root_dir}")

# --- 環境変数から設定ファイルのパスを取得 ---
TUNING_CONFIG_PATH = os.environ.get('TUNING_CONFIG_PATH')
BASE_CONFIG_PATH = os.environ.get('BASE_CONFIG_PATH')  # 追加: 基本設定ファイルパスの環境変数

# --- 環境判定 ---
IN_COLAB = 'google.colab' in sys.modules
COLAB_PROJECT_ROOT = '/content/drive/MyDrive/Time_Series_Classifier'
LOCAL_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = COLAB_PROJECT_ROOT if IN_COLAB else LOCAL_PROJECT_ROOT

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(PROJECT_ROOT)

# プロジェクト固有のインポート - シングルモーダル/マルチモーダルの両方をインポート
from src.models.single_modal import SingleModalClassifier
from src.models.multi_modal import MultimodalClassifier
from src.datamodule import TimeSeriesDataModule

# エポックごとのメトリクス保存用コールバックをインポート
try:
    from tuning.epoch_metrics_callback import EpochMetricsSaverCallback
except ImportError:
    # 相対パスを使ってインポート
    import sys
    import os  # osモジュールをインポート
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tuning.epoch_metrics_callback import EpochMetricsSaverCallback
from src.callbacks import StageWiseUnfreezeCallback # StageWiseUnfreezeCallbackをインポート

def format_metric_value(value, precision=3):
    """メトリック値を指定された小数点以下の桁数でフォーマットする。NaNの場合は'NaN'を返す。"""
    if value is None or math.isnan(value):
        return "NaN"
    return f"{value:.{precision}f}"

class EpochStartBestTrialLoggerCallback(Callback):
    """各エポック開始時に、Optunaの暫定ベストトライアル情報をログに出力するコールバック"""
    def __init__(self, study):
        super().__init__()
        self.study = study
        self.metric_name = study.user_attrs.get("metric_name", "value") # studyからメトリック名を取得

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        try:
            # 完了したトライアルが存在する場合のみベストトライアルを取得試行
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                logger.info(f"Epoch {current_epoch} Start: Interim Best Trial - Not available yet (no completed trials).")
                return

            best_trial_so_far = self.study.best_trial
            logger.info(f"--- Epoch {current_epoch} Start: Interim Best Trial (Optuna Study: {self.study.study_name}) ---")
            logger.info(f"  Best Trial Number: {best_trial_so_far.number}")
            logger.info(f"  Best Value ({self.metric_name}): {best_trial_so_far.value:.6f}")
            logger.info(f"  Best Params: {best_trial_so_far.params}")
        except ValueError: # まだ有効なトライアルがない場合 (例えば最初のトライアルの最初のエポックなど)
            logger.info(f"Epoch {current_epoch} Start: Interim Best Trial - Not available yet (ValueError accessing best_trial).")
        except Exception as e:
            logger.warning(f"Epoch {current_epoch} Start: Error logging interim best trial - {e}")

class ObjectiveCreator:
    """Optunaの最適化目的関数を作成するクラス"""
    
    def __init__(self, base_config, tuning_config, study): # study を追加
        """
        Args:
            base_config (dict): モデルの基本設定
            tuning_config (dict): チューニング関連の設定
            study (optuna.study.Study): OptunaのStudyオブジェクト
        """
        self.base_config = base_config.copy()
        self.param_ranges = tuning_config['parameter_ranges']
        self.study_config = tuning_config['study']
        self.output_config = tuning_config['output']
        self.study = study # studyオブジェクトを保持

        # 暫定ベストトライアル情報保存用ディレクトリの準備
        # output_config['log_dir'] が絶対パスであることを確認し、なければPROJECT_ROOTを基準にする
        # デフォルトのログディレクトリを 'tuning/logs' に設定
        log_dir_base = self.output_config.get('log_dir', 'tuning/logs')
        if not os.path.isabs(log_dir_base):
            log_dir_base = os.path.join(PROJECT_ROOT, log_dir_base)
        self.interim_best_trial_dir = os.path.join(log_dir_base, 'interim_best_trials')
        os.makedirs(self.interim_best_trial_dir, exist_ok=True)
        logger.info(f"暫定ベストトライアル情報保存ディレクトリ: {self.interim_best_trial_dir}")
        
        # --- パス調整処理 (ローカル環境とColab環境のパス差異を吸収) ---
        for key in ['data_dir', 'nasdaq_dir']:
            if key in self.base_config:
                current_path = self.base_config[key]
                # 相対パスの場合、実行環境に応じたプロジェクトルートを基準とした絶対パスに変換
                if isinstance(current_path, str) and not os.path.isabs(current_path):
                    # 相対パスを実行環境のプロジェクトルートから解決
                    self.base_config[key] = os.path.join(PROJECT_ROOT, current_path)
                    logger.info(f"{key} を絶対パスに変換: {self.base_config[key]}")
                elif isinstance(current_path, str):
                    logger.info(f"{key} は絶対パスとして認識: {current_path}")
                # 相対パスの場合、PROJECT_ROOTを基準とした絶対パスに変換
                elif isinstance(current_path, str) and not os.path.isabs(current_path):
                    self.base_config[key] = os.path.join(PROJECT_ROOT, current_path.lstrip('./').lstrip('.\\'))
                    logger.info(f"{key} を絶対パスに変換: {self.base_config[key]}")
                # その他の絶対パスはそのまま
                elif isinstance(current_path, str) and os.path.isabs(current_path):
                     logger.info(f"{key} は絶対パスとして認識: {current_path}")

        # 出力ディレクトリ (checkpoint_dir, log_dir) も絶対パスに変換
        for dir_key in ['checkpoint_dir', 'log_dir']:
            if dir_key in self.output_config:
                path = self.output_config[dir_key]
                if not os.path.isabs(path):
                    self.output_config[dir_key] = os.path.join(PROJECT_ROOT, path.lstrip('./').lstrip('.\\'))
                    logger.info(f"出力ディレクトリ {dir_key} を絶対パスに変換: {self.output_config[dir_key]}")
                os.makedirs(self.output_config[dir_key], exist_ok=True)
            else:
                 # デフォルトのパスを設定 (例: logs/tuning_logs, checkpoints/tuning_checkpoints)
                default_path = os.path.join(PROJECT_ROOT, 'tuning', dir_key.replace('_dir', ''))
                self.output_config[dir_key] = default_path
                os.makedirs(default_path, exist_ok=True)
                logger.info(f"出力ディレクトリ {dir_key} をデフォルトパスに設定: {default_path}")

    def _save_interim_best_trial_info(self, trial_number_being_started):
        """トライアル開始時に、その時点での暫定ベストトライアル情報をログ出力し、JSONで保存する"""
        if trial_number_being_started == 0: # 最初のトライアルでは実行しない
            logger.info("最初のトライアルのため、暫定ベストトライアルのログ出力と保存はスキップします。")
            return

        try:
            # 完了したトライアルが存在し、かつ最良のトライアルが取得できる場合のみ処理
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if not completed_trials:
                logger.info(f"--- Interim Best Trial (before Trial {trial_number_being_started}): Not available yet (no completed trials). ---")
                return

            best_trial_so_far = self.study.best_trial
            metric_name = self.study.user_attrs.get("metric_name", "value")
            
            logger.info(f"--- Interim Best Trial (before Trial {trial_number_being_started}) ---")
            logger.info(f"  Number: {best_trial_so_far.number}")
            logger.info(f"  Value ({metric_name}): {best_trial_so_far.value:.6f}")
            logger.info(f"  Params: {best_trial_so_far.params}")

            # ファイル名に使用するメトリクス値を取得
            train_loss = best_trial_so_far.user_attrs.get('train_loss', float('nan'))
            train_f1 = best_trial_so_far.user_attrs.get('train_f1', float('nan'))
            val_loss = best_trial_so_far.user_attrs.get('val_loss', float('nan'))
            val_f1 = best_trial_so_far.user_attrs.get(self.study_config['metric'], float('nan')) # 最適化対象のメトリック

            # モデル名を取得 (base_configから)
            model_architecture_name = self.base_config.get("model_architecture_name", "unknown_model")

            # ファイル名を生成
            filename_parts = [
                model_architecture_name,
                f"trial{best_trial_so_far.number}",
                f"train_loss_{format_metric_value(train_loss)}",
                f"train_f1_{format_metric_value(train_f1)}",
                f"val_loss_{format_metric_value(val_loss)}",
                f"val_f1_{format_metric_value(val_f1)}"
            ]
            filename = "_".join(filter(None, filename_parts)) + ".json"
            filepath = os.path.join(self.interim_best_trial_dir, filename)

            # JSONデータを準備
            interim_best_info = {
                "trial_number_being_started": trial_number_being_started,
                "best_trial_number_so_far": best_trial.so_far.number,
                f"best_value_so_far ({metric_name})": best_trial_so_far.value,
                "best_params_so_far": best_trial_so_far.params,
                "best_trial_user_attrs": best_trial_so_far.user_attrs, 
                "datetime_saved": datetime.now().isoformat()
            }

            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(interim_best_info, f, indent=2, ensure_ascii=False)
            logger.info(f"暫定ベストトライアル情報をJSONに保存しました: {filepath}")

        except ValueError:
            logger.info(f"--- Interim Best Trial (before Trial {trial_number_being_started}): Not available yet (ValueError accessing best_trial). ---")
        except Exception as e:
            logger.warning(f"--- Interim Best Trial (before Trial {trial_number_being_started}): Error logging/saving - {e} ---")

    def __call__(self, trial):
        """Optunaのtrialごとに呼び出される目的関数
        
        Args:
            trial: Optunaのtrialオブジェクト
            
        Returns:
            float: 評価指標の値（通常はval_f1スコア）
        """
        # --- トライアル開始時の処理 ---
        self._save_interim_best_trial_info(trial.number)

        # 設定のディープコピー（元の設定を変更しないため）
        config = self.base_config.copy()
        if 'model' in self.base_config:
            config['model'] = self.base_config['model'].copy()
        
        # --- 追加: Colab環境でのパス調整 (trialごとにも念のため確認) ---
        if IN_COLAB:
            if 'data_dir' in config and not os.path.isabs(config['data_dir']):
                 config['data_dir'] = os.path.join(COLAB_PROJECT_ROOT, config['data_dir'].lstrip('./'))
                 logger.debug(f"Trial {trial.number}: Colab data_dir 再確認/設定: '{config['data_dir']}'")
        # --- ここまで ---
        
        logger.info(f"--- Starting Trial {trial.number} ---")
        
        # 事後最適化モードかどうかをチェック
        is_post_training = self.study_config.get('is_post_training', False)
        if is_post_training:
            logger.info("事後最適化モードで実行します")
            # 事後最適化用のチェックポイントパスを設定
            checkpoint_path = self.study_config.get('checkpoint_path')
            if not checkpoint_path:
                logger.error("事後最適化モードですが、チェックポイントパスが設定されていません")
                raise ValueError("事後最適化モードでチェックポイントパスが未設定")

            if not os.path.exists(checkpoint_path): # 実際のファイル存在確認
                logger.error(f"指定されたチェックポイントが存在しません: {checkpoint_path}")
                raise FileNotFoundError(f"チェックポイントが見つかりません: {checkpoint_path}")

            # 事後最適化用にエポック数を設定 (これはモデルのオプティマイザ/スケジューラ設定に影響する可能性がある)
            fine_tuning_epochs = self.study_config.get('fine_tuning_epochs', 10)
            config['max_epochs'] = fine_tuning_epochs
            # config にチェックポイントパスを保存しておく。Trainer に渡す際にこれを使用する。
            config['checkpoint_path'] = checkpoint_path # 既存のキーをそのまま使用

            logger.info(f"事後最適化設定: checkpoint_path={checkpoint_path}, fine_tuning_epochs={fine_tuning_epochs}")
        # else: # 事前最適化の場合、config['max_epochs'] は base_config の値が使われる

        # ----- ハイパーパラメータの提案 -----
        # 学習率関連のパラメータ - 簡潔な表記でパラメータを設定
        config['lr_head'] = trial.suggest_float('lr_head', 
            float(self.param_ranges['lr_head']['low']),
            float(self.param_ranges['lr_head']['high']),
            log=self.param_ranges['lr_head'].get('log', False))
        
        config['lr_backbone'] = trial.suggest_float('lr_backbone',
            float(self.param_ranges['lr_backbone']['low']),
            float(self.param_ranges['lr_backbone']['high']),
            log=self.param_ranges['lr_backbone'].get('log', False))
        
        # 設定ファイルの互換性のため、キーの存在を確認
        if 'lr_decay_rate' in self.param_ranges:
            config['lr_decay_rate'] = trial.suggest_float('lr_decay_rate',
                float(self.param_ranges['lr_decay_rate']['low']),
                float(self.param_ranges['lr_decay_rate']['high']),
                log=self.param_ranges['lr_decay_rate'].get('log', False))

        # NFNet特有のパラメータ
        if 'model' not in config:
            config['model'] = {}
        config['model']['drop_path_rate'] = trial.suggest_float('drop_path_rate',
            float(self.param_ranges['drop_path_rate']['low']),
            float(self.param_ranges['drop_path_rate']['high']))
        
        config['agc_clip_factor'] = trial.suggest_float('agc_clip_factor',
            float(self.param_ranges['agc_clip_factor']['low']),
            float(self.param_ranges['agc_clip_factor']['high']),
            log=self.param_ranges['agc_clip_factor'].get('log', False))

        # トレーニング関連のパラメータ
        config['batch_size'] = trial.suggest_categorical('batch_size',
            self.param_ranges['batch_size']['choices'])
        
        config['weight_decay'] = trial.suggest_float('weight_decay',
            float(self.param_ranges['weight_decay']['low']),
            float(self.param_ranges['weight_decay']['high']),
            log=self.param_ranges['weight_decay'].get('log', False))
        
        # aux_loss_weightパラメータが定義されている場合のみ使用
        if 'aux_loss_weight' in self.param_ranges:
            config['aux_loss_weight'] = trial.suggest_float('aux_loss_weight',
                float(self.param_ranges['aux_loss_weight']['low']),
                float(self.param_ranges['aux_loss_weight']['high']))
            logger.info(f"Trial {trial.number}: 補助損失を設定 - 重み: {config['aux_loss_weight']}")
        
        # auto_unfreeze設定の処理
        # 転移学習戦略フラグを取得（デフォルトはFalse）
        use_progressive_unfreezing = config.get('use_progressive_unfreezing', False)
        config['use_progressive_unfreezing'] = use_progressive_unfreezing
        
        if 'auto_unfreeze' in self.base_config:
            config['auto_unfreeze'] = self.base_config['auto_unfreeze'].copy()
            
            # オプション：use_progressive_unfreezing=TrueかつOptunaに範囲設定があれば自動解凍タイミングも最適化
            if use_progressive_unfreezing and 'auto_unfreeze' in self.param_ranges:
                logger.info(f"Trial {trial.number}: 段階的凍結解除が有効 - タイミングを最適化します")
                # ステージ1のエポック設定
                config['auto_unfreeze']['stage1_epoch'] = trial.suggest_int('auto_unfreeze_stage1',
                    self.param_ranges['auto_unfreeze'].get('stage1_min', 5),
                    self.param_ranges['auto_unfreeze'].get('stage1_max', 15))
                
                # ステージ2はステージ1から一定間隔後
                stage1 = config['auto_unfreeze']['stage1_epoch']
                stage2_offset = trial.suggest_int('auto_unfreeze_stage2_offset', 
                    self.param_ranges['auto_unfreeze'].get('stage2_offset_min', 5),
                    self.param_ranges['auto_unfreeze'].get('stage2_offset_max', 15))
                config['auto_unfreeze']['stage2_epoch'] = stage1 + stage2_offset
                
                # ステージ3はステージ2から一定間隔後
                stage2 = config['auto_unfreeze']['stage2_epoch']
                stage3_offset = trial.suggest_int('auto_unfreeze_stage3_offset',
                    self.param_ranges['auto_unfreeze'].get('stage3_offset_min', 10),
                    self.param_ranges['auto_unfreeze'].get('stage3_offset_max', 20))
                config['auto_unfreeze']['stage3_epoch'] = stage2 + stage3_offset
                
                logger.info(f"Trial {trial.number}: auto_unfreeze stages: {config['auto_unfreeze']}")
            elif use_progressive_unfreezing:
                logger.info(f"Trial {trial.number}: 段階的凍結解除が有効 - 固定タイミングを使用します")
            else:
                logger.info(f"Trial {trial.number}: 段階的凍結解除は無効 - 差分学習率を使用します")
        
        # ----- データモジュールとモデルの初期化 -----
        try:
            logger.info(f"Trial {trial.number}: TimeSeriesDataModule初期化 (data_dir='{config.get('data_dir', 'N/A')}')")
            data_module = TimeSeriesDataModule(config)
            
            # モデルモードに応じたモデル選択 (シングルモーダル/マルチモーダル)
            model_mode = config.get("model_mode", "single")
            if model_mode == "multi":
                logger.info(f"Trial {trial.number}: マルチモーダルモデルを使用します")
                model = MultimodalClassifier(config)
            else:
                logger.info(f"Trial {trial.number}: シングルモーダルモデルを使用します")
                model = SingleModalClassifier(config)
                
        except FileNotFoundError as e:
             logger.error(f"Trial {trial.number}: データディレクトリが見つかりません: {e}")
             logger.error(f"使用されたdata_dir: {config.get('data_dir', '設定なし')}")
             raise # データディレクトリ不足はトライアル失敗として扱う
        except Exception as e:
            logger.error(f"Trial {trial.number}: モデル/データモジュール初期化エラー: {e}")
            raise  # このエラーは重大なので再送出
        
        # ----- コールバックの設定 -----
        # モデルモードとアーキテクチャ名に基づいたチェックポイントディレクトリを作成
        model_mode = config.get("model_mode", "single")
        model_architecture_name = config.get("model_architecture_name", "default_model")
        
        # チェックポイント保存先をmodel_mode/model_architecture_name/trial_Xの構造で設定
        trial_checkpoint_dir = os.path.join(
            self.output_config['checkpoint_dir'], 
            model_mode,
            model_architecture_name,
            f'trial_{trial.number}'
        )
        os.makedirs(trial_checkpoint_dir, exist_ok=True)
        
        # パラメータ情報を含む文字列を作成
        param_str = f"lrh{config['lr_head']:.1e}_lrb{config['lr_backbone']:.1e}"
        
        # より詳細な情報を含むチェックポイント名を設定
        # val_f1だけでなくval_lossも含める
        # PyTorch Lightningの形式で変数を参照 ({変数名:フォーマット}形式)
        checkpoint_callback = ModelCheckpoint(
            dirpath=trial_checkpoint_dir,
            filename=f'{model_architecture_name}_trial{trial.number}_{self.study_config["metric"]}'+'-{' + self.study_config["metric"] + ':.4f}' + '_val_loss-{val_loss:.4f}_' + param_str,
            monitor=self.study_config['metric'],
            mode='max' if self.study_config['direction'] == 'maximize' else 'min',
            save_top_k=1
        )
        
        # Optunaの早期打ち切りコールバック
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=self.study_config['metric'])
        
        # 各エポック開始時に暫定ベストトライアル情報をログ出力するコールバック
        epoch_start_logger_callback = EpochStartBestTrialLoggerCallback(self.study)
        
        # エポックごとのメトリクス保存コールバック (Colabの24時間制限対策)
        epoch_metrics_saver = EpochMetricsSaverCallback(
            trial=trial,
            checkpoint_dir=trial_checkpoint_dir,
            model_name=model_architecture_name,
            study_config=self.study_config
        )
        
        # 段階的凍結解除のコールバック設定
        callbacks = [checkpoint_callback, pruning_callback, epoch_start_logger_callback, epoch_metrics_saver]
        
        # auto_unfreezeが有効な場合のみ、凍結解除コールバックを追加
        if use_progressive_unfreezing:
            from src.callbacks import StageWiseUnfreezeCallback
            
            if 'auto_unfreeze' in config:
                # configから凍結解除のタイミングを取得（最適化済みまたは固定値）
                stage1_epoch = config['auto_unfreeze'].get('stage1_epoch', 10)
                stage2_epoch = config['auto_unfreeze'].get('stage2_epoch', 20)
                stage3_epoch = config['auto_unfreeze'].get('stage3_epoch', 30)
                
                unfreeze_callback = StageWiseUnfreezeCallback(
                    stage1_epoch=stage1_epoch,
                    stage2_epoch=stage2_epoch,
                    stage3_epoch=stage3_epoch
                )
                logger.info(f"Trial {trial.number}: 段階的凍結解除を設定 - stage1: {stage1_epoch}, stage2: {stage2_epoch}, stage3: {stage3_epoch}")
                callbacks.append(unfreeze_callback)
            else:
                logger.warning(f"Trial {trial.number}: auto_unfreeze設定が見つからないため、段階的凍結解除は無効になります")
        else:
            # 凍結解除を行わない (差分学習率を使用)
            logger.info(f"Trial {trial.number}: 段階的凍結解除は無効 - 差分学習率を使用します")
        
        # ----- トレーナーの設定 -----
        # is_post_training フラグに基づいてエポック数を決定
        if is_post_training:
            # 事後最適化の場合、study_config の tuning_max_epochs を使用 (事後最適化用に設定された値)
            tuning_epochs = self.study_config.get('tuning_max_epochs', 10) # デフォルトは事後最適化向けの短い値
            logger.info(f"事後最適化のため、エポック数を {tuning_epochs} に設定 (study_config.tuning_max_epochs).")
        else:
            # 事前最適化の場合、study_config の tuning_max_epochs を使用 (事前最適化用に設定された値)
            tuning_epochs = self.study_config.get('tuning_max_epochs', 40) # デフォルトは事前最適化向けの一般的な値
            logger.info(f"事前最適化のため、エポック数を {tuning_epochs} に設定 (study_config.tuning_max_epochs).")

        # Tensor Core 警告対応: Trainer初期化前に追加
        # configからprecision設定を読み込む (trialごとに異なる可能性があるため)
        # Optunaでprecision自体を探索する場合は、trial.suggest_categoricalなどで取得する
        precision_setting = config.get('precision', '32-true')
        # 混合精度計算 ('16-mixed' または 'bf16-mixed') を使用する場合に設定
        if precision_setting in ['16-mixed', 'bf16-mixed']:
            try:
                # Tensor Core を効率的に使用するための設定
                torch.set_float32_matmul_precision('medium')
                logger.info(f"混合精度 ({precision_setting}) と併用するため、torch.set_float32_matmul_precision('medium') を設定しました。")
            except Exception as e:
                # エラーが発生しても処理は続行するが、警告を表示
                logger.warning(f"torch.set_float32_matmul_precision の設定中にエラー: {e}")
        else:
            # 混合精度を使用しない場合は設定不要
            logger.info(f"精度設定: {precision_setting} (matmul精度設定はスキップ)")

        # TensorBoard Loggerの設定
        # model_mode と model_architecture_name を config から取得 (logger で使用する場合があるため事前に)
        # これらの値はチェックポイントパスの作成など、他の場所でも使用されている
        model_mode = config.get("model_mode", "single")
        model_architecture_name = config.get("model_architecture_name", "default_model")

        # ログディレクトリのベースパスを取得
        log_dir_base = self.output_config['log_dir']
        # TensorBoardLogger は save_dir と name/version を組み合わせてログパスを決定する
        # ここでは、trial ごとにサブディレクトリが作られるように name を設定
        pl_logger = pl.loggers.TensorBoardLogger(
            save_dir=log_dir_base, # ベースのログディレクトリ
            name=f"trial_{trial.number}", # trialごとのサブディレクトリ名
            version="" # version を空にすると、name の直下にログが保存される
        )
        logger.info(f"TensorBoardLoggerを設定: save_dir=\'{log_dir_base}\', name=\'trial_{trial.number}\'")

        trainer_init_args = {
            'max_epochs': tuning_epochs,
            'callbacks': callbacks,
            'accelerator': 'gpu' if torch.cuda.is_available() and not config.get('force_cpu', False) else 'cpu',
            'devices': 'auto',
            'logger': pl_logger, # 事前に定義したpl_loggerを使用
            'enable_progress_bar': True,
            'log_every_n_steps': 10,
            'precision': precision_setting,
            'reload_dataloaders_every_n_epochs': 0 # 以前のバージョンで設定されていたため維持
        }

        # 事後最適化の場合、ckpt_path は trainer.fit() に渡すため、ここでは設定しない
        # if is_post_training:
        #     trainer_init_args['ckpt_path'] = config['checkpoint_path']
        #     logger.info(f"Trainerにチェックポイント {config['checkpoint_path']} を設定しました (事後最適化)。")

        trainer = pl.Trainer(**trainer_init_args)

        # ----- モデルの学習 -----
        ckpt_path_for_fit = None
        if is_post_training and 'checkpoint_path' in config and config['checkpoint_path']:
            ckpt_path_for_fit = config['checkpoint_path']
            logger.info(f"Trainer.fit にチェックポイント {ckpt_path_for_fit} を設定しました (事後最適化)。")
        elif is_post_training:
            logger.warning("事後最適化モードですが、有効なチェックポイントパスがconfigに見つかりません。最初から学習します。")

        try:
            logger.info(f"Trial {trial.number}: {tuning_epochs}エポックのトレーニングを開始")
            trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path_for_fit) # ckpt_path を fit に渡す
            logger.info(f"Trial {trial.number}: トレーニング完了")
        except optuna.exceptions.TrialPruned:
            logger.info(f"Trial {trial.number}: 早期打ち切り")
            # プルーニングされた場合でも、その時点でのメトリクスを保存する試み
            if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics:
                trial.set_user_attr('train_loss', trainer.callback_metrics.get('train_loss', float('nan')).item() if hasattr(trainer.callback_metrics.get('train_loss'), 'item') else trainer.callback_metrics.get('train_loss', float('nan')))
                trial.set_user_attr('train_f1', trainer.callback_metrics.get('train_f1', float('nan')).item() if hasattr(trainer.callback_metrics.get('train_f1'), 'item') else trainer.callback_metrics.get('train_f1', float('nan')))
                trial.set_user_attr('val_loss', trainer.callback_metrics.get('val_loss', float('nan')).item() if hasattr(trainer.callback_metrics.get('val_loss'), 'item') else trainer.callback_metrics.get('val_loss', float('nan')))
                trial.set_user_attr(self.study_config['metric'], trainer.callback_metrics.get(self.study_config['metric'], float('nan')).item() if hasattr(trainer.callback_metrics.get(self.study_config['metric']), 'item') else trainer.callback_metrics.get(self.study_config['metric'], float('nan')))
            raise  # Pruned例外はOptunaに再送出
        except Exception as e:
            logger.error(f"Trial {trial.number}: トレーニングエラー: {e}")
            # エラーが発生した場合でも、その時点でのメトリクスを保存する試み
            if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics:
                trial.set_user_attr('train_loss', trainer.callback_metrics.get('train_loss', float('nan')).item() if hasattr(trainer.callback_metrics.get('train_loss'), 'item') else trainer.callback_metrics.get('train_loss', float('nan')))
                trial.set_user_attr('train_f1', trainer.callback_metrics.get('train_f1', float('nan')).item() if hasattr(trainer.callback_metrics.get('train_f1'), 'item') else trainer.callback_metrics.get('train_f1', float('nan')))
                trial.set_user_attr('val_loss', trainer.callback_metrics.get('val_loss', float('nan')).item() if hasattr(trainer.callback_metrics.get('val_loss'), 'item') else trainer.callback_metrics.get('val_loss', float('nan')))
                trial.set_user_attr(self.study_config['metric'], trainer.callback_metrics.get(self.study_config['metric'], float('nan')).item() if hasattr(trainer.callback_metrics.get(self.study_config['metric']), 'item') else trainer.callback_metrics.get(self.study_config['metric'], float('nan')))
            raise  # トレーニングの失敗はOptunaに伝える
        
        # ----- 最終的な評価指標を取得 -----
        metric_value = trainer.callback_metrics.get(self.study_config['metric'])

        # --- トライアル終了時の処理: メトリクスをuser_attrsに保存 ---
        train_loss_val = trainer.callback_metrics.get('train_loss', float('nan'))
        train_f1_val = trainer.callback_metrics.get('train_f1', float('nan'))
        val_loss_val = trainer.callback_metrics.get('val_loss', float('nan'))
        
        trial.set_user_attr('train_loss', train_loss_val.item() if hasattr(train_loss_val, 'item') else train_loss_val)
        trial.set_user_attr('train_f1', train_f1_val.item() if hasattr(train_f1_val, 'item') else train_f1_val)
        trial.set_user_attr('val_loss', val_loss_val.item() if hasattr(val_loss_val, 'item') else val_loss_val)
        if metric_value is not None:
             trial.set_user_attr(self.study_config['metric'], metric_value.item() if hasattr(metric_value, 'item') else metric_value)
        else:
            trial.set_user_attr(self.study_config['metric'], float('nan'))
        
        # メモリ解放処理を追加
        del model # モデルオブジェクトの削除
        del data_module # データモジュールオブジェクトの削除
        del trainer # トレーナーオブジェクトの削除
        gc.collect() # Pythonのガベージコレクションを実行
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # PyTorchのCUDAキャッシュをクリア
            logger.info(f"Trial {trial.number}: Cleared CUDA cache.")
        
        if metric_value is None:
            logger.warning(f"Trial {trial.number}: メトリック'{self.study_config['metric']}'が見つかりません")
            return 0.0 if self.study_config['direction'] == 'maximize' else float('inf')
        
        metric_value = metric_value.item()  # TensorからPythonの値に変換
        logger.info(f"Trial {trial.number}: 完了 - {self.study_config['metric']}: {metric_value:.6f}")
        
        return metric_value


def _try_plot(plot_func, study, vis_dir, plot_name, **kwargs):
    """Helper function to generate and save a plot with error handling."""
    try:
        fig = plot_func(study, **kwargs)
        fig.write_image(os.path.join(vis_dir, f"{plot_name}.png"))
        logger.info(f"{plot_name} プロットを保存しました。")
    except (ValueError, ImportError) as e:
        logger.warning(f"{plot_name} プロットの生成または保存に失敗しました: {e}")
    except Exception as e:
        logger.warning(f"{plot_name} プロット中に予期せぬエラーが発生しました: {e}")


def visualize_results(study, output_dir):
    """最適化結果を可視化して保存する関数
    
    Args:
        study: 完了したOptunaスタディ
        output_dir: 可視化結果を保存するディレクトリ
        
    Returns:
        str: 可視化結果が保存されたディレクトリのパス
    
    Note:
        - この関数は `plotly` および `kaleido` ライブラリが必要です。
        - 試行回数が少ない場合、一部のプロットが生成できないことがあります。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = os.path.join(output_dir, f"visualization_{study.study_name}_{timestamp}")
    os.makedirs(vis_dir, exist_ok=True)
    logger.info(f"可視化結果を保存: {vis_dir}")
    
    # 完了したトライアルがあるか確認
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        logger.warning("完了したトライアルがないため、可視化をスキップします。")
        return vis_dir

    # 各種プロットの生成と保存
    from optuna.visualization import (plot_optimization_history, plot_param_importances,
                                      plot_parallel_coordinate, plot_slice, plot_contour)

    # 1. 最適化履歴
    _try_plot(plot_optimization_history, study, vis_dir, "optimization_history")

    # 2. パラメータ重要度
    _try_plot(plot_param_importances, study, vis_dir, "param_importances")

    # 3. パラレル座標プロット
    _try_plot(plot_parallel_coordinate, study, vis_dir, "parallel_coordinate")

    # 4. スライスプロット
    _try_plot(plot_slice, study, vis_dir, "slice_plot")

    # 5. コンター図
    try:
        importances = optuna.importance.get_param_importances(study)
        top_params = list(importances.keys())[:2]
        if len(top_params) >= 2:
            _try_plot(plot_contour, study, vis_dir, "contour_plot", params=top_params)
        elif len(importances) > 0:
            logger.info("コンタープロットを生成するには、少なくとも2つのパラメータが必要です。スキップします。")
    except (ValueError, ImportError) as e:
        logger.warning(f"コンタープロットの準備または生成に失敗しました: {e}")
    except Exception as e:
        logger.warning(f"コンタープロットの準備または生成中に予期せぬエラーが発生しました: {e}")

    # 最良のトライアル情報をJSONファイルとして保存
    try:
        best_trial = study.best_trial
        best_trial_info = {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "datetime_start": best_trial.datetime_start.isoformat() if best_trial.datetime_start else None,
            "datetime_complete": best_trial.datetime_complete.isoformat() if best_trial.datetime_complete else None,
            "duration": str(best_trial.duration) if best_trial.duration else None,
            "state": str(best_trial.state)
        }
        
        with open(os.path.join(vis_dir, "best_trial.json"), "w", encoding='utf-8') as f:
            json.dump(best_trial_info, f, indent=2, ensure_ascii=False)
        logger.info("最良トライアル情報をJSONに保存しました")
    except ValueError:
         logger.warning("最良のトライアルが見つかりません。JSONファイルの保存をスキップします。")
    except Exception as e:
        logger.warning(f"最良トライアル情報の保存に失敗: {e}")
    
    return vis_dir


def main():
    """メイン実行関数 - Optunaスタディの設定、実行、結果の可視化を行う"""
    
    # ----- 設定ファイルの読み込み -----
    # 1. ベース設定ファイル(環境変数または標準パス)
    if BASE_CONFIG_PATH and os.path.exists(BASE_CONFIG_PATH):
        base_config_path = BASE_CONFIG_PATH
        logger.info(f"環境変数から基本設定ファイルを使用: {base_config_path}")
    elif IN_COLAB:
        base_config_path = os.path.join(COLAB_PROJECT_ROOT, 'configs', 'config_for_google_colab.yaml')
        logger.info(f"Colab用デフォルト基本設定ファイルを使用: {base_config_path}")
    else:
        base_config_path = os.path.join(LOCAL_PROJECT_ROOT, 'configs', 'config.yaml')
        logger.info(f"ローカル用デフォルト基本設定ファイルを使用: {base_config_path}")
    
    # 2. チューニング設定ファイル(環境変数または標準パス)
    if TUNING_CONFIG_PATH and os.path.exists(TUNING_CONFIG_PATH):
        tuning_config_path = TUNING_CONFIG_PATH
        logger.info(f"環境変数からチューニング設定ファイルを使用: {tuning_config_path}")
    elif IN_COLAB:
        tuning_config_path = os.path.join(COLAB_PROJECT_ROOT, 'tuning', 'config_for_google_colab.yaml')
        logger.info(f"Colab用デフォルトチューニング設定ファイルを使用: {tuning_config_path}")
    else:
        tuning_config_path = os.path.join(LOCAL_PROJECT_ROOT, 'tuning', 'config.yaml')
        logger.info(f"ローカル用デフォルトチューニング設定ファイルを使用: {tuning_config_path}")
    
    # 設定ファイルの読み込み
    try:
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        with open(tuning_config_path, 'r', encoding='utf-8') as f:
            tuning_config = yaml.safe_load(f)
        
        logger.info(f"設定ファイルを読み込みました: {base_config_path}, {tuning_config_path}")
    except FileNotFoundError as e:
        logger.error(f"設定ファイルが見つかりません: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"YAMLパースエラー: {e}")
        sys.exit(1)
    
    # ----- パス処理 -----
    # Colabでのパス修正
    if IN_COLAB:
        # 基本設定のパスを修正
        for key in ['data_dir', 'nasdaq_dir', 'logs_dir', 'checkpoint_dir']:
            if key in base_config:
                # J:/ や j:/ で始まるローカルパスはColab環境では使用できない
                if isinstance(base_config[key], str) and ('j:/' in base_config[key].lower()):
                    # パスの末尾部分を取得してColab環境のパスに変換
                    path_suffix = base_config[key].split('/')[-1]
                    base_config[key] = os.path.join(COLAB_PROJECT_ROOT, path_suffix)
                    logger.info(f"Colab用にパスを変換: {key}='{base_config[key]}'")
                # 相対パスを絶対パスに変換
                elif isinstance(base_config[key], str) and not os.path.isabs(base_config[key]):
                    base_config[key] = os.path.join(COLAB_PROJECT_ROOT, base_config[key].lstrip('./'))
                    logger.info(f"Colab用に相対パスを絶対パスに変換: {key}='{base_config[key]}'")
    
    # ----- 出力ディレクトリの準備 -----
    # チューニング関連のパスを準備 (絶対パスに変換済み)
    for dir_path_key in ['log_dir', 'checkpoint_dir']:
        dir_path = tuning_config['output'][dir_path_key]
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"ディレクトリを確認/作成: {dir_path}")

    storage_dir = os.path.dirname(tuning_config['storage']['path'])
    os.makedirs(storage_dir, exist_ok=True)
    logger.info(f"ディレクトリを確認/作成: {storage_dir}")

    # 可視化結果用ディレクトリ
    vis_dir = os.path.join(PROJECT_ROOT, 'tuning', 'visualization') # 修正: PROJECT_ROOTを使用
    os.makedirs(vis_dir, exist_ok=True)
    
    # ----- Optunaストレージの設定 (SQLiteタイムアウト追加) -----
    # タイムアウトを30000ms (30秒)に設定し、ロックエラーを減らす
    storage_url = f"sqlite:///{tuning_config['storage']['path']}?timeout=30000" # 絶対パスを使用
    try:
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=60,
            grace_period=120
        )
        logger.info(f"Optunaストレージを初期化: {storage_url}")
    except Exception as e:
        logger.error(f"ストレージ初期化エラー: {e}")
        sys.exit(1)
    
    # ----- Optunaスタディの作成 -----
    # Prunerの設定 - シンプルなMedianPrunerを使用
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=tuning_config.get('pruner', {}).get('n_startup_trials', 5)
    )
    
    study_name = tuning_config['study']['name']
    study_direction = tuning_config['study']['direction']
    metric_to_optimize = tuning_config['study']['metric'] # 追加: 最適化対象のメトリック名
    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=study_direction,
            storage=storage,
            pruner=pruner,
            load_if_exists=True  # 既存のスタディがあれば再開
        )
        # studyオブジェクトにメトリック名を保存（コールバックで参照するため）
        study.set_user_attr("metric_name", metric_to_optimize)
        logger.info(f"Optunaスタディ '{study_name}' を準備しました (方向: {study_direction}, メトリック: {metric_to_optimize})")
    except Exception as e:
        logger.error(f"スタディ作成エラー: {e}")
        sys.exit(1)
    
    # ----- 最適化の実行 -----
    objective = ObjectiveCreator(base_config, tuning_config, study) # 修正: studyオブジェクトを渡す
    n_trials = tuning_config['study']['n_trials']
    timeout = tuning_config['study'].get('timeout')  # タイムアウトはオプション
    n_jobs = tuning_config['parallel'].get('n_jobs', 1)  # 並列ジョブ数
    
    logger.info(f"最適化を開始: トライアル数={n_trials}, タイムアウト={timeout}秒, 並列数={n_jobs}")
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs
            # callbacks引数は削除 (ObjectiveCreator内で処理するため)
        )
        logger.info("最適化が完了しました")
    except KeyboardInterrupt:
        logger.warning("ユーザーによる中断")
    except Exception as e:
        logger.error(f"最適化中にエラーが発生: {e}")
    
    # ----- 最良の結果を表示 -----
    try:
        logger.info(f"完了したトライアル数: {len(study.trials)}")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logger.warning("完了したトライアルがありません。最良のトライアル情報を表示できません。")
        else:
            best_trial = study.best_trial
            logger.info("最良のトライアル:")
            logger.info(f"  番号: {best_trial.number}")
            logger.info(f"  値 ({tuning_config['study']['metric']}): {best_trial.value:.6f}")
            logger.info("  パラメータ:")
            for key, value in best_trial.params.items():
                logger.info(f"    {key}: {value}")
    except ValueError:
        logger.warning("最良のトライアル情報を取得できません（完了したトライアルがない可能性）")
    except Exception as e:
        logger.error(f"最良トライアル情報の表示中に予期せぬエラーが発生: {e}")

    # ----- 結果の可視化 -----
    try:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
             logger.warning("完了したトライアルがないため、可視化と最良設定の保存をスキップします。")
        else:
            vis_result_dir = visualize_results(study, vis_dir) # vis_dirは絶対パス
            
            # 最良のパラメータをconfig形式で保存
            best_trial = study.best_trial
            best_config = base_config.copy() # 元のbase_configを使用 (パスは調整済み)
            if 'model' in base_config:
                best_config['model'] = base_config['model'].copy()
                
            for key, value in best_trial.params.items():
                if key == 'drop_path_rate':
                    if 'model' not in best_config:
                        best_config['model'] = {}
                    best_config['model']['drop_path_rate'] = value
                elif key.startswith('auto_unfreeze_'):
                    if 'auto_unfreeze' not in best_config:
                         best_config['auto_unfreeze'] = {}
                    if key == 'auto_unfreeze_stage1':
                        best_config['auto_unfreeze']['stage1_epoch'] = value
                else:
                    best_config[key] = value
            
            best_config_path = os.path.join(vis_result_dir, "best_config.yaml")
            with open(best_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"最良の設定を保存: {best_config_path}")

    except ValueError as ve:
         logger.warning(f"可視化または設定保存中にエラーが発生しました: {ve}")
    except Exception as e:
        logger.error(f"可視化または設定保存中に予期せぬエラーが発生: {e}")


if __name__ == "__main__":
    main()
