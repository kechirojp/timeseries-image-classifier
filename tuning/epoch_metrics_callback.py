#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
エポックごとの検証メトリクスをOptunaのデータベースに保存するためのコールバック
これにより、トライアルが中断された場合でも最良のメトリックスを保持できる
"""

import os
import logging
import json
from datetime import datetime
from lightning.pytorch.callbacks import Callback

# ロギング設定
logger = logging.getLogger(__name__)

class EpochMetricsSaverCallback(Callback):
    """エポックごとのメトリクスを保存するコールバック
    
    トライアルが完了しなくても最良のメトリクスが捕捉されるようにする
    """
    def __init__(self, trial, checkpoint_dir, model_name, study_config):
        """
        Args:
            trial (optuna.Trial): Optunaのトライアルオブジェクト
            checkpoint_dir (str): チェックポイント保存ディレクトリ
            model_name (str): モデル名（ファイル名に使用）
            study_config (dict): 最適化の設定情報
        """
        super().__init__()
        self.trial = trial
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.study_config = study_config
        self.best_metric_value = None
        self.best_epoch = -1
        self.metrics_history = {}
        self.metric_name = study_config.get('metric', 'val_f1')
        self.direction = study_config.get('direction', 'maximize')
        self.is_better = self._is_better_func()
        
        # メトリクスの現在地ファイルパス
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.metrics_file = os.path.join(checkpoint_dir, f'trial_{trial.number}_metrics.json')
    
    def _is_better_func(self):
        """最適化の方向に応じて、値が改善されたかどうかを判断する関数を返す"""
        if self.direction == 'maximize':
            return lambda new, best: best is None or new > best
        else:
            return lambda new, best: best is None or new < best
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """バリデーションエポック終了時にメトリクスを保存"""
        # 現在のエポックのメトリクスを取得
        current_epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        # 主要なメトリクスを辞書に格納
        epoch_metrics = {}
        for key in ['train_loss', 'train_f1', 'val_loss', 'val_f1']:
            if key in metrics:
                value = metrics[key]
                if hasattr(value, 'item'):
                    value = value.item()
                epoch_metrics[key] = value
        
        # メトリクス履歴に追加
        self.metrics_history[current_epoch] = epoch_metrics
        
        # 目標メトリクスが存在するか確認
        if self.metric_name in epoch_metrics:
            current_metric = epoch_metrics[self.metric_name]
            
            # より良い値の場合は、トライアルのuser_attrを更新
            if self.is_better(current_metric, self.best_metric_value):
                self.best_metric_value = current_metric
                self.best_epoch = current_epoch
                
                # トライアルにメトリクスを設定（データベースに反映）
                for key, value in epoch_metrics.items():
                    try:
                        self.trial.set_user_attr(key, value)
                    except Exception as e:
                        logger.warning(f"エポック中のuser_attr設定でエラー ({key}={value}): {e}")
                
                # 最良のメトリクスをファイルにも保存（バックアップとして）
                try:
                    best_info = {
                        'trial_number': self.trial.number,
                        'best_epoch': self.best_epoch,
                        'best_metrics': epoch_metrics,
                        'best_metric_name': self.metric_name,
                        'best_metric_value': self.best_metric_value,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(self.metrics_file, 'w', encoding='utf-8') as f:
                        json.dump(best_info, f, indent=2)
                    
                    logger.info(f"トライアル {self.trial.number} エポック {current_epoch}: "
                              f"新しい最良 {self.metric_name}={self.best_metric_value:.6f} "
                              f"(metrics={epoch_metrics})")
                except Exception as e:
                    logger.error(f"メトリクスファイル保存エラー: {e}")
