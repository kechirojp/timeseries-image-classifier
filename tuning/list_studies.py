#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optunaスタディ情報を表示するシンプルなスクリプト
"""

import os
import sys
import optuna
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_studies_in_file(db_path):
    """特定のSQLiteファイル内のスタディ名をリストアップ"""
    if not os.path.exists(db_path):
        logger.error(f"ファイルが存在しません: {db_path}")
        return []
        
    try:
        storage_url = f"sqlite:///{db_path}"
        study_names = optuna.study.get_all_study_names(storage=storage_url)
        return study_names
    except Exception as e:
        logger.error(f"スタディ名の取得中にエラー: {str(e)}")
        return []

def list_all_studies():
    """tuning/studies内のすべてのDBファイルと含まれるスタディ名を表示"""
    # プロジェクトルートの取得
    if os.path.exists('/content/drive/MyDrive/Time_Series_Classifier'):
        # Google Colab環境の場合
        project_root = '/content/drive/MyDrive/Time_Series_Classifier'
    else:
        # ローカル環境の場合
        project_root = '.'
    
    studies_dir = os.path.join(project_root, 'tuning', 'studies')
    
    if not os.path.exists(studies_dir):
        logger.error(f"ディレクトリが存在しません: {studies_dir}")
        return
    
    db_files = [f for f in os.listdir(studies_dir) if f.endswith('.db')]
    
    if not db_files:
        logger.warning(f"{studies_dir} にDBファイルが見つかりません")
        return
    
    logger.info(f"==== {studies_dir} 内のDBファイルとスタディ名 ====")
    
    for db_file in db_files:
        db_path = os.path.join(studies_dir, db_file)
        study_names = list_studies_in_file(db_path)
        
        logger.info(f"DBファイル: {db_file}")
        if study_names:
            for name in study_names:
                logger.info(f"  - スタディ名: {name}")
                
                # スタディの詳細情報も表示
                try:
                    study = optuna.load_study(study_name=name, storage=f"sqlite:///{db_path}")
                    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    logger.info(f"    トライアル数: {len(study.trials)} (完了: {len(completed)})")
                    
                    if study.best_trial:
                        logger.info(f"    ベストトライアル: #{study.best_trial.number}, 値: {study.best_trial.value:.6f}")
                except Exception as e:
                    logger.error(f"    スタディ '{name}' の詳細取得中にエラー: {e}")
        else:
            logger.info("  スタディが見つかりませんでした")
        
        logger.info("----")

if __name__ == "__main__":
    list_all_studies()
