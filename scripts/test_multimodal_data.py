#!/usr/bin/env python3
"""
マルチモーダルデータローディングの動作確認スクリプト

このスクリプトは、マルチモーダル設定でのデータセットの読み込みと
前処理が正しく動作するかを確認します。

使用方法:
    python scripts/test_multimodal_data.py
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path

# プロジェクトルートを取得してパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datamodule import TimeSeriesDataModule
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multimodal_data_loading():
    """マルチモーダルデータローディングをテスト"""
    
    # 設定ファイルの読み込み
    config_path = project_root / 'configs' / 'config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f'設定ファイルの読み込みに失敗: {e}')
        return False
    
    # マルチモーダルモードに設定
    config['model_mode'] = 'multi'
    
    logger.info('=== マルチモーダルデータローディングテスト ===')
    logger.info(f'設定ファイル: {config_path}')
    logger.info(f'モデルモード: {config["model_mode"]}')
    
    # 特徴量ファイルの存在確認
    ts_data_path = config['timeseries']['data_path']
    if not os.path.isabs(ts_data_path):
        ts_data_path = project_root / ts_data_path
    
    if not ts_data_path.exists():
        logger.error(f'特徴量ファイルが見つかりません: {ts_data_path}')
        return False
    
    logger.info(f'特徴量ファイル: {ts_data_path}')
    
    # 特徴量データの内容確認
    try:
        df = pd.read_csv(ts_data_path)
        logger.info(f'特徴量データ形状: {df.shape}')
        logger.info(f'カラム: {list(df.columns)}')
        logger.info(f'先頭5行:\n{df.head()}')
        
        # タイムスタンプの変換テスト
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f'タイムスタンプ範囲: {df["timestamp"].min()} ～ {df["timestamp"].max()}')
        
    except Exception as e:
        logger.error(f'特徴量データの読み込みに失敗: {e}')
        return False
    
    # データモジュールの初期化テスト
    try:
        logger.info('\n=== データモジュール初期化テスト ===')
        data_module = TimeSeriesDataModule(config)
        logger.info('✓ データモジュールの初期化が成功しました')
        
        # prepare_dataの実行テスト
        logger.info('\n=== prepare_data実行テスト ===')
        data_module.prepare_data()
        logger.info('✓ prepare_dataが成功しました')
        logger.info(f'時系列データ形状: {data_module.timeseries_df.shape}')
        
        # setupの実行テスト
        logger.info('\n=== setup実行テスト ===')
        data_module.setup(stage='fit')
        logger.info('✓ setupが成功しました')
        logger.info(f'全データセットサイズ: {len(data_module.full_dataset)}')
        
        # データサンプルの取得テスト
        logger.info('\n=== データサンプル取得テスト ===')
        sample = data_module.full_dataset[0]
        
        if len(sample) == 3:  # マルチモーダル: (img, ts_data, label)
            img, ts_data, label = sample
            logger.info(f'✓ マルチモーダルデータの取得が成功しました')
            logger.info(f'  画像形状: {img.size if hasattr(img, "size") else "PIL画像"}')
            logger.info(f'  時系列データ形状: {ts_data.shape}')
            logger.info(f'  ラベル: {label}')
        else:
            logger.warning(f'予期しないデータ形式: {len(sample)}要素')
            return False
            
    except Exception as e:
        logger.error(f'データモジュールのテストに失敗: {e}')
        logger.error(f'エラー詳細:', exc_info=True)
        return False
    
    logger.info('\n=== テスト完了 ===')
    logger.info('✓ マルチモーダルデータローディングのテストが正常に完了しました')
    return True

def test_singlemodal_comparison():
    """シングルモーダルとの比較テスト"""
    
    logger.info('\n=== シングルモーダル比較テスト ===')
    
    config_path = project_root / 'configs' / 'config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f'設定ファイルの読み込みに失敗: {e}')
        return False
    
    # シングルモーダルモードに設定
    config['model_mode'] = 'single'
    
    try:
        data_module = TimeSeriesDataModule(config)
        data_module.setup(stage='fit')
        
        sample = data_module.full_dataset[0]
        
        if len(sample) == 2:  # シングルモーダル: (img, label)
            img, label = sample
            logger.info(f'✓ シングルモーダルデータの取得が成功しました')
            logger.info(f'  画像形状: {img.size if hasattr(img, "size") else "PIL画像"}')
            logger.info(f'  ラベル: {label}')
        else:
            logger.warning(f'予期しないデータ形式: {len(sample)}要素')
            return False
            
    except Exception as e:
        logger.error(f'シングルモーダルテストに失敗: {e}')
        return False
    
    logger.info('✓ シングルモーダル比較テストが正常に完了しました')
    return True

if __name__ == '__main__':
    logger.info('マルチモーダルデータローディングテストを開始します...')
    
    success = True
    
    # マルチモーダルテスト
    if not test_multimodal_data_loading():
        success = False
    
    # シングルモーダル比較テスト
    if not test_singlemodal_comparison():
        success = False
    
    if success:
        logger.info('\n🎉 すべてのテストが正常に完了しました！')
        sys.exit(0)
    else:
        logger.error('\n❌ テストが失敗しました。上記のエラーメッセージを確認してください。')
        sys.exit(1)
