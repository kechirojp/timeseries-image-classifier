#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
特徴量分析・モデル学習・しきい値最適化スクリプト

LightGBMを使用してデータの特徴量重要度を算出し、
（オプションで）累積重要度に基づいて特徴量を選択し、
Optunaでハイパーパラメータとクラス重みを最適化し、
最終モデルを学習し、上位特徴量をYAMLファイルに保存します。
"""

import os # osモジュールがインポートされていることを確認
import pandas as pd
import lightgbm as lgbm
from typing import Dict, Tuple, Optional, Any, List
# TimeSeriesSplit をインポート
from sklearn.model_selection import train_test_split, TimeSeriesSplit
# f1_score をインポート
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler # StandardScalerをインポート
import numpy as np
import datetime
import warnings
import seaborn as sns
import optuna
import yaml
# --- ここに matplotlib.pyplot のインポートを追加 ---
import matplotlib.pyplot as plt
# ---------------------------------------------

# WARNING抑制
warnings.filterwarnings('ignore')

# Matplotlibを非ブロッキングモードに設定（図が表示されても処理が止まらないようにする）
# plt.ion() # インタラクティブモードをオン -> save_figure内で制御するため不要

# 画像保存の設定
SAVE_FIGURES = True  # 図を自動的にファイルに保存するフラグ
SHOW_FIGURES = False  # 図を画面に表示するかどうかのフラグ
IMAGES_DIR = 'images'  # 画像を保存するサブディレクトリ名

# --- 追加: ウィンドウサイズ定数 ---
# 特徴量生成時のウィンドウサイズに合わせて設定 (例: 40)
# config.yaml のデータセットディレクトリパス名などから判断
WINDOW_SIZE = 40
# ---------------------------------

# --- 追加: 特徴量選択設定 ---
USE_FEATURE_SELECTION = True # 累積重要度に基づく特徴量選択を行うか
CUMULATIVE_IMPORTANCE_THRESHOLD = 0.95 # 累積重要度の閾値 (USE_FEATURE_SELECTION=True の場合)
# -----------------------------

# 画像保存関数を定義
def save_figure(fig: plt.Figure, name: str, output_base_dir: str):
    """
    MatplotlibのFigureオブジェクトを指定ディレクトリ内のimagesサブディレクトリに保存します。

    Args:
        fig (plt.Figure): 保存する Matplotlib の Figure オブジェクト。
        name (str): ファイル名のベース部分 (例: 'confusion_matrix')。
        output_base_dir (str): 保存先のベースディレクトリ (例: 'feature_analysis/outputs')。
    """
    # 保存フラグがFalseの場合は何もしない
    if not SAVE_FIGURES:
        # 表示フラグもFalseなら、メモリ解放のために図を閉じる
        if not SHOW_FIGURES:
            plt.close(fig)
        return

    # imagesサブディレクトリのパスを構築 (直接 outputs/images を指定)
    image_dir = os.path.join(output_base_dir, IMAGES_DIR)
    try:
        # ディレクトリが存在しない場合は作成 (親ディレクトリも必要なら作成)
        os.makedirs(image_dir, exist_ok=True)
    except OSError as e:
        # ディレクトリ作成に関するOSレベルのエラー (権限不足など)
        print(f"画像保存ディレクトリの作成に失敗しました ({image_dir}): {e}")
        # エラー時も、表示フラグがFalseなら図を閉じる
        if not SHOW_FIGURES:
            plt.close(fig)
        return

    # タイムスタンプ付きのファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(image_dir, filename)

    try:
        # 図をファイルに保存 (bbox_inches='tight'で余白を最小化)
        fig.savefig(filepath, bbox_inches='tight')
        print(f"図を保存しました: {filepath}")
    except IOError as e:
        # ファイル書き込み権限がない、ディスク容量不足など
        print(f"図のファイル書き込み中にエラーが発生しました ({filepath}): {e}")
    except ValueError as e:
        # savefig に不正な引数が渡された場合など
        print(f"図の保存関数の引数に問題があります ({filepath}): {e}")
    finally:
        # 保存処理後、表示フラグがFalseなら図を閉じる
        # (エラー発生時も、表示しない設定なら閉じるべき)
        if not SHOW_FIGURES:
            plt.close(fig)

# プロジェクトルート取得関数
def get_project_root() -> str:
    """プロジェクトのルートディレクトリパスを動的に取得する"""
    # このスクリプトの絶対パスを取得
    script_path = os.path.abspath(__file__)
    # このスクリプトが配置されているディレクトリ (例: /path/to/project/feature_analysis)
    script_dir = os.path.dirname(script_path)
    # 親ディレクトリ（プロジェクトルート）
    project_root = os.path.dirname(script_dir)
    return project_root

# 前処理関数
def preprocess_data(actions_path: str, indicators_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, float]]: # 戻り値の型ヒントを修正
    """
    時系列データとアクションデータを読み込み、前処理を行う
    
    Args:
        actions_path: アクションデータのCSVファイルパス
        indicators_path: 時系列指標データのCSVファイルパス
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[int, float]]: 前処理後の特徴量DataFrame(X)、目的変数DataFrame(y)、クラス重み辞書
    """
    print(f"アクションデータを読み込み中: {actions_path}")
    df_actions = pd.read_csv(actions_path, parse_dates=True, index_col='timestamp')
    
    # ラベルのカラム名をactionsに統一
    if 'adjusted_signal' in df_actions.columns:
        df_actions = df_actions.rename(columns={'adjusted_signal': 'actions'})
    elif 'action' in df_actions.columns:
        df_actions = df_actions.rename(columns={'action': 'actions'})
    elif 'label' in df_actions.columns:
        df_actions = df_actions.rename(columns={'label': 'actions'})
    
    print(f"時系列指標データを読み込み中: {indicators_path}")
    df = pd.read_csv(indicators_path, parse_dates=True, index_col='timestamp')
    
    # dataset_idカラムを除去
    # 理由: dataset_idは単なるデータセット識別子であり、時系列データの予測に有用な情報を含まない
    # 機械学習の特徴量として使用すると、データリークや過学習の原因となる可能性がある
    # また、数値であってもカテゴリカル変数として扱うべきIDであり、回帰・分類の特徴量には不適切
    if "dataset_id" in df.columns:
        df = df.drop(["dataset_id"], axis=1, errors='ignore')
    
    # 全ての拡張特徴量を使用（185個の特徴量）
    print(f"拡張特徴量データを使用 - 利用可能なカラム数: {len(df.columns)}")
    print(f"利用する特徴量の例: {list(df.columns[:10])}...")  # 最初の10個を表示
    
    # 特徴量として使用するカラムの選択
    # 対数変換前にカラム選択
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 正負記号付き対数変換
    epsilon = 1e-10  # ゼロまたは負の値に対する小さな定数
    df_log = df[numerical_columns].apply(lambda x: np.sign(x) * np.log1p(np.abs(x) + epsilon))

    # 標準化 (Standardization)
    # scaler = StandardScaler()
    # df_scaled = pd.DataFrame(scaler.fit_transform(df_log), index=df_log.index, columns=df_log.columns)
    # df_processed = df_scaled # 標準化を適用する場合はこちらを使用
    df_processed = df_log # 標準化しない場合はこちらを使用
    
    # インデックスを合わせる
    common_indices = df_processed.index.intersection(df_actions.index) # df_log -> df_processed
    X = df_processed.loc[common_indices] # df_log -> df_processed
    y = df_actions.loc[common_indices]
    
    # クラスの分布を確認
    class_counts = y['actions'].value_counts()
    print(f"クラス分布: \n{class_counts}")
    
    # クラスの重みを計算（compute_class_weightを使用）
    class_weights = compute_class_weight('balanced', classes=np.unique(y['actions']), y=y['actions'])
    class_weight_dict = dict(zip(np.unique(y['actions']), class_weights))
    print(f"計算されたクラス重み: {class_weight_dict}")
    
    print(f"前処理後のデータサイズ - X: {X.shape}, y: {y.shape}")
    return X, y, class_weight_dict  # クラス重みを返すように修正

def compare_n_estimators(X: pd.DataFrame, y: pd.DataFrame, n_estimators_list: List[int], 
                         train_size: float = 0.6, valid_size: float = 0.2) -> pd.DataFrame:
    """
    時系列データに適した方法での異なるn_estimators値の性能比較
    
    Args:
        X: 特徴量DataFrame
        y: 目的変数DataFrame
        n_estimators_list: テストするn_estimatorsの値のリスト
        train_size: 訓練データの割合
        valid_size: 検証データの割合
        
    Returns:
        pd.DataFrame: 各n_estimatorsでの結果を含むDataFrame
    """
    total_rows = len(X)
    train_end = int(total_rows * train_size)
    
    base_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': 5,
        'min_data_in_leaf': 30,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'random_state': 42
    }
    
    # 時系列的な分割ポイントを定義
    splits = []
    window_size = int(total_rows * 0.2)  # スライディングウィンドウのサイズ
    for i in range(3):  # 3回の評価
        if train_end + (i * window_size) >= total_rows:
            break
        split_train_end = train_end + (i * window_size)
        split_valid_end = min(split_train_end + window_size, total_rows)
        splits.append((split_train_end, split_valid_end))
    
    for n_est in n_estimators_list:
        print(f"\nテスト中のn_estimators: {n_est}")
        params = base_params.copy()
        params['n_estimators'] = n_est
        
        fold_results = []
        
        # 時系列的な検証
        for i, (split_train_end, split_valid_end) in enumerate(splits):
            print(f"Fold {i+1}: インデックス {split_train_end} までで学習, "
                  f"インデックス {split_train_end} から {split_valid_end} で検証")
            
            X_train = X.iloc[:split_train_end]
            y_train = y.iloc[:split_train_end]
            X_valid = X.iloc[split_train_end:split_valid_end]
            y_valid = y.iloc[split_train_end:split_valid_end]
            
            model = lgbm.LGBMClassifier(**params)
            model.fit(
                X_train, y_train.values.ravel(),  # ravel()でシリーズを1次元配列に変換
                eval_set=[(X_valid, y_valid.values.ravel())],
                eval_metric='multi_logloss'
            )
            
            y_pred = model.predict(X_valid)
            accuracy = accuracy_score(y_valid, y_pred)
            
            fold_results.append({
                'n_estimators': n_est,
                'fold': i,
                'accuracy': accuracy,
                'train_size': len(X_train),
                'valid_size': len(X_valid)
            })
            
            print(f"Fold {i+1} 精度: {accuracy:.4f}")
        
        results = pd.concat([results, pd.DataFrame(fold_results)], ignore_index=True)
    
    # 結果の集計と表示
    summary = results.groupby('n_estimators').agg({
        'accuracy': ['mean', 'std']
    })
    
    print("\n結果サマリー:")
    print(summary)
    
    # 可視化
    plt.figure(figsize=(12, 6))
    
    # 精度のプロット
    plt.subplot(1, 2, 1)
    plt.errorbar(
        results.groupby('n_estimators')['accuracy'].mean().index,
        results.groupby('n_estimators')['accuracy'].mean(),
        yerr=results.groupby('n_estimators')['accuracy'].std(),
        fmt='o-'
    )
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.title('n_estimatorsに対するモデル性能')
    plt.grid(True)
    
    # 個別のfoldの結果をプロット
    plt.subplot(1, 2, 2)
    for fold in range(len(splits)):
        fold_data = results[results['fold'] == fold]
        plt.plot(fold_data['n_estimators'], fold_data['accuracy'], 
                'o-', label=f'Fold {fold+1}')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.title('Foldごとの性能')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results

# --- 累積重要度分析関数 ---
# コメントアウト解除
def analyze_cumulative_importance(X: pd.DataFrame, y: pd.DataFrame,
                               train_ratio: float = 0.6,
                               validation_ratio: float = 0.2,
                               threshold: float = 0.95,
                               n_splits: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    """
    累積重要度に基づく特徴量分析を行い、選択された特徴量のリストを返す。
    時系列性を考慮したウォークフォワード分析で重要度を計算する。

    Args:
        X (pd.DataFrame): 特徴量DataFrame。
        y (pd.DataFrame): 目的変数DataFrame。
        train_ratio (float, optional): 訓練データの割合。デフォルトは0.6。
        validation_ratio (float, optional): 検証データの割合。デフォルトは0.2。
        threshold (float, optional): 累積重要度の閾値。デフォルトは0.95。
        n_splits (int, optional): ウォークフォワード分析の分割数。デフォルトは3。

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - 累積重要度を含むDataFrame。
            - 選択された特徴量のリスト。
    """
    # 一時的なモデル学習用のパラメータ (Optuna最適化前なので簡易的な設定)
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device': 'cpu', # CPUを使用
        'n_estimators': 500, # 暫定的な値
        'learning_rate': 0.05, # 暫定的な値
        'num_leaves': 31, # 暫定的な値
        'max_depth': -1, # 暫定的な値
        'random_state': 42,
        'verbose': -1
    }

    total_rows = len(X)
    # ウォークフォワード分析のための分割サイズを計算
    # train_size + valid_size が全体の何割になるか
    fold_ratio = train_ratio + validation_ratio
    # 1回のウォークフォワードで進むステップサイズ（検証データサイズに相当）
    step_size = int(total_rows * validation_ratio)
    # 最初の訓練期間の終了インデックス
    initial_train_end = int(total_rows * train_ratio)

    feature_importance_df = pd.DataFrame() # 各Foldの重要度を格納

    print(f"\n--- 累積重要度分析開始 (Walk-Forward, {n_splits} splits) ---")

    # ウォークフォワード分析
    for i in range(n_splits):
        # 訓練期間の開始・終了インデックス
        train_start_idx = i * step_size
        train_end_idx = initial_train_end + i * step_size
        # 検証期間の開始・終了インデックス (ギャップを考慮)
        valid_start_idx = train_end_idx + (WINDOW_SIZE - 1)
        valid_end_idx = valid_start_idx + step_size

        # インデックスがデータ範囲を超えないように調整
        train_end_idx = min(train_end_idx, total_rows)
        valid_start_idx = min(valid_start_idx, total_rows)
        valid_end_idx = min(valid_end_idx, total_rows)

        # 訓練データ・検証データが空にならないかチェック
        if train_start_idx >= train_end_idx or valid_start_idx >= valid_end_idx:
            print(f"Fold {i+1}: データが不足しているためスキップします。")
            continue

        X_train = X.iloc[train_start_idx:train_end_idx]
        y_train = y.iloc[train_start_idx:train_end_idx]
        X_valid = X.iloc[valid_start_idx:valid_end_idx]
        y_valid = y.iloc[valid_start_idx:valid_end_idx]

        print(f"Fold {i+1}: Train [{train_start_idx}:{train_end_idx-1}], Valid [{valid_start_idx}:{valid_end_idx-1}]")

        # モデル学習 (クラス重みはここでは考慮しない)
        try:
            model = lgbm.LGBMClassifier(**params)
            model.fit(
                X_train, y_train.values.ravel(),
                eval_set=[(X_valid, y_valid.values.ravel())],
                eval_metric='multi_logloss'
            )

            # 特徴量重要度の保存
            fold_importance = pd.DataFrame()
            fold_importance['feature'] = X.columns
            fold_importance['importance'] = model.feature_importances_
            fold_importance['fold'] = i + 1 # Fold番号を追加
            feature_importance_df = pd.concat([feature_importance_df, fold_importance], ignore_index=True)

        except ValueError as ve:
             print(f"Fold {i+1} のモデル学習中にエラーが発生しました: {ve}")
             # 特定のFoldでエラーが発生しても、他のFoldの処理は続ける
        except lgbm.basic.LightGBMError as lgbm_err:
             print(f"Fold {i+1} のモデル学習中にLightGBMエラーが発生しました: {lgbm_err}")

    if feature_importance_df.empty:
         print("エラー: 全てのFoldで特徴量重要度を計算できませんでした。")
         # 空のDataFrameと空のリストを返す
         return pd.DataFrame(columns=['feature', 'importance', 'cumulative_importance']), []

    # 平均特徴量重要度の計算
    mean_importance = (feature_importance_df.groupby('feature')['importance']
                      .mean()
                      .sort_values(ascending=False))

    # 累積重要度の計算
    cumulative_importance_df = pd.DataFrame({
        'feature': mean_importance.index,
        'importance': mean_importance.values
    })
    # 重要度が0の特徴量を除外してから累積を計算（数値誤差対策）
    cumulative_importance_df = cumulative_importance_df[cumulative_importance_df['importance'] > 1e-8]
    cumulative_importance_df['cumulative_importance'] = (cumulative_importance_df['importance'].cumsum()
                                                        / cumulative_importance_df['importance'].sum())

    # 閾値に基づく特徴量選択
    selected_features = cumulative_importance_df[cumulative_importance_df['cumulative_importance'] <= threshold]['feature'].tolist()

    # もし閾値以下の特徴量が一つもない場合（最初の特徴量だけで閾値を超えた場合）、
    # 最も重要な特徴量一つだけを選択する
    if not selected_features and not cumulative_importance_df.empty:
        selected_features = [cumulative_importance_df.iloc[0]['feature']]
        print(f"警告: 累積重要度{threshold*100}%を満たす特徴量が1つもありません。最も重要な特徴量 '{selected_features[0]}' を選択します。")
    elif not selected_features: # cumulative_importance_df も空の場合
         print("エラー: 有効な特徴量が見つかりませんでした。")
         return cumulative_importance_df, [] # 空のリストを返す

    print(f"--- 累積重要度分析完了 ---")
    print(f"累積重要度 {threshold*100}% に達するまでの特徴量数: {len(selected_features)}")

    return cumulative_importance_df, selected_features

# --- 累積重要度プロット関数 ---
# コメントアウト解除
def plot_cumulative_importance(cumulative_importance_df: pd.DataFrame, threshold: float, output_dir: str):
    """
    累積重要度のプロットを作成し、保存する。

    Args:
        cumulative_importance_df (pd.DataFrame): 累積重要度を含むDataFrame。
        threshold (float): 累積重要度の閾値。
        output_dir (str): 図の保存先ディレクトリ。
    """
    if cumulative_importance_df.empty:
        print("警告: 累積重要度データが空のため、プロットをスキップします。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # 1行2列に変更

    # 累積重要度プロット (左側)
    ax1 = axes[0]
    num_features = len(cumulative_importance_df)
    ax1.plot(range(1, num_features + 1),
             cumulative_importance_df['cumulative_importance'],
             marker='.', linestyle='-')
    # 閾値ライン
    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100:.0f}% threshold')
    # 閾値を満たす特徴量数で垂直線
    n_selected = len(cumulative_importance_df[cumulative_importance_df['cumulative_importance'] <= threshold])
    if n_selected == 0 and num_features > 0: # 最初の特徴量だけで超えた場合
        n_selected = 1
    if n_selected > 0:
        ax1.axvline(x=n_selected, color='g', linestyle=':', label=f'{n_selected} features')

    ax1.set_xlabel('Number of Features (Sorted by Importance)')
    ax1.set_ylabel('Cumulative Importance')
    ax1.set_title('Cumulative Feature Importance')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend()
    # X軸の範囲を調整
    ax1.set_xlim(0, num_features + 1)
    ax1.set_ylim(0, 1.05) # Y軸の上限を少し上げる

    # 個別の特徴量重要度プロット (右側、上位30件)
    ax2 = axes[1]
    top_n_plot = min(30, num_features) # 上位30件まで表示
    importance_to_plot = cumulative_importance_df.head(top_n_plot)
    ax2.barh(range(top_n_plot), importance_to_plot['importance'][::-1]) # 降順でプロットするため逆順に
    ax2.set_yticks(range(top_n_plot))
    ax2.set_yticklabels(importance_to_plot['feature'][::-1]) # ラベルも逆順に
    ax2.set_xlabel('Mean Importance')
    ax2.set_ylabel('Feature')
    ax2.set_title(f'Individual Feature Importance (Top {top_n_plot})')
    ax2.grid(True, axis='x', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # 図を保存
    save_figure(fig, 'cumulative_feature_importance', output_dir)

    # 表示フラグがTrueの場合のみ表示
    if SHOW_FIGURES:
        plt.show()
    # else: plt.close(fig) # save_figure内で閉じるので不要

def optimize_lightgbm(X: pd.DataFrame, y: pd.DataFrame, n_trials: int = 100) -> optuna.study.Study:
    """
    Optunaを使用してLightGBMのハイパーパラメータとクラスウェイトを最適化し、結果をSQLiteに保存
    
    Args:
        X: 特徴量DataFrame
        y: 目的変数DataFrame
        n_trials: 最適化試行回数
        
    Returns:
        optuna.study.Study: Optunaの最適化結果
    """
    def objective(trial, X=X, y=y, n_splits=3):
        """Optunaの目的関数"""
        # ハイパーパラメータの探索空間を定義
        param = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            # 'device': 'gpu',  # GPUを使用する設定 -> CPUに変更
            'device': 'cpu',   # CPUを使用する設定 (ColabでのOpenCLエラー回避)
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbose': -1
            # class_weight はここで指定しない
        }

        # クラスウェイトの探索空間を定義 (paramと同様の形式に統一)
        class_weight_dict = {
            0: trial.suggest_float('weight_class_0', 0.1, 10.0), # クラス0のウェイト
            1: trial.suggest_float('weight_class_1', 0.1, 10.0), # クラス1のウェイト
            2: trial.suggest_float('weight_class_2', 0.1, 10.0)  # クラス2のウェイト
        }

        # --- 変更箇所: TimeSeriesSplit に gap を追加 ---
        # 時系列データの特性を尊重した層化分割の実装
        # TimeSeriesSplitでデータを時系列に沿って分割し、ウィンドウサイズ分のギャップを設ける
        # gap = WINDOW_SIZE - 1 とすることで、訓練期間と検証期間のウィンドウが重ならないようにする
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=WINDOW_SIZE - 1)
        # -------------------------------------------------
        f1_scores = []

        # 時系列交差検証
        for fold, (train_idx, valid_idx) in enumerate(tscv.split(X)):
            print(f"Fold {fold+1}/{n_splits}の最適化中...")
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            
            # 訓練データと検証データのクラス分布を確認
            train_class_counts = y_train['actions'].value_counts()
            valid_class_counts = y_valid['actions'].value_counts()
            print(f"訓練データのクラス分布: {dict(train_class_counts)}")
            print(f"検証データのクラス分布: {dict(valid_class_counts)}")
            
            # モデルの初期化時にクラスウェイトを渡す
            model = lgbm.LGBMClassifier(**param, class_weight=class_weight_dict)
            
            # fitメソッドでは class_weight を指定しない
            model.fit(
                X_train, y_train.values.ravel(),
                eval_set=[(X_valid, y_valid.values.ravel())],
                eval_metric='multi_logloss'
            )

            # 検証データでの予測
            y_pred = model.predict(X_valid)
            
            # マルチクラス分類のためにmacro平均のF1スコアを使用
            fold_f1 = f1_score(y_valid, y_pred, average='macro')
            f1_scores.append(fold_f1)
            print(f"Fold {fold+1} F1スコア: {fold_f1:.4f}")

        # 平均F1スコアを最大化
        mean_f1 = np.mean(f1_scores)
        print(f"平均F1スコア: {mean_f1:.4f}")
        return mean_f1

    # 最適化処理
    # プロジェクトルートを取得
    project_root = get_project_root()
    # --- データベースファイルのパスを変更 ---
    # studies サブディレクトリを指定
    studies_dir = os.path.join(project_root, 'feature_analysis', 'studies')
    db_path = os.path.join(studies_dir, 'optuna_study.db')
    # ------------------------------------
    storage_name = f"sqlite:///{db_path}"
    study_name = "lightgbm-hyperparameter-optimization" # study名を指定

    # --- studies ディレクトリが存在しない場合は作成 ---
    try:
        os.makedirs(studies_dir, exist_ok=True)
        print(f"Optuna studies ディレクトリを確認/作成しました: {studies_dir}")
    except OSError as e:
        print(f"Optuna studies ディレクトリの作成中にエラーが発生しました: {e}")
        # ディレクトリ作成に失敗した場合、study作成に進むとエラーになる可能性が高い
        # ここで処理を中断するか、エラーをraiseするか検討が必要
        # 今回は警告のみ表示し、続行を試みる
    # ---------------------------------------------

    # --- データベースの存在確認とメッセージ表示 ---
    if os.path.exists(db_path):
        print(f"既存のOptunaデータベースファイルをロードします: {db_path}")
    else:
        print(f"Optunaデータベースファイルが見つかりません。新規に作成します: {db_path}")
    # -----------------------------------------

    print(f"Optuna studyをデータベースに保存/ロードします: {db_path}")

    # studyを作成（存在すればロード）
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',  # F1スコアを最大化
        load_if_exists=True # 既存のstudyがあればロードする
    )
    
    # 最適化を実行
    study.optimize(lambda trial: objective(trial), n_trials=n_trials)

    print('最良のトライアル:')
    trial = study.best_trial

    print(f'  値(F1スコア): {trial.value}')
    print('  パラメータ:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    return study

def plot_optimization_history(study: optuna.study.Study):
    """
    最適化の履歴を可視化
    
    Args:
        study: Optunaの最適化結果
    """
    plt.figure(figsize=(12, 4))

    # 最適化履歴
    plt.subplot(1, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('最適化履歴')

    # パラメータの重要度
    plt.subplot(1, 2, 2)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('パラメータ重要度')

    plt.tight_layout()
    plt.show()

# 例: モデルやデータの保存を行う関数内
def train_optimized_model(X: pd.DataFrame, y: pd.DataFrame, best_params: Dict[str, Any],
                         train_ratio: float = 0.6, valid_ratio: float = 0.2,
                         class_weight_dict: Optional[Dict[int, float]] = None) -> Tuple[lgbm.LGBMClassifier, Dict[str, float], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    最適化されたパラメータでモデルを学習し、結果を返す。
    モデルは feature_analysis/models に、
    データは feature_analysis/outputs に保存する。
    データ分割時にウィンドウサイズを考慮したギャップを設ける。
    """
    total_rows = len(X)
    # --- 変更箇所: データ分割にギャップを追加 ---
    train_end_idx = int(total_rows * train_ratio)
    # 検証データの開始インデックス (訓練終了 + ギャップ)
    valid_start_idx = train_end_idx + (WINDOW_SIZE - 1)
    valid_end_idx = valid_start_idx + int(total_rows * valid_ratio)
    # テストデータの開始インデックス (検証終了 + ギャップ)
    test_start_idx = valid_end_idx + (WINDOW_SIZE - 1)

    # インデックスがデータ範囲を超えないように調整
    valid_start_idx = min(valid_start_idx, total_rows)
    valid_end_idx = min(valid_end_idx, total_rows)
    test_start_idx = min(test_start_idx, total_rows)

    # .iloc を使用して分割 (ギャップを考慮)
    X_train = X.iloc[:train_end_idx]
    y_train = y.iloc[:train_end_idx]

    X_valid = X.iloc[valid_start_idx:valid_end_idx]
    y_valid = y.iloc[valid_start_idx:valid_end_idx]

    X_test = X.iloc[test_start_idx:]
    y_test = y.iloc[test_start_idx:]

    print(f"データ分割 (ギャップ {WINDOW_SIZE - 1} を考慮):")
    print(f"  Train: index 0 ~ {train_end_idx-1} (size: {len(X_train)})")
    print(f"  Valid: index {valid_start_idx} ~ {valid_end_idx-1} (size: {len(X_valid)})")
    print(f"  Test:  index {test_start_idx} ~ {total_rows-1} (size: {len(X_test)})")

    # データが空になっていないかチェック
    if X_valid.empty or X_test.empty:
        print("警告: データ分割により検証データまたはテストデータが空になりました。データ量や比率を確認してください。")
        # 空の場合は処理を中断するか、エラーを発生させるか、あるいは空のDataFrameを返すなどの対応が必要
        # ここでは警告のみ表示し、処理を続行するが、後続処理でエラーになる可能性あり
    # -----------------------------------------

    # --- optimized_class_weights を初期化 ---
    optimized_class_weights = {}
    # --------------------------------------
    # Optunaで最適化されたクラスウェイトをbest_paramsから抽出
    params_to_remove = []
    for key, value in best_params.items():
        if key.startswith('weight_class_'):
            class_index = int(key.split('_')[-1])
            optimized_class_weights[class_index] = value
            params_to_remove.append(key) # モデルパラメータとしては不要なので削除リストに追加

    # モデルパラメータからクラスウェイト関連のキーを削除
    model_params = {k: v for k, v in best_params.items() if k not in params_to_remove}

    # 基本的なパラメータ（multiclass等）を追加
    if 'objective' not in model_params:
        model_params['objective'] = 'multiclass'
    if 'num_class' not in model_params:
        model_params['num_class'] = 3
    if 'metric' not in model_params:
        model_params['metric'] = 'multi_logloss'
    if 'random_state' not in model_params:
        model_params['random_state'] = 42
    if 'verbose' not in model_params:
        model_params['verbose'] = -1

    # モデルの初期化時にクラスウェイトを渡す
    final_class_weight = None
    if optimized_class_weights:
        print(f"Optunaで最適化されたクラス重みを適用してモデルを学習します: {optimized_class_weights}")
        final_class_weight = optimized_class_weights
    elif class_weight_dict: # Optunaで最適化されなかった場合のフォールバック
        print(f"引数で渡されたクラス重みを適用してモデルを学習します: {class_weight_dict}")
        final_class_weight = class_weight_dict
    else:
        print("クラス重みを適用せずにモデルを学習します。")
        
    # モデル初期化時に final_class_weight を渡す
    model = lgbm.LGBMClassifier(**model_params, class_weight=final_class_weight) 
    
    # fitメソッドでは class_weight を指定しない
    # 訓練データが空でない場合のみ学習を実行
    if not X_train.empty and not y_train.empty:
        # 検証データも空でない場合のみ eval_set に含める
        eval_set = []
        if not X_valid.empty and not y_valid.empty:
            eval_set = [(X_valid, y_valid.values.ravel())]
        
        model.fit(
            X_train, y_train.values.ravel(),
            eval_set=eval_set,
            eval_metric='multi_logloss'
        )
    else:
        print("エラー: 訓練データが空のため、モデル学習をスキップします。")
        # 空のモデルや結果を返すか、エラーを発生させる
        # ここでは学習済みでないモデルと空の結果を返す想定
        results = {'train': np.nan, 'valid': np.nan, 'test': np.nan}
        return model, results, X_valid, y_valid, X_test, y_test


    # 各データセットでの評価
    results = {}
    # 評価対象データが空でないか確認してから評価を実行
    for name, X_eval, y_eval in [
        ('train', X_train, y_train),
        ('valid', X_valid, y_valid),
        ('test', X_test, y_test)
    ]:
        if not X_eval.empty and not y_eval.empty:
            try:
                y_pred = model.predict(X_eval)
                accuracy = accuracy_score(y_eval, y_pred)
                results[name] = accuracy
                print(f'{name.capitalize()} 精度: {accuracy:.4f}')
            except Exception as e: # 予測や評価中のエラー
                 print(f"{name.capitalize()} データでの評価中にエラーが発生しました: {e}")
                 results[name] = np.nan
        else:
            print(f"{name.capitalize()} データが空のため、評価をスキップします。")
            results[name] = np.nan


    # --- 保存用ディレクトリの準備 ---
    project_root = get_project_root()
    # --- パス修正 (feature_importance -> feature_analysis) ---
    model_dir = os.path.join(project_root, 'feature_analysis', 'models')
    output_dir = os.path.join(project_root, 'feature_analysis', 'outputs')
    # ---------------
    try:
        os.makedirs(model_dir, exist_ok=True)  # モデル用ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True) # outputs用ディレクトリ作成
    except OSError as e:
        print(f"保存用ディレクトリの作成に失敗しました: {e}")
        return model, results, X_valid, y_valid, X_test, y_test # 保存せずに返す

    # --- モデルの保存 ---
    model_path = os.path.join(model_dir, 'lgbm_model.txt')
    try:
        # LGBMClassifierオブジェクトのbooster_属性からモデルを保存
        model.booster_.save_model(model_path)
        print(f"学習済みモデルを保存しました: {model_path}")
    except AttributeError:
        print("エラー: モデルのBoosterオブジェクトが見つかりません。モデルは正常に学習されましたか？")
    except lgbm.basic.LightGBMError as lgbm_err:
        print(f"モデルの保存中にLightGBMエラーが発生しました ({model_path}): {lgbm_err}")
    except IOError as e:
        print(f"モデルファイルの書き込み中にエラーが発生しました ({model_path}): {e}")

    # --- 検証/テストデータの保存 ---
    # 保存対象データが空でないか確認してから保存
    valid_data_path_X = os.path.join(output_dir, 'validation_data_X.csv')
    valid_data_path_y = os.path.join(output_dir, 'validation_data_y.csv')
    try:
        if not X_valid.empty:
            X_valid.to_csv(valid_data_path_X)
        if not y_valid.empty:
            y_valid.to_csv(valid_data_path_y, header=True)
        if not X_valid.empty or not y_valid.empty:
             print(f"検証データを保存しました: {valid_data_path_X}, {valid_data_path_y}")
        else:
             print("検証データが空のため、保存をスキップしました。")
    except IOError as e:
        print(f"検証データのファイル書き込み中にエラーが発生しました: {e}")

    test_data_path_X = os.path.join(output_dir, 'test_data_X.csv')
    test_data_path_y = os.path.join(output_dir, 'test_data_y.csv')
    try:
        if not X_test.empty:
            X_test.to_csv(test_data_path_X)
        if not y_test.empty:
            y_test.to_csv(test_data_path_y, header=True)
        if not X_test.empty or not y_test.empty:
            print(f"テストデータを保存しました: {test_data_path_X}, {test_data_path_y}")
        else:
            print("テストデータが空のため、保存をスキップします。")
    except IOError as e:
        print(f"テストデータのファイル書き込み中にエラーが発生しました: {e}")

    return model, results, X_valid, y_valid, X_test, y_test

def visualize_model_evaluation(model: lgbm.LGBMClassifier, X: pd.DataFrame, y: pd.DataFrame):
    """
    モデルの評価結果を様々な観点から可視化
    
    Args:
        model: 訓練済みのLGBMClassifierモデル
        X: 特徴量DataFrame
        y: 目的変数DataFrame
    """
    # seabornのスタイル設定
    sns.set_style("whitegrid")
    
    fig = plt.figure(figsize=(15, 10))

    # 混同行列の作成と可視化
    plt.subplot(2, 2, 1)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 特徴量重要度の可視化
    plt.subplot(2, 2, 2)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    n_features = min(20, len(importance))  # 上位20特徴量まで表示
    importance_tail = importance.tail(n_features)
    
    plt.barh(range(len(importance_tail)), importance_tail['importance'])
    plt.yticks(range(len(importance_tail)), importance_tail['feature'])
    plt.title('Feature Importance (Top 20)')
    plt.xlabel('Importance')

    # クラスごとの予測確率分布
    if hasattr(model, 'predict_proba'):
        plt.subplot(2, 2, 3)
        probs = model.predict_proba(X)
        for i in range(probs.shape[1]):
            sns.kdeplot(probs[:, i], label=f'Class {i}')
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend()

    # 学習曲線（検証用スコア）
    if hasattr(model, 'evals_result_'):
        plt.subplot(2, 2, 4)
        results = model.evals_result_
        for metric in results['valid_0'].keys():
            plt.plot(results['valid_0'][metric], label=metric)
        plt.title('Learning Curves')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()

    plt.tight_layout()
    plt.show()

def print_feature_importance_ranking(model: lgbm.LGBMClassifier, feature_names: List[str]):
    """
    特徴量重要度のランキングを出力
    
    Args:
        model: 訓練済みのLGBMClassifierモデル
        feature_names: 特徴量名のリスト
    
    Returns:
        pd.DataFrame: 重要度順にソートされた特徴量重要度DataFrame
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    print("\n特徴量重要度ランキング:")
    print("============================")
    for i, (feature, imp) in enumerate(zip(importance['feature'], importance['importance']), 1):
        print(f"{i:2d}. {feature:<30} {imp:.4f}")
    
    # 累積寄与率も計算
    importance['cumulative_importance'] = importance['importance'].cumsum() / importance['importance'].sum()
    print("\n累積重要度:")
    print("=====================")
    for threshold in [0.5, 0.7, 0.9, 0.95]:
        n_features = len(importance[importance['cumulative_importance'] <= threshold])
        if n_features == 0:  # 最初の特徴量だけで閾値を超えた場合
            n_features = 1
        print(f"{threshold*100}%の重要度に必要な特徴量数: {n_features}")
    
    return importance

def save_top_features_to_yaml(importance_df: pd.DataFrame, n_top: int,
                             multimodal_yaml_path: str):
    """
    重要度の上位n個の特徴量をマルチモーダル設定YAMLファイルに保存
    """
    if os.path.exists(multimodal_yaml_path):
        try:
            # 既存の設定を読み込む
            with open(multimodal_yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # timeseries セクションがなければ追加
            if 'timeseries' not in config or not isinstance(config['timeseries'], dict):
                config['timeseries'] = {}
            # 上位n個の特徴量を取得
            top_features = importance_df.head(n_top)['feature'].tolist()
            # timeseries 配下を更新
            config['timeseries']['feature_columns'] = top_features
            config['timeseries']['feature_dim']    = len(top_features)

            # 更新した設定を保存
            with open(multimodal_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            print(f"{multimodal_yaml_path} の timeseries 設定を更新しました：上位 {n_top} 特徴量")
        except yaml.YAMLError as e:
            print(f"YAML 処理エラー ({multimodal_yaml_path}): {e}")
        except IOError as e:
            print(f"I/O エラー ({multimodal_yaml_path}): {e}")
    else:
        print(f"Warning: 設定ファイルが見つかりません ({multimodal_yaml_path})。更新をスキップします。")


def main():
    # --- main 関数の冒頭で plt をインポート ---
    import matplotlib.pyplot as plt
    # -----------------------------------------

    # Matplotlibのインタラクティブモード設定
    if SHOW_FIGURES:
        plt.ion()
    else:
        plt.ioff()

    # プロジェクトのルートディレクトリを取得
    project_root = get_project_root()
    print(f"プロジェクトルートディレクトリ: {project_root}")
    # --- パス修正 (feature_importance -> feature_analysis) ---
    output_dir = os.path.join(project_root, 'feature_analysis', 'outputs') # output_dir を定義

    # データパス
    actions_path = os.path.join(project_root, 'data/fix_labeled_data_timeseries_15m.csv')
    indicators_path = os.path.join(project_root, 'data/timeseries_15m_202412301431.csv')

    # データの前処理
    X_orig, y_orig, class_weight_dict = preprocess_data(actions_path, indicators_path) # 元のデータを保持

    # --- 特徴量選択ステップ ---
    selected_features: Optional[List[str]] = None # 初期化
    X = X_orig.copy() # 最適化・学習に使用するXを初期化
    y = y_orig.copy()

    if USE_FEATURE_SELECTION:
        # 累積重要度分析を実行
        cumulative_importance_df, selected_features_list = analyze_cumulative_importance(
            X_orig, y_orig, threshold=CUMULATIVE_IMPORTANCE_THRESHOLD
        )

        if selected_features_list:
            # 累積重要度をプロット・保存
            plot_cumulative_importance(cumulative_importance_df, CUMULATIVE_IMPORTANCE_THRESHOLD, output_dir)

            # 選択された特徴量でXをフィルタリング
            selected_features = selected_features_list # 後で使うために保持
            X = X_orig[selected_features] # Xを更新
            print(f"\n選択された特徴量 ({len(selected_features)}個):")
            print(selected_features)
            print(f"特徴量選択後のXの形状: {X.shape}")
        else:
            print("警告: 特徴量選択に失敗したため、全ての特徴量を使用します。")
            # selected_features は None のまま
            # X は X_orig のまま
    else:
        print("\n累積重要度に基づく特徴量選択はスキップされました。")
    # -------------------------

    # ステップ1: n_estimatorsの最適化 (コメントアウト)
    # n_estimators_list = [100, 200, 500, 1000, 1500, 2000]
    # results_n_est = compare_n_estimators(X, y, n_estimators_list)
    # best_n_est = results_n_est.groupby('n_estimators')['accuracy'].mean().idxmax()
    # print(f"\n平均精度に基づく最適なn_estimators: {best_n_est}")
    
    # ステップ2: 特徴量選択の閾値テスト (コメントアウト)
    # thresholds = [0.80, 0.90, 0.95, 0.99]
    # results_threshold = {}
    # cumulative_importance = None # 初期化
    # for threshold in thresholds:
    #     print(f"\n閾値{threshold}での分析")
    #     cumulative_importance, selected_features, _ = analyze_cumulative_importance(
    #         X, y, threshold=threshold
    #     )
    #     results_threshold[threshold] = {
    #         'n_features': len(selected_features),
    #         'features': selected_features
    #     }
    #     print(f"選択された特徴量数: {len(selected_features)}")
    # if cumulative_importance is not None:
    #     plot_cumulative_importance(cumulative_importance)

    # ステップ3: Optunaによるハイパーパラメータとクラスウェイトの最適化
    # (objective 関数内で TimeSeriesSplit(gap=WINDOW_SIZE-1) が使われる)
    # 特徴量選択後のX, yを使用
    print("\nOptunaによるハイパーパラメータとクラスウェイトの最適化を開始します...")
    study = optimize_lightgbm(X, y, n_trials=100)  # 時間短縮のためn_trialsを調整可能

    # 最適化結果の可視化
    # Optunaの可視化関数はmatplotlibに依存するため、必要に応じてインポート
    try:
        # import matplotlib.pyplot as plt # main 関数の冒頭でインポート済み
        try:
            import optuna.visualization.matplotlib as vis
            optuna_vis_available = True
        except ImportError:
            print("optuna.visualization.matplotlibが利用できません。可視化をスキップします。")
            optuna_vis_available = False
            
        # 日本語表示が必要な場合はjapanize_matplotlibをインポート
        if optuna_vis_available:
            try:
                import japanize_matplotlib
                japanize_matplotlib.japanize()
                print("japanize_matplotlibを適用しました。")
            except ImportError:
                print("japanize_matplotlibが見つかりません。日本語が文字化けする可能性があります。")

        # 最適化履歴のプロットと保存
        if optuna_vis_available:
            fig_opt_hist = vis.plot_optimization_history(study)
            # save_figure の呼び出しで output_dir を渡す
            save_figure(fig_opt_hist.figure, 'optimization_history', output_dir)
            if SHOW_FIGURES: plt.show()
            # else: plt.close(fig_opt_hist.figure) # save_figure内で閉じるので不要

        # パラメータ重要度のプロットと保存
        if optuna_vis_available:
            fig_param_imp = vis.plot_param_importances(study)
            # save_figure の呼び出しで output_dir を渡す
            save_figure(fig_param_imp.figure, 'param_importances', output_dir)
            if SHOW_FIGURES: plt.show()
            # else: plt.close(fig_param_imp.figure) # save_figure内で閉じるので不要

    except Exception as e: # 可視化中の予期せぬエラー
        print(f"最適化結果の可視化中にエラーが発生しました: {e}")

    # ステップ4: 最適化されたパラメータでモデル学習（Optunaの結果を使用）
    # (train_optimized_model 内でギャップ付きデータ分割が行われる)
    # 特徴量選択後のX, yを使用
    print("\n最適化されたパラメータを使用してモデルの学習を開始します...")
    # train_optimized_model は best_params からクラスウェイトを抽出し、初期化時に使用する
    # class_weight_dict は Optuna で最適化されなかった場合のフォールバックとして渡す
    model, results, X_valid, y_valid, X_test, y_test = train_optimized_model(
        X, y, study.best_params, class_weight_dict=class_weight_dict
    )

    # ステップ5: モデルの評価と可視化 (テストデータを使用)
    print("\nテストデータでのモデル評価:")
    # テストデータが空でない場合のみ評価と可視化を実行
    if not X_test.empty and not y_test.empty:
        # visualize_model_evaluation は内部で plt.figure を呼ぶので、その Figure を取得
        # fig_eval = plt.figure(figsize=(15, 10)) # Figureオブジェクトを事前に作成 -> visualize内で作成されるはず
        # visualize_model_evaluation を呼び出し、返り値の Figure を受け取る
        fig_eval = visualize_model_evaluation(model, X_test, y_test) # 修正: 返り値を受け取る
        if fig_eval: # Figureが正常に作成された場合のみ保存
            # save_figure の呼び出しで output_dir を渡す
            save_figure(fig_eval, 'model_evaluation_test_data', output_dir) # 図を保存
            if SHOW_FIGURES:
                plt.show()
            # else: plt.close(fig_eval) # save_figure内で閉じるので不要
        else:
             print("警告: モデル評価の可視化中にエラーが発生したため、図の保存をスキップします。")
    else:
        print("テストデータが空のため、モデル評価の可視化をスキップします。")


    # ステップ6: 特徴量重要度のランキング表示と保存
    # モデルが学習されているか（訓練データが空でなかったか）確認
    if hasattr(model, 'feature_importances_'):
        # モデルが学習に使用した特徴量名を取得 (X.columns)
        feature_names_used = X.columns.tolist()
        importance_df = print_feature_importance_ranking(model, feature_names_used)

        # ステップ7: 上位6個の特徴量をYAMLに保存
        top_n = 6  # 上位特徴量数

        # 設定ファイルのパス（multimodalフォルダではなく直接configsフォルダ下）
        yaml_path = os.path.join(project_root, 'configs/config.yaml')
        
        # 設定ファイルに保存
        save_top_features_to_yaml(importance_df, top_n, yaml_path)
        
        # Colab用の設定ファイルも更新
        yaml_colab_path = os.path.join(project_root, 'configs/config_for_google_colab.yaml')
        save_top_features_to_yaml(importance_df, top_n, yaml_colab_path)
    else:
        print("モデルが学習されなかったため、特徴量重要度の表示とYAML保存をスキップします。")


    print("\n処理が完了しました")

if __name__ == "__main__":
    # スクリプトのエントリーポイント
    main()