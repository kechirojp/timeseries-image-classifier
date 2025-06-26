import os
import pandas as pd
import numpy as np
import lightgbm as lgbm
import json
from typing import Dict, Optional, Any, List # List をインポート

# --- プロジェクトルートパス取得 ---
def get_project_root() -> str:
    """
    プロジェクトのルートディレクトリパスを動的に取得します。
    このスクリリプトが配置されているディレクトリの親ディレクトリをルートと仮定します。

    Returns:
        str: プロジェクトルートの絶対パス。
    """
    # このスクリプトの絶対パスを取得
    script_path = os.path.abspath(__file__)
    # このスクリプトが配置されているディレクトリ (例: /path/to/project/feature_analysis)
    script_dir = os.path.dirname(script_path)
    # 親ディレクトリ（プロジェクトルート）
    project_root = os.path.dirname(script_dir)
    return project_root

# --- モデル・しきい値のロード関数 ---
def load_model(model_path: str) -> Optional[lgbm.Booster]:
    """
    指定されたパスからLightGBMモデルファイルをロードします。

    Args:
        model_path (str): モデルファイル (.txt) へのパス。

    Returns:
        Optional[lgbm.Booster]: ロードされたLightGBM Boosterオブジェクト。
                                 ロードに失敗した場合はNone。
    """
    try:
        # --- 変更箇所: ファイルパスではなく、ファイル内容を読み込んで渡す ---
        # ファイルをUTF-8エンコーディングで読み込む
        with open(model_path, 'r', encoding='utf-8') as f:
            model_string = f.read()
        # 文字列からモデルをロード
        model_booster = lgbm.Booster(model_str=model_string)
        # ----------------------------------------------------------
        print(f"モデルをロードしました: {model_path}")
        return model_booster
    except FileNotFoundError:
        # 指定されたパスにファイルが存在しない場合
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        return None
    except lgbm.basic.LightGBMError as lgbm_err:
        # LightGBM固有のロードエラー (モデル文字列パースエラーなど)
        print(f"モデルのロード中にLightGBMエラーが発生しました: {lgbm_err}")
        return None
    except IOError as e:
        # ファイル読み込み権限がない場合など
        print(f"モデルファイルの読み込み中にIOエラーが発生しました ({model_path}): {e}")
        return None
    except UnicodeDecodeError as ude:
        # UTF-8での読み込みに失敗した場合 (ファイルがUTF-8でない可能性)
        print(f"モデルファイルのデコード中にエラーが発生しました ({model_path}): {ude}")
        return None

# 関数名を変更: load_thresholds -> load_class_thresholds
def load_class_thresholds(thresholds_path: str) -> Optional[Dict[str, float]]:
    """
    指定されたパスから最適化された分類しきい値を含むJSONファイルをロードします。

    Args:
        thresholds_path (str): しきい値JSONファイルへのパス。

    Returns:
        Optional[Dict[str, float]]: しきい値を含む辞書 ('class1_threshold'など)。
                                     ロードまたはパースに失敗した場合はNone。
    """
    try:
        # JSONファイルを読み込みモードで開く
        with open(thresholds_path, 'r') as f:
            # JSONデータをPython辞書にパースする
            thresholds_data = json.load(f)
        # 'thresholds' キーの下にしきい値があると仮定
        thresholds = thresholds_data.get('thresholds')
        if thresholds and isinstance(thresholds, dict):
             print(f"分類しきい値をロードしました: {thresholds_path}")
             return thresholds
        else:
             print(f"エラー: しきい値ファイルに 'thresholds' キーが見つからないか、形式が不正です: {thresholds_path}")
             return None
    except FileNotFoundError:
        # 指定されたパスにファイルが存在しない場合
        print(f"エラー: 分類しきい値ファイルが見つかりません: {thresholds_path}")
        return None
    except json.JSONDecodeError as json_err:
        # JSONファイルの形式が不正な場合
        print(f"分類しきい値ファイルのJSONデコード中にエラーが発生しました: {json_err}")
        return None
    except IOError as e:
        # ファイル読み込み権限がない場合など
        print(f"分類しきい値ファイルの読み込み中にIOエラーが発生しました ({thresholds_path}): {e}")
        return None

# --- クラス分類決定関数 ---
# 関数名を変更: get_trading_action -> get_classification
def get_classification(model_booster: lgbm.Booster,
                       data: pd.DataFrame,
                       thresholds: Dict[str, float]) -> int:
    """
    学習済みモデル、入力データ、しきい値に基づいてクラス (0, 1, 2) を決定します。

    Args:
        model_booster (lgbm.Booster): 学習済みのLightGBM Boosterオブジェクト。
        data (pd.DataFrame): 分類を行うための特徴量データ (1サンプル以上)。
        thresholds (Dict[str, float]): 最適化されたしきい値 ('class1_threshold'など)。

    Returns:
        int: 決定されたクラス (0, 1, 2)。エラー時は -1。
             入力データが複数サンプルの場合、最初のサンプルのクラスを返す。
    """
    if model_booster is None or data.empty or thresholds is None:
        # 必要なデータが不足している場合
        print("エラー: モデル、データ、またはしきい値が不足しているため、クラスを決定できません。")
        return -1 # エラーを示す値

    try:
        # モデルを使って予測確率を計算 (Nサンプル x 3クラス)
        y_pred_proba = model_booster.predict(data)

        # 最初のサンプルの予測確率を取得
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[0] > 0:
            probs = y_pred_proba[0]
        else:
            print(f"エラー: 予測確率の形式が不正です (shape: {y_pred_proba.shape})。")
            return -1

        # しきい値を取得 (キー名を変更)
        class1_threshold = thresholds.get("class1_threshold", 0.7)
        class0_threshold = thresholds.get("class0_threshold", 0.7)
        class2_threshold = thresholds.get("class2_threshold", 0.6)
        prob_diff_threshold = thresholds.get("prob_diff_threshold", 0.2)

        # 予測確率に基づいてクラスを決定 (変数名を変更)
        prob_class0, prob_class1, prob_class2 = probs[0], probs[1], probs[2]
        classification = 2 # デフォルトはクラス2

        # クラス2条件 (優先)
        if prob_class2 >= class2_threshold and prob_class2 > prob_class1 and prob_class2 > prob_class0:
            classification = 2
        # クラス1条件
        elif prob_class1 >= class1_threshold and prob_class1 > prob_class0 and (prob_class1 - prob_class0) > prob_diff_threshold:
            classification = 1
        # クラス0条件
        elif prob_class0 >= class0_threshold and prob_class0 > prob_class1 and (prob_class0 - prob_class1) > prob_diff_threshold:
            classification = 0
        # 上記以外はデフォルトのクラス2(classification=2)のまま

        return classification

    except lgbm.basic.LightGBMError as lgbm_err:
        # LightGBMの予測エラー
        print(f"予測確率の計算中にLightGBMエラーが発生しました: {lgbm_err}")
        return -1
    except (ValueError, TypeError, IndexError) as e:
        # 確率へのアクセスや比較時のエラー
        print(f"クラス決定ロジックでエラーが発生しました: {e}")
        return -1

# --- メイン実行関数 ---
# 関数名を変更: main_strategy -> main_classification_example
def main_classification_example():
    """分類実行のメイン処理 (ダミーデータ使用例)"""
    project_root = get_project_root()
    # --- ディレクトリパス設定 ---
    model_dir = os.path.join(project_root, 'feature_analysis', 'models')
    output_dir = os.path.join(project_root, 'feature_analysis', 'outputs')

    model_path = os.path.join(model_dir, 'lgbm_model.txt')

    # --- 最新のしきい値ファイルの選択 ---
    thresholds_path: Optional[str] = None # 初期化
    try:
        all_files = os.listdir(output_dir)
        # 検索するファイル名を変更: optimal_thresholds_ -> optimal_class_thresholds_
        threshold_files = [f for f in all_files if f.startswith('optimal_class_thresholds_') and f.endswith('.json')]

        if not threshold_files:
            # メッセージを修正
            print(f"エラー: 最適化された分類しきい値ファイルが見つかりません ({output_dir})")
            return # 処理中断

        latest_threshold_file = sorted(threshold_files, reverse=True)[0]
        thresholds_path = os.path.join(output_dir, latest_threshold_file)
        # メッセージを修正
        print(f"使用する分類しきい値ファイル: {thresholds_path}")

    except FileNotFoundError:
        print(f"エラー: 出力ディレクトリが見つかりません: {output_dir}")
        return
    except OSError as e:
        print(f"出力ディレクトリへのアクセス中にエラーが発生しました ({output_dir}): {e}")
        return

    # --- モデルとしきい値のロード ---
    model_booster = load_model(model_path)
    # しきい値ロード関数名を変更
    thresholds = load_class_thresholds(thresholds_path) if thresholds_path else None

    if model_booster is None or thresholds is None:
        # メッセージを修正
        print("モデルまたは分類しきい値のロードに失敗したため、処理を中断します。")
        return

    # --- ダミーデータでの分類実行例 ---
    # メッセージを修正
    print("\n--- ダミーデータでの分類テスト ---")
    try:
        feature_names = model_booster.feature_name()
        dummy_data = pd.DataFrame(np.random.rand(1, len(feature_names)), columns=feature_names)
        print("生成したダミーデータ (最初の5列):")
        print(dummy_data.iloc[:, :5])

        # 分類関数を呼び出し (関数名を変更)
        classification_result = get_classification(model_booster, dummy_data, thresholds)

        # 結果を表示 (メッセージを修正)
        print(f"\nダミーデータに基づく分類結果: Class {classification_result}")
        if classification_result == -1:
            print("(エラーが発生しました)")

    except lgbm.basic.LightGBMError as lgbm_err:
         print(f"ダミーデータでのテスト中にLightGBMエラーが発生しました: {lgbm_err}")
    except (AttributeError, ValueError, TypeError) as e:
         print(f"ダミーデータでのテスト中にエラーが発生しました: {e}")

    print("\n------------------------------------")
    # メッセージを修正
    print("分類スクリプトの実行例 (実際のデータ処理は未実装)")

# --- スクリプト実行のエントリポイント ---
if __name__ == "__main__":
    # メイン関数名を変更
    main_classification_example()