import os
import pandas as pd
import numpy as np
import lightgbm as lgbm
import optuna
import json
import datetime
from typing import Dict, Tuple, Optional, List

# --- 可視化と評価のためのライブラリ ---
import matplotlib.pyplot as plt
import seaborn as sns
# f1_score に加えて fbeta_score をインポート
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, fbeta_score

# --- グローバル設定 ---
SAVE_FIGURES = True  # 図をファイルに保存するかどうか
SHOW_FIGURES = False # 図を画面に表示するかどうか
IMAGES_DIR = 'images' # 画像保存用サブディレクトリ名
# --- 追加: 最適化メトリック設定 ---
OPTIMIZATION_METRIC = 'fbeta' # 'f1' または 'fbeta' を選択
BETA_VALUE = 2.0             # 'fbeta' を選択した場合の beta 値 (OPTIMIZATION_METRIC='f1' の場合は無視される)
# ---------------------------------

# --- Google Colab/ローカルパス判定 ---
def is_running_in_colab() -> bool:
    """
    現在の実行環境がGoogle Colaboratoryであるかを判定します。

    Returns:
        bool: Google Colab環境であればTrue、そうでなければFalse。
    """
    try:
        # google.colab モジュールのインポートを試みる
        import google.colab
        return True
    except ImportError:
        # インポート失敗時はColab環境ではない
        return False

def get_project_root() -> str:
    """
    実行環境に応じてプロジェクトのルートディレクトリパスを取得します。

    Returns:
        str: プロジェクトルートの絶対パス。
             (Colab: '/content/drive/MyDrive/NFNet_Classifier_pretrained',
              Local: 'J:/マイドライブ/NFNet_Classifier_pretrained')
    """
    if is_running_in_colab():
        # Google Colab 環境のパス
        return '/content/drive/MyDrive/NFNet_Classifier_pretrained'
    else:
        # ローカル環境 (Windows) のパス
        return 'J:/マイドライブ/NFNet_Classifier_pretrained'

# --- 図の保存関数 ---
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

# --- しきい値適用ロジック (ヘルパー関数) ---
# 関数名を変更: calculate_actions_with_thresholds -> classify_with_thresholds
def classify_with_thresholds(y_pred_proba: np.ndarray, thresholds: Dict[str, float]) -> Optional[np.ndarray]:
    """
    予測確率と指定されたしきい値に基づいてクラス (0: Class 0, 1: Class 1, 2: Class 2) を計算します。

    Args:
        y_pred_proba (np.ndarray): モデルによる予測確率 (Nサンプル x 3クラス)。
        thresholds (Dict[str, float]): 最適化されたしきい値 ('class1_threshold'など)。

    Returns:
        Optional[np.ndarray]: 計算されたクラスラベルの配列。入力不正や計算エラー時はNone。
    """
    # 入力データの基本的なチェック
    if y_pred_proba is None or thresholds is None:
        # メッセージを修正
        print("警告: 予測確率またはしきい値が不足しているため、クラス分類を実行できません。")
        return None
    # 予測確率が2次元配列で、クラス数が3であることを確認
    if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 3:
        # メッセージを修正
        print(f"警告: 予測確率の形式が不正です (shape: {y_pred_proba.shape})。クラス分類を実行できません。")
        return None

    # 変数名を変更: actions -> classifications
    classifications: List[int] = [] # 各サンプルの分類結果を格納するリスト
    # しきい値のキー名を変更: buy -> class1, sell -> class0, hold -> class2
    class1_threshold = thresholds.get("class1_threshold", 0.7)
    class0_threshold = thresholds.get("class0_threshold", 0.7)
    class2_threshold = thresholds.get("class2_threshold", 0.6)
    # キー名を変更: prob_diff -> prob_diff
    prob_diff_threshold = thresholds.get("prob_diff_threshold", 0.2) # Class 1/Class 0 判断時の確率差の閾値

    try:
        # 各サンプル (各行の確率) に対してクラスを決定
        for probs in y_pred_proba:
            # 変数名を変更: prob_sell -> prob_class0, prob_buy -> prob_class1, prob_hold -> prob_class2
            prob_class0, prob_class1, prob_class2 = probs[0], probs[1], probs[2]
            classification = 2 # デフォルト分類はクラス2

            # クラス2条件 (最も優先度が高い)
            if prob_class2 >= class2_threshold and prob_class2 > prob_class1 and prob_class2 > prob_class0:
                classification = 2
            # クラス1条件
            elif prob_class1 >= class1_threshold and prob_class1 > prob_class0 and (prob_class1 - prob_class0) > prob_diff_threshold:
                classification = 1
            # クラス0条件
            elif prob_class0 >= class0_threshold and prob_class0 > prob_class1 and (prob_class0 - prob_class1) > prob_diff_threshold:
                classification = 0
            # 上記のいずれにも当てはまらない場合は、デフォルトのクラス2(classification=2)のまま

            classifications.append(classification) # 決定した分類結果をリストに追加
        # 変数名を変更: actions -> classifications
        return np.array(classifications) # 分類結果リストをNumpy配列に変換して返す
    except (TypeError, IndexError) as e: # 確率へのアクセスや比較時のエラー
        # メッセージを修正
        print(f"クラス分類計算中にエラーが発生しました: {e}")
        return None

# --- 取引シミュレーション関数 ---
# この関数はコメントアウトされているため変更不要
# def simulate_trading_performance(...): ...

# --- Optuna目的関数 (スコア最大化) ---
# 関数名を変更: objective_thresholds -> objective_class_thresholds
# 引数に metric_type と beta を追加
def objective_class_thresholds(trial: optuna.Trial, model_booster: lgbm.Booster, X_valid: pd.DataFrame, y_valid: pd.Series, metric_type: str = 'fbeta', beta: float = 2.0) -> float:
    """
    Optunaのトライアルごとに呼び出される目的関数。
    提案されたしきい値でクラス分類を実行し、指定されたメトリック (F1またはF-beta) の
    Class 1とClass 0のマクロ平均スコアを返します。

    Args:
        trial (optuna.Trial): Optunaのトライアルオブジェクト。しきい値を提案する。
        model_booster (lgbm.Booster): 学習済みモデル。
        X_valid (pd.DataFrame): 検証用特徴量データ。
        y_valid (pd.Series): 検証用真ラベルデータ。
        metric_type (str, optional): 使用する評価指標 ('f1' or 'fbeta')。デフォルトは 'fbeta'。
        beta (float, optional): F-betaスコアのbeta値 ('fbeta'選択時)。デフォルトは 2.0。

    Returns:
        float: このトライアルでのClass 1/Class 0のマクロ平均スコア。
               エラー発生時は低い値 (-1.0) を返す。
    """
    # 探索するしきい値のキー名を変更
    class1_threshold = trial.suggest_float("class1_threshold", 0.5, 0.95, step=0.01)
    class0_threshold = trial.suggest_float("class0_threshold", 0.5, 0.95, step=0.01)
    class2_threshold = trial.suggest_float("class2_threshold", 0.4, 0.8, step=0.01)
    prob_diff_threshold = trial.suggest_float("prob_diff_threshold", 0.0, 0.3, step=0.01)

    # 提案されたしきい値を辞書にまとめる (キー名を変更)
    thresholds = {
        "class1_threshold": class1_threshold,
        "class0_threshold": class0_threshold,
        "class2_threshold": class2_threshold,
        "prob_diff_threshold": prob_diff_threshold
    }

    y_pred_proba_valid: Optional[np.ndarray] = None
    try:
        # 検証データに対する予測確率を計算
        y_pred_proba_valid = model_booster.predict(X_valid)
    except lgbm.basic.LightGBMError as lgbm_err:
        # 予測エラーが発生した場合、このトライアルは失敗とする
        print(f"トライアル {trial.number}: 予測確率の計算中にエラー: {lgbm_err}")
        return -1.0 # スコアは0以上なので、エラー時は負の値を返す

    # しきい値に基づいてクラス分類を実行 (関数名を変更)
    classifications = classify_with_thresholds(y_pred_proba_valid, thresholds)

    # クラス分類に失敗した場合は、低い値を返す
    if classifications is None:
        # メッセージを修正
        print(f"トライアル {trial.number}: クラス分類計算に失敗しました。")
        return -1.0

    try:
        # --- 変更箇所: metric_type に基づいてスコアを計算 ---
        if metric_type == 'f1':
            # Class 1とClass 0のマクロ平均F1スコアを計算
            score = f1_score(y_valid, classifications, labels=[0, 1], average='macro', zero_division=0.0)
        elif metric_type == 'fbeta':
            # Class 1とClass 0のマクロ平均F-betaスコアを計算
            score = fbeta_score(y_valid, classifications, beta=beta, labels=[0, 1], average='macro', zero_division=0.0)
        else:
            # 未知のメトリックタイプが指定された場合
            print(f"トライアル {trial.number}: 未知のメトリックタイプ '{metric_type}' が指定されました。")
            return -1.0
        # -----------------------------------------------------
        return score
    except ValueError as ve:
        # y_valid や classifications の形式が不正な場合など
        print(f"トライアル {trial.number}: スコア計算中にエラー: {ve}")
        return -1.0

# --- しきい値最適化実行関数 ---
# 関数名を変更: optimize_trading_thresholds -> optimize_classification_thresholds
# 引数に metric_type と beta を追加
def optimize_classification_thresholds(model_booster: lgbm.Booster, X_valid: pd.DataFrame, y_valid: pd.Series, n_trials: int = 100, metric_type: str = 'fbeta', beta: float = 2.0) -> Tuple[Optional[Dict[str, float]], float]:
    """
    Optunaを使用して分類パフォーマンス (指定されたメトリック) を最大化するしきい値を探索します。

    Args:
        model_booster (lgbm.Booster): 学習済みモデル。
        X_valid (pd.DataFrame): 検証用特徴量データ。
        y_valid (pd.Series): 検証用真ラベルデータ。
        n_trials (int, optional): Optunaの試行回数。デフォルトは100。
        metric_type (str, optional): 使用する評価指標 ('f1' or 'fbeta')。デフォルトは 'fbeta'。
        beta (float, optional): F-betaスコアのbeta値 ('fbeta'選択時)。デフォルトは 2.0。

    Returns:
        Tuple[Optional[Dict[str, float]], float]:
            - 最適化されたしきい値の辞書。最適化に失敗した場合はNone。
            - 最良のトライアルでのパフォーマンススコア。
    """
    # --- 変更箇所: 目標指標名を動的に設定 ---
    metric_name = f"F-beta(beta={beta})" if metric_type == 'fbeta' else "F1"
    print(f"\n--- しきい値最適化開始 (目標: Class 1/Class 0 マクロ平均 {metric_name} 最大化, 試行回数: {n_trials}) ---")
    # -----------------------------------------
    try:
        # Optuna Studyオブジェクトを作成 (目的: 最大化)
        study = optuna.create_study(direction="maximize") # スコアを最大化
        # --- 変更箇所: objective関数に metric_type と beta を渡す ---
        study.optimize(lambda trial: objective_class_thresholds(trial, model_booster, X_valid, y_valid, metric_type=metric_type, beta=beta),
                       n_trials=n_trials,
                       show_progress_bar=True) # 進捗バーを表示
        # ---------------------------------------------------------

        # 最良の結果を取得
        best_thresholds = study.best_params
        best_performance = study.best_value # ここには最良のスコアが入る

        print("--- しきい値最適化完了 ---")
        # --- 変更箇所: パフォーマンス指標名を動的に表示 ---
        print(f"最良パフォーマンス (Class 1/Class 0 マクロ平均 {metric_name} スコア): {best_performance:.4f}")
        # ---------------------------------------------
        print("最適化されたしきい値:")
        for key, value in best_thresholds.items():
            print(f"  {key}: {value:.4f}")

        return best_thresholds, best_performance

    except optuna.exceptions.OptunaError as optuna_err:
        # Optunaの最適化プロセス中のエラー
        print(f"Optuna最適化中にエラーが発生しました: {optuna_err}")
        return None, -1.0 # エラー時は低いスコアを返す
    except Exception as e: # 予期せぬエラー (より具体的に捕捉することが望ましい)
        print(f"しきい値最適化中に予期せぬエラーが発生しました: {e}")
        return None, -1.0 # エラー時は低いスコアを返す

# --- 最適しきい値保存関数 ---
# 関数名を変更: save_optimal_thresholds -> save_optimal_class_thresholds
# 引数に metric_type と beta を追加
def save_optimal_class_thresholds(thresholds: Dict[str, float], performance: float, output_dir: str, metric_type: str, beta: Optional[float] = None):
    """
    最適化されたしきい値、パフォーマンススコア、および使用したメトリック情報をJSONファイルに保存します。

    Args:
        thresholds (Dict[str, float]): 最適化されたしきい値の辞書。
        performance (float): 最適化時のパフォーマンススコア。
        output_dir (str): 保存先のディレクトリ (例: 'feature_analysis/outputs')。
        metric_type (str): 最適化に使用したメトリック ('f1' or 'fbeta')。
        beta (Optional[float], optional): F-betaスコアのbeta値 ('fbeta'選択時)。デフォルトはNone。
    """
    # --- 変更箇所: 保存するデータにメトリック情報を追加 ---
    data_to_save = {
        "optimization_timestamp": datetime.datetime.now().isoformat(),
        "metric_type": metric_type,
        "best_performance": performance,
        "thresholds": thresholds
    }
    # F-betaの場合のみbeta値を追加
    if metric_type == 'fbeta' and beta is not None:
        data_to_save["beta_value"] = beta
    # -----------------------------------------------------

    # タイムスタンプ付きのファイル名を生成 (ファイル名を変更)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"optimal_class_thresholds_{timestamp}.json" # optimal_thresholds -> optimal_class_thresholds
    filepath = os.path.join(output_dir, filename)

    try:
        # JSONファイルに書き込みモードで保存 (indentで整形)
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"最適化されたしきい値を保存しました: {filepath}")
    except IOError as e:
        # ファイル書き込み権限がない、ディスク容量不足など
        print(f"しきい値ファイルの書き込み中にエラーが発生しました ({filepath}): {e}")
    except TypeError as e:
        # JSONシリアライズできないデータ型が含まれている場合
        print(f"しきい値データのJSONシリアライズ中にエラーが発生しました: {e}")

# --- 可視化関数 ---
# 関数名を変更: visualize_optimization_results -> visualize_threshold_evaluation
def visualize_threshold_evaluation(y_true: pd.Series,
                                   y_pred_proba: np.ndarray, # 確率分布表示のため必要
                                   classifications: np.ndarray, # しきい値適用後の分類結果
                                   metric_type: str = 'fbeta', # 最適化に使用したメトリック
                                   beta: Optional[float] = None # beta値 (fbetaの場合)
                                   ) -> Optional[plt.Figure]:
    """
    予測確率分布と、しきい値適用後の分類結果に基づく混同行列を含むFigureオブジェクトを作成します。
    タイトルに最適化に使用したメトリック情報を追加します。

    Args:
        y_true (pd.Series): 検証データの真のラベル (Series)。
        y_pred_proba (np.ndarray): モデルによる予測確率 (Numpy Array, Nサンプル x 3クラス)。
        classifications (np.ndarray): しきい値適用後の予測クラスラベル (Numpy Array)。
        metric_type (str, optional): 最適化に使用したメトリック。デフォルトは 'fbeta'。
        beta (Optional[float], optional): F-betaスコアのbeta値 ('fbeta'選択時)。デフォルトはNone。

    Returns:
        Optional[plt.Figure]: 作成されたMatplotlib Figureオブジェクト。入力不正時はNone。
    """
    # 入力データの基本的なチェック (変数名を変更: actions -> classifications)
    if y_pred_proba is None or classifications is None:
        print("警告: 予測確率または分類結果データが不足しているため、可視化をスキップします。")
        return None
    # 真ラベルと分類結果のサンプル数が一致するか確認 (変数名を変更: actions -> classifications)
    if y_true.shape[0] != classifications.shape[0]:
         print(f"警告: 真ラベルと予測クラスラベルのサンプル数が異なります (True: {y_true.shape[0]}, Predicted: {classifications.shape[0]})。可視化をスキップします。")
         return None

    # スタイル設定
    plt.style.use('seaborn-v0_8-whitegrid')
    # FigureとAxesを作成 (1行2列、横長のレイアウト)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- 1. 予測確率分布のプロット (左側: axes[0]) ---
    try:
        # 予測確率の形式チェック (2次元配列かつ3クラス)
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 3:
            # DataFrameに変換してseabornでプロット (カラム名を変更)
            df_proba = pd.DataFrame(y_pred_proba, columns=['Class 0 Prob', 'Class 1 Prob', 'Class 2 Prob'])
            sns.kdeplot(data=df_proba, ax=axes[0])
            axes[0].set_title('Prediction Probability Distribution (Validation Set)')
            axes[0].set_xlabel('Probability')
            axes[0].set_ylabel('Density')
            axes[0].legend(title='Class Probability') # 凡例タイトルを変更
        else:
            print("警告: 予測確率の形式が不正なため、分布プロットをスキップします。")
    except (ValueError, TypeError, np.linalg.LinAlgError) as e: # kdeplot で発生しうるエラー
         print(f"確率分布のプロット中にエラーが発生しました: {e}")
         # エラーが発生しても、混同行列のプロットは試みる

    # --- 2. 混同行列のプロット (右側: axes[1]) ---
    y_true_np = y_true.values # Series から numpy 配列へ変換
    labels = [0, 1, 2] # クラスラベル (Class 0, Class 1, Class 2)
    # 表示用ラベルを変更
    display_labels = ['Class 0', 'Class 1', 'Class 2']

    try:
        # 混同行列を計算 (変数名を変更: actions -> classifications)
        cm = confusion_matrix(y_true_np, classifications, labels=labels)
        # --- 変更箇所: 分類レポートのタイトルにメトリック情報を追加 ---
        metric_info = f"(Optimized for {metric_type.upper()}"
        if metric_type == 'fbeta' and beta is not None:
            metric_info += f" with beta={beta})"
        else:
            metric_info += ")"
        print(f"\n--- Classification Report {metric_info} on Validation Set ---")
        # -----------------------------------------------------------------
        # 分類レポートを計算して表示 (変数名とラベル名を変更)
        report = classification_report(y_true_np, classifications, labels=labels, target_names=display_labels, zero_division=0)
        print(report)
        print("--------------------------------------------------------------------------")

        # 混同行列を ConfusionMatrixDisplay を使ってプロット (ラベル名を変更)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=axes[1], cmap='Blues', values_format='d') # 右側のAxesに描画
        # --- 変更箇所: 混同行列のタイトルにメトリック情報を追加 ---
        axes[1].set_title(f'Confusion Matrix {metric_info}')
        # ----------------------------------------------------

    except ValueError as ve:
        # confusion_matrix や classification_report で発生する可能性のあるエラー
        # (例: y_true と classifications の要素の型や範囲が不正)
        print(f"混同行列または分類レポートの計算・表示中にエラーが発生しました: {ve}")
        # エラーが発生してもFigureオブジェクトは返す（片方のプロットはできている可能性があるため）

    # --- 図全体の調整 ---
    plt.tight_layout() # サブプロット間の重なりを防ぐようにレイアウトを自動調整

    # 作成したFigureオブジェクトを返す
    return fig


# --- メイン関数 ---
def main_optimizer():
    """しきい値最適化のメイン処理"""
    # Matplotlibのインタラクティブモード設定 (バックエンドでの描画を制御)
    if SHOW_FIGURES:
        plt.ion() # 表示する場合: インタラクティブモードON
    else:
        plt.ioff() # 表示しない場合: インタラクティブモードOFF

    # プロジェクトルートと関連ディレクトリのパスを取得
    project_root = get_project_root()
    model_dir = os.path.join(project_root, 'feature_analysis', 'models')
    output_dir = os.path.join(project_root, 'feature_analysis', 'outputs') # output_dir を定義

    # モデルファイルと検証データファイルのパスを構築
    model_path = os.path.join(model_dir, 'lgbm_model.txt')
    valid_data_path_X = os.path.join(output_dir, 'validation_data_X.csv')
    valid_data_path_y = os.path.join(output_dir, 'validation_data_y.csv')

    # --- モデルのロード ---
    model_booster: Optional[lgbm.Booster] = None # 型ヒント
    try:
        # --- 変更箇所: ファイルパスではなく、ファイル内容を読み込んで渡す ---
        # ファイルをUTF-8エンコーディングで読み込む
        with open(model_path, 'r', encoding='utf-8') as f:
            model_string = f.read()
        # 文字列からモデルをロード
        model_booster = lgbm.Booster(model_str=model_string)
        # ----------------------------------------------------------
        print(f"モデルをロードしました: {model_path}")
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        return # モデルがないと続行不可
    except lgbm.basic.LightGBMError as lgbm_err:
        # モデル文字列のパースエラーなどもここで捕捉される可能性あり
        print(f"モデルのロード中にLightGBMエラーが発生しました: {lgbm_err}")
        return # モデルロード失敗時は続行不可
    except IOError as e:
        # ファイル読み込み自体のエラー (権限など)
        print(f"モデルファイルの読み込み中にIOエラーが発生しました ({model_path}): {e}")
        return # ファイルアクセスエラー時も続行不可
    except UnicodeDecodeError as ude:
        # UTF-8での読み込みに失敗した場合 (ファイルがUTF-8でない可能性)
        print(f"モデルファイルのデコード中にエラーが発生しました ({model_path}): {ude}")
        return

    # --- 検証データのロード ---
    X_valid: Optional[pd.DataFrame] = None # 型ヒント
    y_valid: Optional[pd.Series] = None # 型ヒント
    try:
        # 特徴量データ(X)をCSVから読み込み (インデックス列を指定)
        X_valid = pd.read_csv(valid_data_path_X, index_col=0)
        # 正解ラベル(y)をCSVから読み込み (インデックス列とヘッダーを指定し、Seriesに変換)
        y_valid = pd.read_csv(valid_data_path_y, index_col=0, header=0).squeeze("columns")
        # y_validがDataFrameのままの場合 (squeezeが効かないケース)、Seriesに変換
        if isinstance(y_valid, pd.DataFrame) and y_valid.shape[1] == 1:
             y_valid = y_valid.iloc[:, 0]
        # 最終的にSeriesでなければエラー
        elif not isinstance(y_valid, pd.Series):
             raise TypeError("y_valid がSeries形式ではありません。")
        print(f"検証データをロードしました: {valid_data_path_X}, {valid_data_path_y}")
    except FileNotFoundError:
        print(f"エラー: 検証データファイルが見つかりません: {valid_data_path_X} または {valid_data_path_y}")
        return # データがないと続行不可
    except pd.errors.EmptyDataError:
        print(f"エラー: 検証データファイルが空です: {valid_data_path_X} または {valid_data_path_y}")
        return # データが空では続行不可
    except (ValueError, TypeError, KeyError, IndexError, pd.errors.ParserError) as data_err:
        # データ読み込み/処理中の様々なエラーを捕捉
        print(f"検証データの読み込みまたは処理中にエラーが発生しました: {data_err}")
        return # データ処理エラー時も続行不可

    # --- 予測確率の取得 ---
    y_pred_proba_valid: Optional[np.ndarray] = None # 型ヒント
    # モデルと検証データが正常にロードされた場合のみ実行
    if model_booster is not None and X_valid is not None:
        try:
            # モデルを使って検証データに対する予測確率を計算
            y_pred_proba_valid = model_booster.predict(X_valid)
            print("検証データに対する予測確率を取得しました。")
        except lgbm.basic.LightGBMError as lgbm_err:
            # LightGBMの予測エラー
            print(f"予測確率の計算中にLightGBMエラーが発生しました: {lgbm_err}")
            return # 予測できない場合は続行不可
    else:
        # モデルまたはデータがロードされていない場合
        print("警告: モデルまたは検証データがロードされていないため、予測確率を計算できません。")
        return # 予測できない場合は続行不可

    # --- しきい値の最適化を実行 ---
    optimal_thresholds: Optional[Dict[str, float]] = None # 型ヒント
    best_performance: float = -1e9 # 初期値 (低い値)
    # モデル、検証データ(X, y)がすべて揃っている場合のみ実行
    if model_booster is not None and X_valid is not None and y_valid is not None:
        n_trials = 50 # Optunaの試行回数
        # --- 変更箇所: グローバル設定からメトリックタイプとbeta値を取得して渡す ---
        optimal_thresholds, best_performance = optimize_classification_thresholds(
            model_booster, X_valid, y_valid, n_trials=n_trials,
            metric_type=OPTIMIZATION_METRIC, beta=BETA_VALUE
        )
        # ---------------------------------------------------------------------
    else:
         print("警告: モデルまたは検証データが不足しているため、しきい値最適化をスキップします。")

    # --- しきい値適用後の分類結果を計算 ---
    # 変数名を変更: actions_optimized -> classifications_optimized
    classifications_optimized: Optional[np.ndarray] = None # 型ヒント
    # 予測確率と最適化しきい値が得られた場合のみ実行
    if y_pred_proba_valid is not None and optimal_thresholds is not None:
        # 分類関数を呼び出し (関数名を変更)
        classifications_optimized = classify_with_thresholds(y_pred_proba_valid, optimal_thresholds)
        if classifications_optimized is None:
            # 分類計算に失敗した場合
            print("警告: しきい値適用後の分類結果計算に失敗しました。")
    else:
        # 予測確率またはしきい値がない場合
        print("警告: 予測確率または最適しきい値が不足しているため、分類結果計算をスキップします。")


    # --- 最適化結果の可視化 ---
    # 可視化に必要なデータ (真ラベル, 予測確率, 計算済み分類結果) がすべて揃っているか確認
    # 変数名を変更: actions_optimized -> classifications_optimized
    if y_valid is not None and y_pred_proba_valid is not None and classifications_optimized is not None:
        # --- 変更箇所: 可視化関数にメトリック情報を渡す ---
        fig_results = visualize_threshold_evaluation(
            y_valid, y_pred_proba_valid, classifications_optimized,
            metric_type=OPTIMIZATION_METRIC, beta=BETA_VALUE
        )
        # -------------------------------------------------

        # Figureオブジェクトが正常に作成された場合のみ処理
        if fig_results:
            # 図を保存 (save_figure関数を使用、output_dir を渡す, ファイル名を変更)
            save_figure(fig_results, 'threshold_optimization_summary', output_dir)
            # 図を表示 (SHOW_FIGURESフラグがTrueの場合)
            if SHOW_FIGURES:
                plt.show()
            # else: plt.close(fig_results) # save_figure内で閉じるので不要
    else:
        # 必要なデータが不足している場合
        print("警告: 可視化に必要なデータが不足しているため、結果の可視化をスキップします。")

    # --- 最適化されたしきい値を保存 ---
    # 最適なしきい値が得られた場合のみ実行
    if optimal_thresholds:
        # --- 変更箇所: しきい値保存関数にメトリック情報を渡す ---
        save_optimal_class_thresholds(
            optimal_thresholds, best_performance, output_dir,
            metric_type=OPTIMIZATION_METRIC, beta=BETA_VALUE
        )
        # ----------------------------------------------------
    else:
        # 最適化に失敗したか、実行されなかった場合
        print("警告: 最適なしきい値が得られなかったため、ファイルへの保存をスキップします。")

    print("\nしきい値最適化処理が完了しました。")

# --- スクリプト実行のエントリポイント ---
if __name__ == "__main__":
    # このスクリプトが直接実行された場合に main_optimizer 関数を呼び出す
    main_optimizer()