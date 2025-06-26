import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns # seaborn をインポート
import re # 正規表現モジュールをインポート
import traceback # traceback をインポート

# プロジェクトルートをモジュールパスに追加
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

# --- 修正: 必要なモジュールをインポート ---
from model import StockClassifier
from multimodal_model import MultimodalStockClassifier # マルチモーダルモデルもインポート
from datamodule import StockDataModule
from configs.config_utils import load_config, get_project_root # get_project_root もインポート

# --- 追加: find_best_checkpoint 関数の定義 (evaluate.py と同じ) ---
# utils.py に同等の関数がある場合はそちらを使用し、この定義は削除
def find_best_checkpoint(config, metric="f1"):
    """
    指定されたメトリックに基づいて最適なチェックポイントファイルを見つける。
    新しいディレクトリ構造とファイル名形式に対応。

    Args:
        config (dict): 設定辞書。'model_mode', 'model_architecture_name' を含む。
        metric (str): 最適化するメトリック ('f1' または 'loss')。

    Returns:
        str or None: 最適なチェックポイントファイルのパス。見つからない場合はNone。
    """
    project_root = get_project_root() # プロジェクトルートを取得
    model_mode = config.get("model_mode", "single")
    model_architecture_name = config.get("model_architecture_name", "default_model")
    # 新しいチェックポイントディレクトリパスを構築
    checkpoint_dir = os.path.join(project_root, "checkpoints", model_mode, model_architecture_name)

    print(f"チェックポイントディレクトリを検索中: {checkpoint_dir}")

    if not os.path.isdir(checkpoint_dir):
        print(f"エラー: チェックポイントディレクトリが見つかりません: {checkpoint_dir}")
        return None

    # 新しいファイル名形式 'epoch={epoch:05d}-val_loss={val_loss:.4f}-val_f1={val_f1:.4f}.ckpt'
    # metric に応じた正規表現パターン
    if metric == "f1":
        # val_f1 を抽出するパターン
        pattern = re.compile(r'epoch=(\d+)-val_loss=([\d.]+)-val_f1=([\d.]+)\.ckpt')
        metric_index = 2 # F1スコアは3番目のキャプチャグループ (0-based index)
        best_metric_val = -1.0 # F1スコアは高いほど良い
        compare_func = lambda current, best: current > best
    elif metric == "loss":
        # val_loss を抽出するパターン
        pattern = re.compile(r'epoch=(\d+)-val_loss=([\d.]+)-val_f1=([\d.]+)\.ckpt')
        metric_index = 1 # 損失は2番目のキャプチャグループ
        best_metric_val = float('inf') # 損失は低いほど良い
        compare_func = lambda current, best: current < best
    else:
        print(f"エラー: 未知のメトリック '{metric}'。'f1' または 'loss' を使用してください。")
        return None

    best_checkpoint_path = None
    found_checkpoints = []
    last_ckpt_path = None # last.ckpt のパスを初期化

    try:
        for filename in os.listdir(checkpoint_dir):
            match = pattern.match(filename)
            if match:
                found_checkpoints.append(filename)
                try:
                    # 指定されたメトリックの値を抽出
                    current_metric_val = float(match.group(metric_index + 1)) # グループインデックスは1から始まるため+1
                    print(f"  チェックポイント '{filename}' の {metric}: {current_metric_val}")
                    # 最良のメトリック値を更新
                    if compare_func(current_metric_val, best_metric_val):
                        best_metric_val = current_metric_val
                        best_checkpoint_path = os.path.join(checkpoint_dir, filename)
                except (ValueError, IndexError) as e:
                    print(f"  警告: ファイル '{filename}' のメトリック値の解析中にエラー: {e}")
                    continue
            # 'last.ckpt' も候補として保持 (最良が見つからない場合に使用)
            elif filename == "last.ckpt":
                 last_ckpt_path = os.path.join(checkpoint_dir, filename)

    except OSError as e:
        print(f"エラー: チェックポイントディレクトリの読み取り中にエラーが発生しました: {e}")
        return None

    if best_checkpoint_path:
        print(f"最適な {metric} ({best_metric_val:.4f}) を持つチェックポイントが見つかりました: {best_checkpoint_path}")
        return best_checkpoint_path
    elif last_ckpt_path and os.path.exists(last_ckpt_path): # last_ckpt_path が None でないことを確認
         print(f"警告: 最適な {metric} を持つチェックポイントが見つかりませんでした。'last.ckpt' を使用します: {last_ckpt_path}")
         return last_ckpt_path
    else:
        print(f"警告: 有効なチェックポイントファイルがディレクトリ '{checkpoint_dir}' に見つかりませんでした。")
        return None
# --- find_best_checkpoint 関数の定義ここまで ---


def plot_confusion_matrix(config, checkpoint_path=None, metric="f1", save_path=None):
    """
    混同行列を可視化する

    Args:
        config: 設定辞書
        checkpoint_path: 使用するモデルチェックポイントのパス（Noneの場合は自動検出）
        metric: チェックポイントの選択基準（'f1'または'loss'）
        save_path: 可視化結果の保存パス（Noneの場合は表示のみ）

    Returns:
        tuple: (混同行列 (numpy.ndarray), 分類レポート (str)) または (None, None)
    """
    # チェックポイントが指定されていない場合は自動検出
    if checkpoint_path is None:
        # --- 修正: find_best_checkpoint を呼び出す ---
        checkpoint_path = find_best_checkpoint(config, metric)
        if checkpoint_path is None:
            print("有効なチェックポイントが見つかりません。可視化を中止します。")
            return None, None # cm, report を返せないので None を返す

    print(f"混同行列の計算に使用するチェックポイント: {checkpoint_path}")

    # データモジュールとモデルの準備
    try:
        data_module = StockDataModule(config)
        data_module.setup("test") # テストデータを準備
    except Exception as e:
        print(f"エラー: DataModuleの初期化またはセットアップ中にエラーが発生しました: {e}")
        return None, None

    # --- 修正: モデルのロード (model_mode に基づく) ---
    model_mode = config.get("model_mode", "single")
    try:
        if model_mode == "multi":
            print("マルチモーダルモデルをロードします...")
            model = MultimodalStockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        else:
            print("シングルモーダルモデルをロードします...")
            model = StockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        print(f"モデルを正常にロードしました: {type(model).__name__}")
    except FileNotFoundError:
        print(f"エラー: チェックポイントファイルが見つかりません: {checkpoint_path}")
        return None, None
    except Exception as e:
        print(f"エラー: モデルのロード中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return None, None

    # デバイスの決定
    # --- 修正: config から force_cpu を考慮 ---
    use_gpu = torch.cuda.is_available() and not config.get('force_cpu', False)
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"使用デバイス: {device}")

    # モデルを適切なデバイスに移動
    try:
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"エラー: モデルのデバイス転送または評価モード設定中にエラー: {e}")
        return None, None

    # テストデータローダーから全サンプルの予測を取得
    all_preds = []
    all_labels = []
    try:
        test_loader = data_module.test_dataloader()
    except Exception as e:
        print(f"エラー: テストデータローダーの取得中にエラー: {e}")
        return None, None

    print("テストデータ全体で予測を収集中...")
    with torch.no_grad():
        try:
            for batch in test_loader:
                # --- 修正: マルチモーダル対応 ---
                # StockDataModule が返す形式に依存
                if model_mode == "multi":
                     # マルチモーダルの場合のバッチ形式に合わせて調整が必要
                     # 例: (images, numerical_features), labels = batch
                     print("警告: マルチモーダルモデルのバッチ処理は未実装です。画像のみと仮定します。")
                     # この例では画像のみと仮定して進めるが、実際には要修正
                     images, labels = batch # 仮
                     images = images.to(device)
                     # numerical_features = numerical_features.to(device) # 例
                     # logits = model(images, numerical_features) # 例
                     logits = model(images) # 仮
                else:
                    images, labels = batch
                    images = images.to(device)
                    # --- 修正: モデルのforward呼び出し (logitsのみを期待) ---
                    logits = model(images)
                    # --- 修正ここまで ---

                preds = torch.argmax(logits, dim=1)

                # CPU上に戻してからNumPy配列に変換
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy()) # labels は最初からCPUにある可能性もある
        except StopIteration:
             print("テストデータローダーが空です。")
             # データがない場合は None を返す
             return None, None
        except Exception as e:
            print(f"エラー: 予測の収集中にエラーが発生しました: {e}")
            traceback.print_exc()
            return None, None

    if not all_labels or not all_preds:
        print("予測結果が収集されませんでした。")
        return None, None

    # クラス名を取得（configから）
    class_names = config.get('class_names', None)
    if class_names is None:
        # クラス名がない場合、ラベルの最大値からクラス数を推定
        try:
            num_classes = int(max(all_labels) + 1)
            class_names = [f"Class {i}" for i in range(num_classes)]
            print(f"警告: configにclass_namesが見つかりません。推定されたクラス名を使用: {class_names}")
        except Exception: # all_labels が空の場合など
             class_names = ["Unknown"] * 3 # 仮のクラス名
             print(f"警告: configにclass_namesが見つからず、推定もできませんでした。仮のクラス名を使用: {class_names}")

    # 混同行列の計算
    try:
        cm = confusion_matrix(all_labels, all_preds)
    except ValueError as e:
        print(f"エラー: 混同行列の計算中にエラー: {e}")
        return None, None

    # 混同行列の可視化 (seabornを使用)
    plt.figure(figsize=(8, 6)) # サイズ調整
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
    except Exception as e:
        print(f"エラー: 混同行列の描画中にエラー: {e}")
        plt.close() # エラー発生時はフィギュアを閉じる
        return cm, None # 行列は計算できたかもしれないので返す

    plt.tight_layout()

    # 保存するか表示するか
    if save_path:
        try:
            # 保存先ディレクトリが存在しない場合は作成
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path)
            print(f"混同行列を保存しました: {save_path}")
        except Exception as e:
            print(f"エラー: 混同行列の保存中にエラー: {e}")
    else:
        plt.show()
    plt.close() # メモリ解放のためフィギュアを閉じる

    # 分類レポートの表示
    report = None
    try:
        # zero_division=0 を追加して警告抑制
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
        print("\n分類レポート:")
        print(report)
    except ValueError as e:
        print(f"エラー: 分類レポートの生成中にエラー: {e}")
        # レポート生成に失敗しても行列は返す
    except Exception as e:
        print(f"エラー: 分類レポートの生成中に予期せぬエラー: {e}")

    return cm, report

def visualize_misclassified(config, checkpoint_path=None, metric="f1", num_samples=8, save_path=None):
    """
    誤分類されたサンプルを可視化する

    Args:
        config: 設定辞書
        checkpoint_path: 使用するモデルチェックポイントのパス（Noneの場合は自動検出）
        metric: チェックポイントの選択基準（'f1'または'loss'）
        num_samples: 表示するサンプル数
        save_path: 可視化結果の保存パス（Noneの場合は表示のみ）

    Returns:
        list: 誤分類されたサンプルのリスト [(画像テンソル, 正解ラベル, 予測ラベル, 信頼度), ...] または None
    """
    # チェックポイントが指定されていない場合は自動検出
    if checkpoint_path is None:
        # --- 修正: find_best_checkpoint を呼び出す ---
        checkpoint_path = find_best_checkpoint(config, metric)
        if checkpoint_path is None:
            print("有効なチェックポイントが見つかりません。可視化を中止します。")
            return None # misclassified リストを返せないので None

    print(f"誤分類サンプルの可視化に使用するチェックポイント: {checkpoint_path}")

    # データモジュールとモデルの準備
    try:
        data_module = StockDataModule(config)
        data_module.setup("test") # テストデータを準備
    except Exception as e:
        print(f"エラー: DataModuleの初期化またはセットアップ中にエラーが発生しました: {e}")
        return None

    # --- 修正: モデルのロード (model_mode に基づく) ---
    model_mode = config.get("model_mode", "single")
    try:
        if model_mode == "multi":
            print("マルチモーダルモデルをロードします...")
            model = MultimodalStockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        else:
            print("シングルモーダルモデルをロードします...")
            model = StockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        print(f"モデルを正常にロードしました: {type(model).__name__}")
    except FileNotFoundError:
        print(f"エラー: チェックポイントファイルが見つかりません: {checkpoint_path}")
        return None
    except Exception as e:
        print(f"エラー: モデルのロード中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return None

    # デバイスの決定
    # --- 修正: config から force_cpu を考慮 ---
    use_gpu = torch.cuda.is_available() and not config.get('force_cpu', False)
    device = torch.device("cuda" if use_gpu else "cpu")
    print(f"使用デバイス: {device}")

    # モデルを適切なデバイスに移動
    try:
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"エラー: モデルのデバイス転送または評価モード設定中にエラー: {e}")
        return None

    # テストデータローダーから全サンプルを処理して誤分類サンプルを見つける
    misclassified = []
    try:
        test_loader = data_module.test_dataloader()
    except Exception as e:
        print(f"エラー: テストデータローダーの取得中にエラー: {e}")
        return None

    print("誤分類サンプルを検索中...")
    with torch.no_grad():
        try:
            for batch in test_loader:
                # --- 修正: マルチモーダル対応 (plot_confusion_matrix と同様の注意点) ---
                if model_mode == "multi":
                     print("警告: マルチモーダルモデルのバッチ処理は未実装です。画像のみと仮定します。")
                     images, labels = batch # 仮
                     images = images.to(device)
                     # numerical_features = numerical_features.to(device) # 例
                     # logits = model(images, numerical_features) # 例
                     logits = model(images) # 仮
                else:
                    images, labels = batch
                    images = images.to(device)
                    # --- 修正: モデルのforward呼び出し (logitsのみを期待) ---
                    logits = model(images)
                    # --- 修正ここまで ---

                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                # 誤分類サンプルを特定
                for i in range(len(images)):
                    # labels も device 上にある可能性があるので cpu() を呼ぶ
                    # デバイスを合わせて比較
                    true_label_on_device = labels[i].to(preds.device)
                    if preds[i] != true_label_on_device:
                        # (画像, 正解ラベル, 予測ラベル, 予測確率) を保存
                        misclassified.append((
                            images[i].cpu(),  # CPUに移動して保存
                            labels[i].cpu().item(), # CPUに移動してitem()
                            preds[i].cpu().item(),  # CPUに移動してitem()
                            probs[i][preds[i]].cpu().item() # CPUに移動してitem()
                        ))

                        # 指定数集まったら終了
                        if len(misclassified) >= num_samples:
                            break # 内側のループを抜ける

                if len(misclassified) >= num_samples:
                    break # 外側のループも抜ける
        except StopIteration:
             print("テストデータローダーが空です。")
             # データがない場合は None を返す
             return None
        except Exception as e:
            print(f"エラー: 誤分類サンプルの検索中にエラーが発生しました: {e}")
            traceback.print_exc()
            return None # エラー発生時は None を返す

    if not misclassified:
        print("誤分類されたサンプルが見つかりませんでした。")
        return [] # 空リストを返す

    # 誤分類サンプルの可視化
    actual_num_samples = len(misclassified) # 実際に見つかった数
    n_cols = min(4, actual_num_samples)
    n_rows = (actual_num_samples + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 5, n_rows * 5)) # figsize を調整

    # 画像の逆正規化（configから取得）
    try:
        mean = config.get('dataset_mean', [0.485, 0.456, 0.406])
        std = config.get('dataset_std', [0.229, 0.224, 0.225])
        inv_normalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
    except ZeroDivisionError:
        print("エラー: 設定ファイル内の dataset_std にゼロが含まれています。")
        return misclassified # 画像表示はできないがリストは返す
    except Exception as e:
        print(f"エラー: 逆正規化の設定中にエラー: {e}")
        return misclassified # 画像表示はできないがリストは返す

    # クラス名を取得（configから）
    class_names = config.get('class_names', None)
    if class_names is None:
        try:
            # misclassified からラベルの最大値を取得して推定
            all_true_labels = [item[1] for item in misclassified]
            all_pred_labels = [item[2] for item in misclassified]
            num_classes = int(max(max(all_true_labels), max(all_pred_labels)) + 1)
            class_names = [f"Class {i}" for i in range(num_classes)]
            print(f"警告: configにclass_namesが見つかりません。推定されたクラス名を使用: {class_names}")
        except Exception:
             class_names = ["Unknown"] * 3 # 仮
             print(f"警告: configにclass_namesが見つからず、推定もできませんでした。仮のクラス名を使用: {class_names}")

    print(f"{actual_num_samples} 個の誤分類サンプルを可視化します...")
    for i, (img, true_label, pred_label, confidence) in enumerate(misclassified):
        ax = plt.subplot(n_rows, n_cols, i+1)

        # 画像を逆正規化して表示用に変換（すでにCPU上のテンソル）
        try:
            img_np = inv_normalize(img).permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)  # 値を0-1の範囲にクリップ
            ax.imshow(img_np)
        except Exception as e:
            print(f"警告: サンプル {i} の画像表示中にエラー: {e}")
            ax.set_title("表示エラー")
            ax.axis('off')
            continue # 次のサンプルへ

        # クラス名を使用するか、ラベルの数値を使用するか
        try:
            true_class = class_names[true_label] if 0 <= true_label < len(class_names) else f"Unknown({true_label})"
            pred_class = class_names[pred_label] if 0 <= pred_label < len(class_names) else f"Unknown({pred_label})"
            title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}"
            ax.set_title(title, color='red') # 誤分類なので赤色
        except IndexError:
             ax.set_title(f"ラベルエラー\nTrue:{true_label}, Pred:{pred_label}", color='orange')
        except Exception as e:
             print(f"警告: サンプル {i} のラベル表示中にエラー: {e}")
             ax.set_title("ラベル表示エラー", color='orange')

        ax.axis('off')

    plt.tight_layout()

    # 保存するか表示するか
    if save_path:
        try:
            # 保存先ディレクトリが存在しない場合は作成
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path)
            print(f"誤分類サンプルの可視化を保存しました: {save_path}")
        except Exception as e:
            print(f"エラー: 誤分類サンプルの保存中にエラー: {e}")
    else:
        plt.show()
    plt.close() # メモリ解放のためフィギュアを閉じる

    return misclassified

# --- 削除またはコメントアウト: analyze_feature_importance 関数 ---
# モデルが reasoning を返さなくなったため、この関数は現状動作しない
# def analyze_feature_importance(config, checkpoint_path=None, metric="f1", save_path=None):
#     """
#     モデルの特徴重要度を分析する（中間表現の活性化値を使用）
#     *** 注意: 現在のモデルは reasoning を返さないため、この関数は動作しません ***
#     """
#     print("警告: analyze_feature_importance は現在のモデルでは動作しません。")
#     return None
#     # ... (元のコード) ...

def main():
    """
    コマンドラインからの実行用エントリーポイント
    """
    parser = argparse.ArgumentParser(description="学習済みモデルの詳細な可視化と分析")
    # --- 修正: config デフォルトパスを修正 ---
    # プロジェクトルートからの相対パスでデフォルトを指定
    default_config_path = os.path.join("configs", "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="設定ファイルのパス")
    parser.add_argument("--checkpoint", type=str, default=None, help="分析するモデルチェックポイントのパス")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "loss"], help="チェックポイント選択の基準")
    # --- 修正: mode の選択肢から 'feature' を削除 ---
    # 'all' は confusion と misclassified の両方を実行するように変更
    parser.add_argument("--mode", type=str, default="all",
                      choices=["confusion", "misclassified", "all"],
                      help="実行モード（confusion: 混同行列, misclassified: 誤分類サンプル, all: すべて）")
    parser.add_argument("--num_samples", type=int, default=8, help="誤分類サンプルを可視化する数")
    parser.add_argument("--save_dir", type=str, default=None, help="可視化結果の保存ディレクトリ (例: ./output/visualizations)")
    args = parser.parse_args()

    # 設定ファイルを読み込む
    # --- 修正: プロジェクトルートからの相対パスを絶対パスに変換 ---
    config_path = os.path.join(get_project_root(), args.config)
    try:
        config = load_config(config_path)
        print(f"設定ファイルを読み込みました: {config_path}")
    except FileNotFoundError:
        print(f"エラー: 設定ファイルが見つかりません: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 設定ファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

    # 保存ディレクトリの設定
    save_dir = args.save_dir
    if save_dir:
        # 絶対パスに変換
        save_dir = os.path.abspath(save_dir)
        try:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"保存ディレクトリを作成しました: {save_dir}")
        except OSError as e:
            print(f"エラー: 保存ディレクトリの作成中にエラーが発生しました: {e}")
            # 保存なしで続行するか、終了するか選択
            save_dir = None # 保存しない
            print("警告: 結果は保存されません。")

    # 混同行列モード
    if args.mode in ["confusion", "all"]:
        print("\n=== 混同行列の可視化 ===")
        save_path_cm = os.path.join(save_dir, "confusion_matrix.png") if save_dir else None
        cm, report = plot_confusion_matrix(config, args.checkpoint, args.metric, save_path_cm)

        # レポートをテキストファイルとして保存
        if report and save_dir:
            report_path = os.path.join(save_dir, "classification_report.txt")
            try:
                with open(report_path, "w", encoding='utf-8') as f:
                    f.write(report)
                print(f"分類レポートを保存しました: {report_path}")
            except IOError as e:
                print(f"エラー: 分類レポートの保存中にエラー: {e}")
        elif not report:
             print("分類レポートが生成されなかったため、保存されません。")

    # 誤分類サンプルモード
    if args.mode in ["misclassified", "all"]:
        print("\n=== 誤分類サンプルの可視化 ===")
        save_path_mis = os.path.join(save_dir, "misclassified_samples.png") if save_dir else None
        visualize_misclassified(config, args.checkpoint, args.metric, args.num_samples, save_path_mis)

    # --- 削除: 特徴重要度モードの呼び出し ---
    # if args.mode in ["feature", "all"]:
    #     print("\n=== 特徴重要度の可視化 ===")
    #     print("警告: 特徴重要度の分析は現在サポートされていません。")
    #     # save_path_feat = os.path.join(save_dir, "feature_importance.png") if save_dir else None
    #     # analyze_feature_importance(config, args.checkpoint, args.metric, save_path_feat)

if __name__ == "__main__":
    main()
