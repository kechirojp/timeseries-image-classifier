import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import traceback # traceback をインポート
import re # 正規表現モジュールをインポート
from lightning.pytorch import Trainer
from torchvision import transforms
 # 正規表現モジュールをインポート

# プロジェクト内の他のモジュールをインポートできるように設定
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

# --- 修正: モデルとユーティリティのインポート ---
# model_mode に応じて適切なモデルをインポートできるようにする
from model import StockClassifier
from multimodal_model import MultimodalStockClassifier # マルチモーダルモデルもインポート
# utils.py から find_best_checkpoint をインポート (後で修正)
# from utils import find_best_checkpoint # utils.py が存在しないか、関数がない場合はコメントアウト
from configs.config_utils import load_config, get_project_root # get_project_root もインポート
# --- 追加: datamodule のインポート ---
from datamodule import StockDataModule

# --- 追加: find_best_checkpoint 関数の定義 (utils.py がない場合) ---
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


def evaluate_model(config, checkpoint_path=None, metric="f1"):
    """
    学習済みモデルを評価し、精度やF1スコアを計算する

    Args:
        config: 設定辞書
        checkpoint_path: 評価するモデルチェックポイントのパス（Noneの場合は自動検出）
        metric: チェックポイントの選択基準（'f1'または'loss'）

    Returns:
        精度、F1スコアなどを含む評価結果
    """
    # チェックポイントが指定されていない場合は自動検出
    if checkpoint_path is None:
        # --- 修正: find_best_checkpoint を呼び出す ---
        checkpoint_path = find_best_checkpoint(config, metric)
        if checkpoint_path is None:
            print("有効なチェックポイントが見つかりません。評価を中止します。") # エラーメッセージを修正
            return None # 評価できない場合は None を返す

    print(f"評価に使用するチェックポイント: {checkpoint_path}")

    # データモジュールの準備
    try:
        data_module = StockDataModule(config)
        # data_module.setup("test") # setupはtrainer.test内で呼ばれる
    except Exception as e:
        print(f"エラー: DataModuleの初期化中にエラーが発生しました: {e}")
        return None

    # --- 修正: モデルのロード (model_mode に基づく) ---
    model_mode = config.get("model_mode", "single")
    try:
        if model_mode == "multi":
            print("マルチモーダルモデルをロードします...") # ログ追加
            model = MultimodalStockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        else:
            print("シングルモーダルモデルをロードします...") # ログ追加
            model = StockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        print(f"モデルを正常にロードしました: {type(model).__name__}")
    except FileNotFoundError:
        print(f"エラー: チェックポイントファイルが見つかりません: {checkpoint_path}")
        return None
    except Exception as e:
        print(f"エラー: モデルのロード中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return None

    # 評価の実行
    try:
        # --- 修正: トレーナー設定をconfigから取得 ---
        accelerator_setting = "gpu" if torch.cuda.is_available() and not config.get('force_cpu', False) else "cpu"
        # --- 修正: devices_setting を修正 ---
        # acceleratorが'gpu'の場合、'auto'ではなく利用可能なGPU数を指定するか、1を指定する方が安定することがある
        # ここでは 'auto' のままにするが、問題があれば 1 や torch.cuda.device_count() に変更を検討
        devices_setting = "auto" if accelerator_setting == "gpu" else 1
        precision_setting = config.get('precision', '32-true')

        trainer = Trainer(
            accelerator=accelerator_setting,
            devices=devices_setting,
            logger=False, # 評価時は通常ロガー不要
            precision=precision_setting
        )
        print(f"トレーナー設定: accelerator='{accelerator_setting}', devices='{devices_setting}', precision='{precision_setting}'") # 設定確認ログ
        results = trainer.test(model, datamodule=data_module)
        print("評価が完了しました。") # 完了ログ
    except Exception as e:
        print(f"エラー: モデルの評価中にエラーが発生しました: {e}")
        traceback.print_exc()
        return None

    return results

def visualize_predictions(config, checkpoint_path=None, metric="f1", num_samples=8, save_path=None):
    """
    モデルの予測結果を可視化する

    Args:
        config: 設定辞書
        checkpoint_path: 使用するモデルチェックポイントのパス（Noneの場合は自動検出）
        metric: チェックポイントの選択基準（'f1'または'loss'）
        num_samples: 表示するサンプル数
        save_path: 可視化結果の保存パス（Noneの場合は表示のみ）
    """
    # チェックポイントが指定されていない場合は自動検出
    if checkpoint_path is None:
        # --- 修正: find_best_checkpoint を呼び出す ---
        checkpoint_path = find_best_checkpoint(config, metric)
        if checkpoint_path is None:
            print("有効なチェックポイントが見つかりません。可視化を中止します。") # エラーメッセージを修正
            return # 何も返さない

    print(f"可視化に使用するチェックポイント: {checkpoint_path}") # ログ追加

    # データモジュールの準備
    try:
        data_module = StockDataModule(config)
        data_module.setup("test") # 可視化のためにテストデータを準備
    except Exception as e:
        print(f"エラー: DataModuleの初期化またはセットアップ中にエラーが発生しました: {e}")
        return

    # --- 修正: モデルのロード (model_mode に基づく) ---
    model_mode = config.get("model_mode", "single")
    try:
        if model_mode == "multi":
            print("マルチモーダルモデルをロードします...") # ログ追加
            model = MultimodalStockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        else:
            print("シングルモーダルモデルをロードします...") # ログ追加
            model = StockClassifier.load_from_checkpoint(checkpoint_path, config=config, strict=False)
        print(f"モデルを正常にロードしました: {type(model).__name__}")
    except FileNotFoundError:
        print(f"エラー: チェックポイントファイルが見つかりません: {checkpoint_path}")
        return
    except Exception as e:
        print(f"エラー: モデルのロード中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
        return

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
        return

    # テストデータローダーからサンプルを取得
    try:
        test_loader = data_module.test_dataloader()
        # --- 修正: マルチモーダル対応 ---
        # バッチの内容は DataModule の実装に依存する
        # ここでは画像とラベルのタプルを仮定
        batch = next(iter(test_loader))
        if model_mode == "multi":
            # マルチモーダルの場合のバッチ形式に合わせて調整が必要
            # 例: (images, numerical_features), labels = batch
            print("警告: マルチモーダルモデルのバッチ処理は未実装です。画像のみと仮定します。")
            images, labels = batch # 仮
        else:
            images, labels = batch
    except StopIteration:
        print("エラー: テストデータローダーが空です。")
        return
    except Exception as e:
        print(f"エラー: テストデータローダーからのデータ取得中にエラー: {e}")
        return

    # データも同じデバイスに移動
    try:
        # --- 修正: マルチモーダル対応 ---
        if model_mode == "multi":
            # images, numerical_features = images.to(device), numerical_features.to(device) # 例
            images = images.to(device) # 仮
        else:
            images = images.to(device)
        # labels は後でCPUで使うので移動しないか、必要なら移動して後で戻す
        # labels = labels.to(device)
    except Exception as e:
        print(f"エラー: 画像データのデバイス転送中にエラー: {e}")
        return

    # 予測を実行
    print("予測を実行中...")
    with torch.no_grad():
        try:
            # --- 修正: モデルのforward呼び出し (logitsのみを期待) ---
            if model_mode == "multi":
                # logits = model(images, numerical_features) # 例
                logits = model(images) # 仮
            else:
                logits = model(images)
            # --- 修正ここまで ---

            # 結果をCPUに戻してから計算（メモリ節約とNumPy変換のため）
            logits = logits.cpu()
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            print("予測完了。") # ログ追加
        except Exception as e:
            print(f"エラー: モデルの予測中にエラーが発生しました: {e}") # エラーメッセージを修正
            traceback.print_exc()
            return # エラー時は終了

    # 画像の逆正規化（表示用）
    # --- 修正: configからmean/stdを取得 ---
    try:
        mean = config.get('dataset_mean', [0.485, 0.456, 0.406])
        std = config.get('dataset_std', [0.229, 0.224, 0.225])
        inv_normalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
    except ZeroDivisionError:
        print("エラー: 設定ファイル内の dataset_std にゼロが含まれています。")
        return
    except Exception as e:
        print(f"エラー: 逆正規化の設定中にエラー: {e}")
        return

    # 結果の可視化
    n_cols = 4
    actual_num_samples = min(num_samples, len(images))
    if actual_num_samples == 0:
        print("表示するサンプルがありません。")
        return # サンプルがない場合は終了

    n_rows = (actual_num_samples + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 5, n_rows * 5)) # figsize を調整

    # クラス名のマッピング（configから取得）
    class_names = config.get('class_names', None)
    if class_names is None:
        # クラス名がない場合、ラベルの最大値からクラス数を推定
        try:
            num_classes = int(labels.max().item() + 1)
            class_names = [f"Class {i}" for i in range(num_classes)]
            print(f"警告: configにclass_namesが見つかりません。推定されたクラス名を使用: {class_names}")
        except Exception: # labels が空の場合など
             class_names = ["Unknown"] * 3 # 仮のクラス名
             print(f"警告: configにclass_namesが見つからず、推定もできませんでした。仮のクラス名を使用: {class_names}")

    # 各サンプルを表示
    print("結果をプロット中...")
    for i in range(actual_num_samples):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        # 画像をCPUに戻し、正規化を解除して表示用に次元を並び替え
        img_tensor = images[i].cpu() # 元のテンソルをCPUへ
        try:
            img = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
            # 値を0-1の範囲にクリップ (正規化解除で範囲外になる可能性)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        except Exception as e:
            print(f"警告: サンプル {i} の画像表示中にエラー: {e}")
            ax.set_title("表示エラー")
            ax.axis('off')
            continue # 次のサンプルへ

        # ラベルと予測を取得
        true_label_idx = labels[i].item()
        pred_label_idx = preds[i].item()

        # クラス名リスト外のインデックスアクセスを防ぐ
        true_label_name = class_names[true_label_idx] if 0 <= true_label_idx < len(class_names) else f"Unknown({true_label_idx})"
        pred_label_name = class_names[pred_label_idx] if 0 <= pred_label_idx < len(class_names) else f"Unknown({pred_label_idx})"

        # タイトルに真ラベル、予測ラベル、予測確率を表示
        title = f"True: {true_label_name}\nPred: {pred_label_name} (Prob: {probs[i][pred_label_idx]:.2f})"
        # 正解なら緑、不正解なら赤で表示
        color = "green" if true_label_idx == pred_label_idx else "red"
        ax.set_title(title, color=color)
        ax.axis('off') # 軸を非表示に

    plt.tight_layout()

    # 保存するか表示するか
    if save_path:
        try:
            # 保存先ディレクトリが存在しない場合は作成
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path)
            print(f"予測結果の可視化を保存しました: {save_path}")
        except Exception as e:
            print(f"エラー: 可視化結果の保存中にエラー: {e}")
    else:
        plt.show()
    plt.close() # メモリ解放のためフィギュアを閉じる

def main():
    """
    コマンドラインからの実行用エントリーポイント
    """
    parser = argparse.ArgumentParser(description="学習済みモデルの評価と可視化")
    # --- 修正: config デフォルトパスを修正 ---
    # プロジェクトルートからの相対パスでデフォルトを指定
    default_config_path = os.path.join("configs", "config.yaml")
    parser.add_argument("--config", type=str, default=default_config_path, help="設定ファイルのパス")
    parser.add_argument("--checkpoint", type=str, default=None, help="評価するモデルチェックポイントのパス")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "loss"], help="チェックポイント選択の基準")
    # --- 修正: mode の選択肢から 'reasoning' を削除 ---
    # 'all' は evaluate と visualize の両方を実行するように変更
    parser.add_argument("--mode", type=str, default="both", choices=["evaluate", "visualize", "both"],
                      help="実行モード（evaluate: 評価のみ、visualize: 可視化のみ、both: 評価と可視化）")
    parser.add_argument("--num_samples", type=int, default=8, help="可視化するサンプル数")
    parser.add_argument("--save_path", type=str, default=None, help="可視化結果の保存パス (例: ./output/predictions.png)")
    args = parser.parse_args()

    # 設定ファイルを読み込む
    # --- 修正: プロジェクトルートからの相対パスを絶対パスに変換 ---
    config_path = os.path.join(get_project_root(), args.config)
    try:
        config = load_config(config_path)
        print(f"設定ファイルを読み込みました: {config_path}")
    except FileNotFoundError:
        print(f"エラー: 設定ファイルが見つかりません: {config_path}")
        sys.exit(1) # エラー終了
    except Exception as e:
        print(f"エラー: 設定ファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit(1) # エラー終了

    # 評価モード
    if args.mode in ["evaluate", "both"]:
        print("\n=== モデル評価 ===") # セクションタイトル追加
        results = evaluate_model(config, args.checkpoint, args.metric)
        if results:
            print("\n評価結果:")
            # results はリストのリストになることがあるので、整形して表示
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
                 for key, value in results[0].items():
                     print(f"  {key}: {value:.4f}")
            else:
                 print(results) # そのまま表示
        else:
            print("評価に失敗しました。")

    # 可視化モード
    if args.mode in ["visualize", "both"]:
        print("\n=== 予測結果の可視化 ===") # セクションタイトル追加
        visualize_predictions(config, args.checkpoint, args.metric, args.num_samples, args.save_path)

    # --- 削除: 推論過程の可視化モード呼び出し ---

if __name__ == "__main__":
    main()
