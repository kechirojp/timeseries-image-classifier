import os
import torch
import logging

def find_best_checkpoint(config, metric="f1"):
    """
    指定されたメトリックに基づいて最良のチェックポイントファイルを探す
    
    Args:
        config: 設定辞書
        metric: 'f1'または'loss'（デフォルトはF1スコア）
    
    Returns:
        最良のチェックポイントのパス。見つからない場合はNone
    """
    checkpoint_dir = os.path.join(config.get("base_dir", "./"), config.get("checkpoint_dir", "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        print(f"チェックポイントディレクトリが見つかりません: {checkpoint_dir}")
        return None
    
    # チェックポイントのファイル名パターンを検索
    # 実際のファイル名は "model_epoch_" で始まるものを探す
    prefix = "model_epoch_"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix)]
    
    if not checkpoints:
        print(f"{prefix}で始まるチェックポイントが見つかりません")
        print(f"利用可能なチェックポイントファイルを検索します")
        # 他の可能性のあるパターンを試す
        alternative_patterns = ["resume_model_epoch_", "last", "last_checkpoint"]
        for pattern in alternative_patterns:
            alt_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(pattern)]
            if alt_checkpoints:
                print(f"{pattern}で始まるチェックポイントが見つかりました: {len(alt_checkpoints)}個")
                checkpoints = alt_checkpoints
                break
    
    if not checkpoints:
        print(f"有効なチェックポイントが見つかりません")
        return None
    
    # F1の場合は最大値、損失の場合は最小値が最良
    if metric == "f1":
        # チェックポイントからF1スコアを抽出する関数
        def extract_f1(filename):
            try:
                # val_f1_0.6373.ckpt のようなパターンからF1値を抽出
                f1_part = filename.split("val_f1_")[-1].split(".ckpt")[0]
                return float(f1_part)
            except:
                return -float('inf')  # 抽出失敗時は最低値を返す
        
        # F1スコアでソートし、最大値を持つチェックポイントを選択
        checkpoints_with_f1 = [(f, extract_f1(f)) for f in checkpoints]
        checkpoints_with_f1.sort(key=lambda x: x[1], reverse=True)  # F1は大きい方が良いので降順
        best_checkpoint = os.path.join(checkpoint_dir, checkpoints_with_f1[0][0])
        print(f"最良のF1スコア {checkpoints_with_f1[0][1]} でチェックポイントを選択: {checkpoints_with_f1[0][0]}")
    else:
        # 損失値を抽出するための関数
        def extract_loss(filename):
            try:
                # val_loss_0.8376_val_f1 のようなパターンから損失値を抽出
                loss_part = filename.split("val_loss_")[-1].split("_val_f1")[0]
                return float(loss_part)
            except:
                return float('inf')  # 抽出失敗時は最大値を返す
        
        # 損失値でソートし、最小値を持つチェックポイントを選択
        losses = [(f, extract_loss(f)) for f in checkpoints]
        losses.sort(key=lambda x: x[1])  # 損失は小さい方が良いので昇順
        best_checkpoint = os.path.join(checkpoint_dir, losses[0][0])
        print(f"最小の損失値 {losses[0][1]} でチェックポイントを選択: {losses[0][0]}")
    
    return best_checkpoint

def is_running_in_colab():
    """
    Google Colab環境で実行されているかどうかを判断する関数
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False
        
def setup_logger(name, log_file, level=logging.INFO):
    """
    ロギング設定を構成する関数
    
    Args:
        name: ロガーの名前
        log_file: ログファイルのパス
        level: ログレベル (デフォルトはINFO)
        
    Returns:
        設定済みのロガーオブジェクト
    """
    # ロガーの作成
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ファイルハンドラの設定
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    return logger