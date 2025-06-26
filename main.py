import os
import sys
import platform
import torch
import lightning.pytorch as pl

# プロジェクトルートをモジュールパスに追加
# main.py があるディレクトリの親ディレクトリ (プロジェクトルート) をパスに追加
project_root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root_dir)

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from configs.config_utils import load_config, get_project_root
from src.datamodule import TimeSeriesDataModule
from src.models.single_modal import SingleModalClassifier
from src.models.multi_modal import MultimodalClassifier
from src.callbacks import StageWiseUnfreezeCallback

def is_running_in_colab():
    """
    Google Colab環境で実行されているかどうかを判断する関数
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def main():
    # プロジェクトルートと設定ファイルのパスを取得
    project_root = get_project_root()

    # Google Colab環境検出と適切な設定ファイルの選択
    if is_running_in_colab():
        print("Google Colab 環境を検出しました。")
        config_file = "config_for_google_colab.yaml"
    else:
        print("ローカル環境で実行します。")
        config_file = "config.yaml"

    config_path = os.path.join(project_root, "configs", config_file)

    try:
        config = load_config(config_path)
        print(f"設定ファイルを読み込みました: {config_path}")
    except FileNotFoundError as e:
        print(f"エラー: 設定ファイルが見つかりません: {e}")
        sys.exit(1)
    except Exception as load_err:
        print(f"エラー: 設定ファイルの読み込み中にエラーが発生しました: {load_err}")
        sys.exit(1)

    # --- モデルモードとアーキテクチャ名の取得 ---
    model_mode = config.get("model_mode", "single") # デフォルトは single
    # アーキテクチャ名をconfigから取得 (例: 'nfnet', 'efficientnet', 'nfnet_transformer')
    # このキーはYAMLファイルで定義する必要があります
    model_architecture_name = config.get("model_architecture_name", "default_model")
    if model_architecture_name == "default_model":
        print("警告: configファイルに 'model_architecture_name' が定義されていません。デフォルト値 'default_model' を使用します。")

    # --- トライアル番号の取得 (新しい命名規則用) ---
    # configファイルに 'trial_number' が定義されていることを想定
    # 例: trial_number: 24
    trial_number = config.get("trial_number", "unknown") # 見つからない場合は "unknown" を使用

    # TensorCoreを活用するための精度設定（警告メッセージ対応）
    precision_setting = config.get('precision', '32-true') # デフォルトは '32-true'
    # 混合精度計算 ('16-mixed' または 'bf16-mixed') を使用する場合に設定
    if precision_setting in ['16-mixed', 'bf16-mixed']:
        try:
            # Tensor Core を効率的に使用するための設定 ('medium' または 'high')
            # 混合精度計算と併用することで、float32行列積のパフォーマンスを向上させる
            torch.set_float32_matmul_precision('medium')
            print(f"混合精度 ({precision_setting}) と併用するため、torch.set_float32_matmul_precision('medium') を設定しました。")
        except Exception as e:
            # エラーが発生しても処理は続行するが、警告を表示
            print(f"警告: torch.set_float32_matmul_precision の設定中にエラーが発生しました: {e}")
    else:
        # 混合精度を使用しない場合は設定不要
        print(f"精度設定: {precision_setting} (matmul精度設定はスキップ)")

    # --- ディレクトリ設定 (改善案に基づき修正) ---
    # ベースとなるログディレクトリとチェックポイントディレクトリ
    base_logs_dir = os.path.join(project_root, "logs")
    base_checkpoint_dir = os.path.join(project_root, "checkpoints")

    # モードとアーキテクチャに基づいたサブディレクトリを作成
    logs_dir = os.path.join(base_logs_dir, model_mode, model_architecture_name)
    checkpoint_dir = os.path.join(base_checkpoint_dir, model_mode, model_architecture_name)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"ログディレクトリ: {logs_dir}")
    print(f"チェックポイントディレクトリ: {checkpoint_dir}")

    # DataModuleの初期化 (内部で model_mode を参照する可能性あり)
    data_module = TimeSeriesDataModule(config)
    # data_module.setup("fit") # trainer.fitの中で呼ばれる

    # --- モデルの選択と初期化 --- (model_mode に基づく)
    if model_mode == "multi":
        print(f"マルチモーダルモデル ({model_architecture_name}) を初期化します。")
        # MultimodalClassifier は config 全体を受け取り、
        # 内部で 'multi_modal_settings' などを参照することを想定
        model = MultimodalClassifier(config)
    else: # single または未指定
        print(f"シングルモーダルモデル ({model_architecture_name}) を初期化します。")
        model = SingleModalClassifier(config)

    # TensorBoardのロガー設定 (改善案に基づき修正)
    # save_dir は logs/{model_mode}/{model_architecture_name}
    # name は実験名など、さらに詳細化したい場合に設定 (今回は空にするか固定値でも良い)
    # logger_name = f"classifier_{model_mode}_{model_architecture_name}" # より詳細な名前
    logger_name = "training_logs" # または experiment_name など config から取得
    logger = TensorBoardLogger(save_dir=logs_dir, name=logger_name, default_hp_metric=False)
    print(f"TensorBoardロガーを設定しました (save_dir={logs_dir}, name={logger_name})")

    # --- コールバック設定 --- (改善案に基づき修正)
    # ModelCheckpoint: val_f1が最高のモデルを保存
    # 新しい命名規則の例: efficientnet_b4_epoch_24_val_f1_0.7180_val_loss_0.5636_train_f1_0.701_train_loss_0.54.ckpt
    # この命名規則を適用するには、'train_f1' と 'train_loss' が training_step でログに記録されている必要があります。
    checkpoint_filename_format = (
        f'{model_architecture_name}_epoch={{epoch:02d}}_'  # trial_number を削除し、epoch を追加
        'val_f1={{val_f1:.4f}}_val_loss={{val_loss:.4f}}_'
        'train_f1={{train_f1:.4f}}_train_loss={{train_loss:.4f}}'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=checkpoint_dir, # モードとアーキテクチャ別のディレクトリを使用
        filename=checkpoint_filename_format, # 新しいファイル名形式
        save_top_k=1,
        mode='max',
        save_weights_only=False, # モデル全体を保存
        every_n_epochs=1,
        auto_insert_metric_name=False # filenameにメトリック名が自動挿入されるのを防ぐ
    )

    # EarlyStopping: val_f1が一定エポック数改善しなかったら停止
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=config.get('early_stopping_patience', 10),
        mode='max',
        verbose=True,
    )

    # 最後のエポックのチェックポイントも保存
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir, # モードとアーキテクチャ別のディレクトリを使用
        filename="last", # ファイル名は 'last.ckpt' になる
        save_last=True,
        save_weights_only=False, # モデル全体を保存
    )

    # 学習率モニター
    lr_monitor = LearningRateMonitor(logging_interval='epoch') # エポックごとにログ記録

    # コールバックリストの初期化
    callbacks = [checkpoint_callback, early_stop_callback, last_checkpoint_callback, lr_monitor]

    # --- 段階的凍結解除コールバックの条件付き追加 ---
    # configからフラグを読み込む (デフォルトはFalse)
    use_progressive_unfreezing = config.get("use_progressive_unfreezing", False)

    if use_progressive_unfreezing:
        print("段階的凍結解除 (Progressive Unfreezing) が有効です。")
        # auto_unfreeze設定が存在する場合のみコールバックを追加
        if config.get("auto_unfreeze"):
            unfreeze_config = config["auto_unfreeze"]
            # StageWiseUnfreezeCallback はモデルタイプに関わらず共通で使える想定
            # (モデル側で適切な unfreeze_layerX メソッドを持つ必要がある)
            unfreeze_callback = StageWiseUnfreezeCallback(
                stage1_epoch=unfreeze_config.get('stage1_epoch', 10),
                stage2_epoch=unfreeze_config.get('stage2_epoch', 20),
                stage3_epoch=unfreeze_config.get('stage3_epoch', 30)
            )
            callbacks.append(unfreeze_callback)
            print("StageWiseUnfreezeCallback を追加しました。")
        else:
            print("警告: use_progressive_unfreezing が true ですが、auto_unfreeze 設定が見つかりません。凍結解除は行われません。")
    else:
        print("ステージ毎差分学習率 (Differential Learning Rates) が有効です (段階的凍結解除は無効)。")

    # --- トレーナーの設定 --- (configから読み込み)
    trainer_config = {
        'accelerator': 'gpu' if torch.cuda.is_available() and not config.get('force_cpu', False) else 'cpu',
        'devices': 'auto' if torch.cuda.is_available() and not config.get('force_cpu', False) else 1,
        'max_epochs': config.get('max_epochs', 100),
        'logger': logger,
        'callbacks': callbacks,
        'precision': precision_setting, # configから読み込んだ精度設定
        'accumulate_grad_batches': config.get('accumulate_grad_batches', 1),
        'log_every_n_steps': config.get('log_every_n_steps', 50),
        # 'deterministic': True, # 再現性が必要な場合
        # 'gradient_clip_val': 0.5, # 必要に応じて勾配クリッピング
    }

    # 再開設定
    resume_checkpoint_path = config.get('resume_from_checkpoint')
    if resume_checkpoint_path:
        # 相対パスが指定された場合、現在のチェックポイントディレクトリからの相対パスと解釈する
        if not os.path.isabs(resume_checkpoint_path):
            resume_checkpoint_path = os.path.join(checkpoint_dir, resume_checkpoint_path)

        if os.path.exists(resume_checkpoint_path):
            print(f"チェックポイントから学習を再開します: {resume_checkpoint_path}")
            # trainer_config['ckpt_path'] = resume_checkpoint_path  # この行を削除
        else:
            print(f"警告: 指定された再開チェックポイントが見つかりません: {resume_checkpoint_path}")
            resume_checkpoint_path = None  # 見つからない場合はNoneに設定

    trainer = pl.Trainer(**trainer_config)

    # --- 学習の実行 ---
    print("学習を開始します...")
    # チェックポイントから再開する場合は、trainer.fitにckpt_pathを渡す
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_checkpoint_path)
    else:
        trainer.fit(model, datamodule=data_module)
    print("学習が完了しました。")

    # --- テストの実行 ---
    print("テストを開始します...")
    # bestモデルをロードしてテスト (ckpt_path='best' は v1.5以降)
    # trainer.test(model, datamodule=data_module, ckpt_path='best')
    # または、最後に保存された最良モデルのパスを指定
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"最良モデルをロードしてテスト: {best_model_path}")
        trainer.test(model, datamodule=data_module, ckpt_path=best_model_path)
    else:
        print("最良モデルのチェックポイントが見つからないため、現在のモデルでテストします。")
        trainer.test(model, datamodule=data_module)
    print("テストが完了しました。")

if __name__ == "__main__":
    main()