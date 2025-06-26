import lightning.pytorch as pl  # pytorch_lightning から変更
import logging
import torch  # torch をインポート

class StageWiseUnfreezeCallback(pl.Callback):
    """
    複数エポックに渡って段階的にバックボーンの凍結解除を実施するためのコールバック。
    
    コンストラクタで各ステージの開始エポックを指定し、
    該当のエポックになるとモデル側の対応メソッド（unfreeze_layerX）を呼び出す。
    
    NFNetとResNetの両方の構造に対応するようにエラーハンドリングを強化。
    学習率の調整は行わず、凍結解除のみを担当する。
    """
    def __init__(self, stage1_epoch=5, stage2_epoch=15, stage3_epoch=30):
        super().__init__()
        self.stage1_epoch = stage1_epoch  # 最も深い層の凍結解除タイミング
        self.stage2_epoch = stage2_epoch  # 次の層の凍結解除タイミング
        self.stage3_epoch = stage3_epoch  # さらに次の層の凍結解除タイミング
        self.logger = logging.getLogger(__name__)
        
    def on_train_epoch_start(self, trainer, pl_module):
        """
        各学習エポック開始時に呼び出され、現在のエポック数に応じてモデルの凍結解除メソッドを呼ぶ。
        エラーハンドリングを追加し、異常があっても学習が継続できるようにする。
        学習率の調整は行わない。
        """
        current_epoch = trainer.current_epoch
        try:
            if current_epoch == self.stage1_epoch:
                # Stage 1: 最も深い層（通常はLayer 4）の凍結解除
                if hasattr(pl_module, 'unfreeze_layer4'):
                    pl_module.unfreeze_layer4()
                    self.logger.info(f"Epoch {current_epoch}: Stage 1 (Layer 4) unfreezing triggered.")
                    self._log_trainable_params(pl_module)
                else:
                    self.logger.warning(f"Model does not have 'unfreeze_layer4' method.")
                    
            elif current_epoch == self.stage2_epoch:
                # Stage 2: 次の層（通常はLayer 3）の凍結解除
                if hasattr(pl_module, 'unfreeze_layer3'):
                    pl_module.unfreeze_layer3()
                    self.logger.info(f"Epoch {current_epoch}: Stage 2 (Layer 3) unfreezing triggered.")
                    self._log_trainable_params(pl_module)
                else:
                    self.logger.warning(f"Model does not have 'unfreeze_layer3' method.")
                    
            elif current_epoch == self.stage3_epoch:
                # Stage 3: さらに次の層（通常はLayer 2）の凍結解除
                if hasattr(pl_module, 'unfreeze_layer2'):
                    pl_module.unfreeze_layer2()
                    self.logger.info(f"Epoch {current_epoch}: Stage 3 (Layer 2) unfreezing triggered.")
                    self._log_trainable_params(pl_module)
                else:
                    self.logger.warning(f"Model does not have 'unfreeze_layer2' method.")
                    
        except AttributeError as ae:  # モデルにメソッドがない場合
            self.logger.error(f"Error during unfreezing (AttributeError) at epoch {current_epoch}: {ae}", exc_info=True)
        except RuntimeError as rte:  # PyTorch関連のランタイムエラー
            self.logger.error(f"Runtime error during callback at epoch {current_epoch}: {rte}", exc_info=True)
    
    def _log_trainable_params(self, model):
        """学習可能なパラメータの数をログに記録"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Trainable parameters: {trainable_params:,d} / Total parameters: {total_params:,d} ({100.0 * trainable_params / total_params:.2f}%)")