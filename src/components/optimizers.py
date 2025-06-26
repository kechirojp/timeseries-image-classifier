import torch

# NFNet用のAGC（Adaptive Gradient Clipping）カスタム実装
class AGC(torch.optim.Optimizer):
    """
    NFNetのために設計されたAdaptive Gradient Clipping
    論文: "High-Performance Large-Scale Image Recognition Without Normalization"
    """
    def __init__(self, optimizer, clip_factor=0.01, eps=1e-3):
        # optimizer: ラップする元のオプティマイザ
        # clip_factor: クリッピング係数 (パラメータノルムに対する勾配ノルムの最大比率)
        # eps: ゼロ除算を防ぐための小さな値
        if clip_factor <= 0.0:
            raise ValueError("clip_factor should be positive.")
        if eps <= 0.0:
            raise ValueError("eps should be positive.")

        self.optimizer = optimizer
        self.clip_factor = float(clip_factor)  # 明示的に浮動小数点数に変換
        self.eps = float(eps)  # 明示的に浮動小数点数に変換

        # オプティマイザのプロパティを継承 (スケジューラなどがアクセスするため)
        self.defaults = optimizer.defaults
        self._optimizer = optimizer  # 内部オプティマイザを保持 (デバッグ用)

    # PyTorch Learning Rate Schedulerが必要とする属性を公開
    @property
    def param_groups(self):
        """オプティマイザのパラメータグループを返すプロパティ"""
        return self.optimizer.param_groups

    @property
    def state(self):
        """オプティマイザの状態を返すプロパティ"""
        return self.optimizer.state

    def step(self, closure=None):
        """
        勾配クリッピングを実行し、内部オプティマイザのステップを実行する。
        Args:
            closure (callable, optional): 損失を再評価するためのクロージャ。
        Returns:
            Optional[float]: closureが指定された場合の損失。
        """
        # 各パラメータグループ内のパラメータに対して処理
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                # 勾配が存在しない場合はスキップ
                if p.grad is None:
                    continue

                # パラメータと勾配のノルムを計算
                # L2ノルム (Frobeniusノルム) を使用
                param_norm = torch.norm(p.data.float(), p=2)
                grad_norm = torch.norm(p.grad.data.float(), p=2)

                # パラメータノルムと勾配ノルムがゼロでない場合のみクリッピングを適用
                if param_norm > self.eps and grad_norm > self.eps:
                    # クリッピング閾値 (最大許容勾配ノルム) を計算
                    max_norm = self.clip_factor * param_norm

                    # 勾配ノルムが閾値を超えているかチェック
                    # clip_coef = max_norm / (grad_norm + self.eps) # 元の実装に近い形
                    # より安定した計算 (PyTorchの実装参考)
                    total_norm = grad_norm
                    clip_coef = max_norm / (total_norm + self.eps)

                    # クリッピング係数が1未満の場合、勾配をスケーリング
                    if clip_coef < 1:
                        # p.grad.data.copy_(p.grad.data * clip_coef) # 元の実装
                        # inplace演算でメモリ効率を改善
                        p.grad.detach().mul_(clip_coef.to(p.grad.device))

        # 内部オプティマイザのステップを実行
        # closure引数を正しく渡す
        if closure is not None:
            return self.optimizer.step(closure=closure)
        else:
            return self.optimizer.step()

    def zero_grad(self, set_to_none: bool = False):
        """内部オプティマイザの勾配をゼロにする"""
        # PyTorch 1.7.0以降の set_to_none 引数に対応
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # 他のオプティマイザメソッドの委譲 (必要に応じて追加)
    def add_param_group(self, param_group):
        """内部オプティマイザにパラメータグループを追加する"""
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        """内部オピティマイザの状態辞書を返す"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """内部オプティマイザの状態辞書を読み込む"""
        self.optimizer.load_state_dict(state_dict)

    def __repr__(self):
        return f"AGC(optimizer={self.optimizer}, clip_factor={self.clip_factor}, eps={self.eps})"

    # __getstate__ と __setstate__ は通常、内部オプティマイザが処理するため不要