import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Transformer用の位置エンコーディング。
    PyTorchのチュートリアルに基づいた実装。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model: 埋め込み次元数。
            dropout: ドロップアウト率。
            max_len: 想定される最大シーケンス長。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置エンコーディング行列の計算
        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        # div_term: (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # (max_len, 1, d_model)
        # sinを偶数インデックスに、cosを奇数インデックスに適用
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # バッファとして登録 (モデルのstate_dictに含まれるが、パラメータではない)
        # 永続的バッファとして登録 (state_dictに保存される)
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルに位置エンコーディングを加算する。
        Args:
            x: 入力テンソル, shape [batch_size, seq_len, embedding_dim] (batch_first=Trueを想定)
        Returns:
            位置エンコーディングが加算されたテンソル。
        """
        # 入力が (batch_size, seq_len, embedding_dim) の場合 (batch_first=True)
        # peを (1, max_len, d_model) に変形して加算
        # xのシーケンス長に合わせてスライス
        # self.pe は (max_len, 1, dim) なので、転置して (1, max_len, dim) にする
        # さらに x のシーケンス長に合わせてスライス [:x.size(1)]
        x = x + self.pe[:x.size(1)].transpose(0, 1)

        return self.dropout(x)