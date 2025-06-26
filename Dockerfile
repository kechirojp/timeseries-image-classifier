# ベースイメージとして CUDA 12.1 と開発ツールを含む Ubuntu 22.04 を使用
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# メタデータの追加
LABEL maintainer="kechiro <kechirojp@example.com>"
LABEL description="Production-ready time-series image classifier using EfficientNet/NFNet with PyTorch Lightning"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/kechirojp/timeseries-image-classifier"

# 環境変数を設定 (非対話モードでインストール)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 必要なシステムパッケージのインストールとアップデート
# Python 3.9 をインストールするために deadsnakes PPA を追加
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    python3-pip \
    git \
    && \
    # 不要なパッケージキャッシュを削除
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# python3.9 をデフォルトの python3 に設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# pip のアップグレードと setuptools のインストール
RUN python3 -m pip install --upgrade pip setuptools

# アプリケーションコードを配置するディレクトリを作成
WORKDIR /app

# requirements.txt をコンテナにコピー
COPY requirements.txt .

# PyTorch, torchvision, torchaudio を指定のバージョンとインデックス URL でインストール
# CUDA 12.1 と互換性のあるバージョンを使用
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# requirements.txt に基づいて残りの Python パッケージをインストール
# --ignore-installed オプションを追加して既存パッケージとの衝突を回避
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

RUN pip install --no-cache-dir optuna-integration[pytorch_lightning]

# アプリケーションコードをコンテナにコピー (requirements.txt は既にコピー済みなので除外)
COPY . .

# コンテナ起動時に実行されるデフォルトコマンド
CMD ["python3", "main.py"]
