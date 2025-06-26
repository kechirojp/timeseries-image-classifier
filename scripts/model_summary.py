"""
efficientnet_b4 のモデルサマリーを表示するスクリプト
Usage:
    python model_summary.py
"""

import sys

try:
    import torch
except ImportError as e:
    print(f"必要なライブラリ torch のインポートに失敗しました: {e}")
    sys.exit(1)

try:
    import timm
except ImportError as e:
    print(f"必要なライブラリ timm のインポートに失敗しました: {e}")
    sys.exit(1)

try:
    from torchinfo import summary
except ImportError:
    print("torchinfo がインストールされていません。モデルサマリーの表示には torchinfo を pip install torchinfo してください。")
    sys.exit(1)

def main():
    # デバイスを自動判定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # efficientnet_b4 モデルをロード（分類ヘッドなし）
    try:
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
    except (RuntimeError, ValueError) as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        sys.exit(1)

    model.to(device)

    # 入力サイズは公式が推奨する 3x224x224
    input_size = (1, 3, 224, 224)
    # サマリーを表示
    try:
        summary(model, input_size=input_size, device=str(device))
    except Exception as e:
        print(f"モデルサマリーの生成中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
