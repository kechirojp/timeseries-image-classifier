import os
import yaml
import json

def load_config(config_path):
    """
    指定されたパスのYAMLまたはJSON形式の設定ファイルを読み込み、
    設定辞書を返す。ファイルが存在しなければ例外を発生させる。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが存在しません: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError(f"サポートされていない形式です: {config_path}")

def get_project_root():
    """
    このユーティリティファイルから見たプロジェクトのルートディレクトリの絶対パスを返す。
    現在の実装は、このファイル (config_utils.py) がプロジェクトルート直下の
    サブディレクトリ (例: configs) に存在することを想定しています。
    """
    # __file__ はこのスクリプトファイルのパス (例: j:\マイドライブ\NFNet_Classifier_pretrained\configs\config_utils.py)
    # os.path.dirname(__file__) はこのスクリプトファイルが存在するディレクトリ (例: j:\マイドライブ\NFNet_Classifier_pretrained\configs)
    # os.path.join(..., "..") は親ディレクトリへのパス (例: j:\マイドライブ\NFNet_Classifier_pretrained)
    # os.path.abspath(...) は絶対パスに変換します
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))