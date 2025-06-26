import os
import numpy as np
import pandas as pd # pandas をインポート
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, Dataset # Dataset をインポート
from torchvision import transforms
from torchvision.datasets import ImageFolder
import lightning.pytorch as pl
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import torch
import platform
from typing import List, Dict, Tuple, Optional, Union
import logging # ロギングを追加

logger = logging.getLogger(__name__) # ロガーを取得

# --- 追加: マルチモーダル用カスタムデータセット ---
class MultimodalDataset(Dataset):
    """
    ImageFolderデータセットをラップし、画像に対応する時系列データを追加するカスタムデータセット。
    """
    def __init__(self, image_folder_dataset: ImageFolder, timeseries_df: pd.DataFrame, window_size: int, feature_columns: List[str]):
        """
        Args:
            image_folder_dataset (ImageFolder): 元となるImageFolderデータセット。
            timeseries_df (pd.DataFrame): 時系列データを含むDataFrame。インデックスはTimestamp型であること。
            window_size (int): Transformerに入力する時系列データのウィンドウサイズ。
            feature_columns (List[str]): 使用する時系列特徴量のカラム名リスト。
        """
        self.image_folder_dataset = image_folder_dataset
        self.timeseries_df = timeseries_df # スケーリング済みのデータを期待
        self.window_size = window_size
        self.feature_columns = feature_columns

        # 画像ファイル名からタイムスタンプを抽出する処理を改善
        # 例: 'nasdaq100_15m_202401020930_label_1.png' -> pd.Timestamp('2024-01-02 09:30:00')
        self.timestamps = []
        self.valid_indices = [] # 有効なタイムスタンプを持つインデックスを保存
        for i, (img_path, _) in enumerate(self.image_folder_dataset.samples):
            try:
                # ファイル名から日付時刻部分を抽出 (より堅牢な方法)
                filename = os.path.basename(img_path)
                parts = filename.split('_')
                # 'YYYYMMDDHHMM' 形式の部分を探す
                datetime_str = None
                for part in parts:
                    if len(part) == 12 and part.isdigit():
                        datetime_str = part
                        break
                if datetime_str:
                    ts = pd.to_datetime(datetime_str, format='%Y%m%d%H%M')
                    # 時系列データフレームのインデックスに存在するか確認
                    if ts in self.timeseries_df.index:
                        self.timestamps.append(ts)
                        self.valid_indices.append(i) # 有効なインデックスとして追加
                    else:
                        # logger.warning(f"タイムスタンプ {ts} が時系列データインデックスに見つかりません。画像 {filename} をスキップします。")
                        self.timestamps.append(None) # スキップするデータとしてNoneを追加
                else:
                    # logger.warning(f"ファイル名から日付時刻を抽出できませんでした: {filename}。画像をスキップします。")
                    self.timestamps.append(None) # 抽出失敗時はNoneを追加
            except Exception as e:
                logger.error(f"エラー: ファイル名 '{filename}' のタイムスタンプ変換中にエラー: {e}。画像をスキップします。")
                self.timestamps.append(None)

        # 有効なインデックスのみを使用するようにフィルタリング
        self.original_indices = self.valid_indices
        logger.info(f"有効なタイムスタンプを持つ画像数: {len(self.original_indices)} / {len(self.image_folder_dataset)}")


    def __len__(self):
        # 有効なデータの数を返す
        return len(self.original_indices)

    def __getitem__(self, idx):
        """
        指定されたインデックスの画像、時系列データ、ラベルを返す。
        idx はフィルタリング後のインデックス (0 から len(self)-1)。
        """
        # フィルタリング後のインデックスから元のImageFolderのインデックスを取得
        original_idx = self.original_indices[idx]

        # 1. 画像とラベルを取得
        img, label = self.image_folder_dataset[original_idx]

        # 2. 対応するタイムスタンプを取得 (Noneでないはず)
        timestamp = self.timestamps[original_idx]
        if timestamp is None:
             # この状況は __init__ でフィルタリングされているため、基本的には発生しないはず
             logger.error(f"予期せぬエラー: インデックス {idx} (元:{original_idx}) のタイムスタンプがNoneです。")
             # 代替としてゼロデータを返すか、エラーを発生させる
             ts_data = torch.zeros((self.window_size, len(self.feature_columns)), dtype=torch.float32)
             return img, ts_data, label

        # 3. タイムスタンプに対応する時系列データを取得
        ts_data = torch.zeros((self.window_size, len(self.feature_columns)), dtype=torch.float32) # デフォルトはゼロ埋め
        try:
            # タイムスタンプをインデックスとして使用して、過去window_size分のデータを取得
            # loc は timestamp を含む行を取得
            # タイムスタンプがインデックスに存在するか確認 (再確認)
            if timestamp in self.timeseries_df.index:
                end_loc = self.timeseries_df.index.get_loc(timestamp)
                # 開始位置を計算 (end_locを含まないwindow_size個前)
                start_loc = max(0, end_loc - self.window_size + 1) # +1 して window_size 個にする

                # データを取得 (ilocを使用)
                # .values を使って NumPy 配列を取得
                ts_sequence = self.timeseries_df.iloc[start_loc : end_loc + 1][self.feature_columns].values

                # NumPy配列をTensorに変換
                ts_tensor = torch.tensor(ts_sequence, dtype=torch.float32)

                # 取得したデータがwindow_sizeに満たない場合はゼロパディング
                if ts_tensor.shape[0] < self.window_size:
                    padding_size = self.window_size - ts_tensor.shape[0]
                    # 前方にゼロパディング
                    padding = torch.zeros((padding_size, ts_tensor.shape[1]), dtype=torch.float32)
                    ts_data = torch.cat((padding, ts_tensor), dim=0)
                else:
                    # ちょうどwindow_sizeかそれ以上の場合 (通常はちょうどのはず)
                    ts_data = ts_tensor[-self.window_size:] # 最後のwindow_size個を取得
            else:
                 # この警告も基本的には発生しないはず
                 logger.warning(f"警告: タイムスタンプ {timestamp} が時系列データインデックスに見つかりません (getitem)。ゼロデータを返します。")

        except KeyError:
            # この警告も基本的には発生しないはず
            logger.warning(f"警告: タイムスタンプ {timestamp} が時系列データインデックスに見つかりません (KeyError)。ゼロデータを返します。")
        except Exception as e:
            logger.error(f"エラー: 時系列データ取得中に予期せぬエラー (idx={idx}, orig_idx={original_idx}, ts={timestamp}): {e}")
            # エラー発生時もゼロデータを返す（あるいはNoneを返してcollate_fnで処理）
            ts_data = torch.zeros((self.window_size, len(self.feature_columns)), dtype=torch.float32)


        # 画像、時系列データ、ラベルのタプルを返す
        return img, ts_data, label
# --- カスタムデータセットここまで ---


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    画像および時系列データセットの読み込み、前処理、データ分割を行うクラス。
    シングルモーダルとマルチモーダルの両方に対応。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config # config全体を保持
        # --- base_dir の設定 ---
        # config に base_dir があればそれを使用、なければカレントディレクトリ
        self.base_dir = config.get("base_dir", os.getcwd())
        logger.info(f"プロジェクトベースディレクトリ: {self.base_dir}")

        self.batch_size = config.get("batch_size", 32)
        self.model_mode = config.get("model_mode", "single") # モデルモードを追加
        logger.info(f"モデルモード: {self.model_mode}")

        # --- マルチモーダル設定の読み込み ---
        self.timeseries_config = config.get("timeseries", {}) if self.model_mode == "multi" else {}
        if self.model_mode == "multi":
            self.ts_data_path = self.timeseries_config.get("data_path")
            self.ts_feature_columns = self.timeseries_config.get("feature_columns", [])
            self.ts_window_size = self.timeseries_config.get("window_size", 40)
            if not self.ts_data_path or not self.ts_feature_columns:
                logger.error("マルチモーダルモードが選択されましたが、'timeseries' 設定 (data_path, feature_columns) が不十分です。")
                raise ValueError("マルチモーダルモードには timeseries 設定が必要です。")
            # 時系列データパスが相対パスの場合、base_dir を基準にする
            if not os.path.isabs(self.ts_data_path):
                self.ts_data_path = os.path.join(self.base_dir, self.ts_data_path)
            logger.info(f"時系列データパス: {self.ts_data_path}")
            logger.info(f"使用する時系列特徴量: {self.ts_feature_columns}")
            logger.info(f"時系列ウィンドウサイズ: {self.ts_window_size}")

        # --- 環境判定とデータローダー設定 ---
        # 実行環境の検出
        self.is_windows = platform.system() == 'Windows'

        # Google Colab環境の検出を試みる
        self.is_colab = False
        try:
            # google.colab のインポートを試みるのが最も確実
            import google.colab
            self.is_colab = True
            logger.info("google.colab モジュールのインポートに成功しました。Colab環境と判定します。")
        except ImportError:
            logger.info("google.colab モジュールのインポートに失敗しました。Colab環境ではない可能性があります。")
            # 予備的なチェック (IPython経由) - Colab以外でもIPythonは使われるため注意
            try:
                import IPython
                ipython_instance = IPython.get_ipython()
                if ipython_instance and 'google.colab' in str(ipython_instance):
                    self.is_colab = True
                    logger.info("IPython情報からColab環境を検出しました。")
                else:
                    logger.info("IPython情報からはColab環境を検出できませんでした。")
            except (ImportError, NameError, AttributeError):
                logger.info("IPython環境の検出に失敗しました。")
                self.is_colab = False

        # num_workers の決定ロジック
        # configからnum_workersを取得、なければデフォルト値Noneを使用
        config_num_workers = config.get("num_workers", None)
        logger.info(f"Configから取得したnum_workers: {config_num_workers} (Noneは未設定)")

        default_workers_non_windows = 4 # Windows以外のデフォルトワーカー数

        if self.is_colab:
            # Colabの場合: configに指定があればそれを優先、なければデフォルト値を使用
            self.num_workers = config_num_workers if config_num_workers is not None else default_workers_non_windows
            self.pin_memory = config.get("pin_memory", True) # config値 or True
            self.prefetch_factor = config.get("prefetch_factor", 2)
            self.persistent_workers = self.num_workers > 0 and config.get("persistent_workers", True) # ワーカーがいればTrue
            logger.info(f"Google Colab環境を検出。num_workers={self.num_workers} (config値優先、デフォルト{default_workers_non_windows})")
        elif self.is_windows:
            # Windowsの場合: 強制的に0
            self.num_workers = 0
            self.pin_memory = config.get("pin_memory", False) # Windowsではpin_memory=Falseが安全な場合がある
            self.prefetch_factor = None
            self.persistent_workers = False
            logger.warning("Windows環境を検出。num_workersを強制的に0に設定します。データローディングが遅くなる可能性があります。")
        else:
            # その他のLinux/Macの場合: configに指定があればそれを優先、なければデフォルト値を使用
            self.num_workers = config_num_workers if config_num_workers is not None else default_workers_non_windows
            self.pin_memory = config.get("pin_memory", True) # config値 or True
            self.prefetch_factor = config.get("prefetch_factor", 2)
            self.persistent_workers = self.num_workers > 0 and config.get("persistent_workers", True) # ワーカーがいればTrue
            logger.info(f"Linux/Mac (非Colab) 環境を検出。num_workers={self.num_workers} (config値優先、デフォルト{default_workers_non_windows})")

        # データローダー設定の概要を表示
        logger.info(f"最終的なデータローダー設定: workers={self.num_workers}, persistent={self.persistent_workers}, pin_memory={self.pin_memory}, prefetch_factor={self.prefetch_factor}")

        # その他の設定値
        self.num_folds = config.get("num_folds", 5)
        self.fold = config.get("fold", 0)
        self.seed = config.get("seed", 42)
        self.img_size = config.get("image_size", 224)

        # ディレクトリ設定
        # logs_dir は config から取得し、なければ logs サブディレクトリ
        self.logs_dir = config.get("logs_dir", os.path.join(self.base_dir, "logs"))
        if not os.path.isabs(self.logs_dir):
            self.logs_dir = os.path.join(self.base_dir, self.logs_dir)
        logger.info(f"ログディレクトリ: {self.logs_dir}")

        # data_dir は config から取得し、なければ data サブディレクトリ
        self.data_dir = config.get("data_dir", os.path.join(self.base_dir, "data"))
        if not os.path.isabs(self.data_dir):
            self.data_dir = os.path.join(self.base_dir, self.data_dir)
        logger.info(f"データディレクトリ (ルート): {self.data_dir}")


        # --- 銘柄データディレクトリの設定 ---
        self.symbols = config.get("symbols", ["nasdaq100"])
        self.symbol_dirs = {}
        for symbol in self.symbols:
            symbol_key = f"{symbol}_dir"
            if symbol_key in config:
                # configにパスがあればそれを使用
                symbol_path = config[symbol_key]
                # 相対パスの場合は base_dir を結合
                if not os.path.isabs(symbol_path):
                    symbol_path = os.path.join(self.base_dir, symbol_path)
            else:
                # configになければデフォルトパスを生成 (base_dir基準)
                # 例: ./data/dataset_a_15m_winsize40
                symbol_path = os.path.join(self.base_dir, f"{symbol}_15m_winsize40")

            if os.path.isdir(symbol_path):
                self.symbol_dirs[symbol] = symbol_path
                logger.info(f"銘柄 '{symbol}' の画像ディレクトリ: {symbol_path}")
            else:
                logger.warning(f"銘柄 '{symbol}' のディレクトリが見つかりません: {symbol_path}")
        if not self.symbol_dirs:
             raise FileNotFoundError("設定された銘柄の画像ディレクトリが一つも見つかりませんでした。")


        # --- データ変換 ---
        # train と val/test で異なる変換を定義
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.RandomHorizontalFlip() は金融時系列データでは不適切なため削除
            # 水平方向の反転は時系列の順序を逆にしてしまい、データの特性が変わる
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.val_test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- データセット変数の初期化 ---
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.full_dataset = None # 全データセットを保持
        self.timeseries_df = None # マルチモーダル用時系列データ

    def prepare_data(self):
        """
        データのダウンロードや前処理など、一度だけ実行する処理。
        ここでは主に時系列データの読み込みを行う。
        """
        if self.model_mode == "multi":
            if self.ts_data_path and os.path.exists(self.ts_data_path):
                try:
                    # 時系列データを読み込み、タイムスタンプをインデックスに設定
                    df = pd.read_csv(self.ts_data_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    # 欠損値を平均値で補完 (より高度な補完も検討可能)
                    for col in self.ts_feature_columns:
                        if df[col].isnull().any():
                            mean_val = df[col].mean()
                            df[col].fillna(mean_val, inplace=True)
                            logger.info(f"時系列特徴量 '{col}' の欠損値を平均値 ({mean_val:.4f}) で補完しました。")
                    # 簡単な標準化 (StandardScaler)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    df[self.ts_feature_columns] = scaler.fit_transform(df[self.ts_feature_columns])
                    logger.info("時系列データの特徴量を標準化しました。")

                    self.timeseries_df = df[self.ts_feature_columns] # 必要なカラムのみ保持
                    logger.info(f"時系列データを読み込みました: {self.ts_data_path}, サイズ: {self.timeseries_df.shape}")
                except Exception as e:
                    logger.error(f"時系列データの読み込みまたは前処理中にエラーが発生: {e}")
                    raise
            else:
                logger.error(f"マルチモーダルモードですが、時系列データファイルが見つかりません: {self.ts_data_path}")
                raise FileNotFoundError(f"時系列データファイルが見つかりません: {self.ts_data_path}")

    def setup(self, stage: Optional[str] = None):
        """
        データセットの読み込みと分割を行う。
        stage は 'fit', 'validate', 'test', 'predict' のいずれか。
        """
        logger.info(f"データセットのセットアップを開始 (stage: {stage})")

        # --- 全画像データセットの読み込み ---
        all_symbol_datasets = []
        for symbol, symbol_dir in self.symbol_dirs.items():
            symbol_datasets_parts = [] # train と test を一時的に格納するリスト
            for split in ["train", "test"]: # train と test の両方を読み込む
                split_dir = os.path.join(symbol_dir, split)
                if os.path.isdir(split_dir):
                    try:
                        # ImageFolderに初期transformとしてNoneを設定
                        dataset_part = ImageFolder(root=split_dir, transform=None)
                        logger.info(f"銘柄 '{symbol}' の '{split}' データを読み込みました: {len(dataset_part)} 件")
                        # --- デバッグ用: 各ImageFolderのラベル分布を確認 ---
                        try:
                            unique_labels, counts = np.unique(dataset_part.targets, return_counts=True)
                            label_counts = dict(zip(unique_labels, counts))
                            logger.info(f"  '{split}' データのラベル分布: {label_counts}")
                            if 2 not in label_counts or label_counts[2] == 0:
                                logger.warning(f"    警告: 銘柄 '{symbol}' の '{split}' データにラベル2が含まれていません！ ディレクトリ '{split_dir}' を確認してください。")
                        except Exception as e_inner:
                            logger.error(f"    '{split}' データのラベル分布確認中にエラー: {e_inner}")
                        # --- デバッグ用ログここまで ---
                        symbol_datasets_parts.append(dataset_part)
                    except FileNotFoundError:
                        logger.warning(f"ディレクトリが見つかりません: {split_dir}")
                    except Exception as e:
                         logger.error(f"銘柄 '{symbol}' の '{split}' データ読み込み中にエラー: {e}")
                         # エラーが発生しても処理を続ける場合があるかもしれないが、基本的には問題
                         # raise e # 必要に応じてエラーを再発生させる
                else:
                    logger.warning(f"銘柄 '{symbol}' に '{split}' ディレクトリが見つかりません: {split_dir}")

            if symbol_datasets_parts:
                # train と test データセットを結合
                combined_symbol_dataset = ConcatDataset(symbol_datasets_parts)
                logger.info(f"銘柄 '{symbol}' の train/test データを結合しました: 合計 {len(combined_symbol_dataset)} 件")
                all_symbol_datasets.append(combined_symbol_dataset)
            else:
                logger.warning(f"銘柄 '{symbol}' で読み込めるデータ (train/test) がありませんでした。")


        if not all_symbol_datasets:
            raise ValueError("読み込める画像データセットがありません。")

        # 複数の銘柄データセットを結合
        # (各要素はすでに train/test が結合された ConcatDataset)
        self.full_image_dataset = ConcatDataset(all_symbol_datasets)
        logger.info(f"全銘柄の画像データ合計: {len(self.full_image_dataset)} 件")

        # --- デバッグ用: 最終的な full_image_dataset のラベル分布を確認 ---
        full_image_labels = []
        try:
            # full_image_dataset は ConcatDataset の ConcatDataset になっている
            for symbol_concat_ds in self.full_image_dataset.datasets: # 各銘柄のConcatDataset
                for imagefolder_ds in symbol_concat_ds.datasets: # 各ImageFolder (train or test)
                    if isinstance(imagefolder_ds, ImageFolder):
                        full_image_labels.extend(imagefolder_ds.targets)
                    else:
                        logger.warning(f"予期しないデータセットタイプがネストされています: {type(imagefolder_ds)}")

            if full_image_labels:
                unique_labels, counts = np.unique(full_image_labels, return_counts=True)
                label_counts = dict(zip(unique_labels, counts))
                logger.info(f"結合後の全画像データセット (full_image_dataset) のラベル分布: {label_counts}")
                if 2 not in label_counts or label_counts[2] == 0:
                     logger.warning("警告: 結合後の全画像データセットにラベル2が含まれていません！")
            else:
                logger.warning("結合後の全画像データセットのラベルを取得できませんでした。")
        except Exception as e:
            logger.error(f"結合後の全画像データセットのラベル分布確認中にエラー: {e}", exc_info=True)
        # --- デバッグ用ログここまで ---

        # --- マルチモーダルデータセットの作成 (必要な場合) ---
        if self.model_mode == "multi":
            if self.timeseries_df is None:
                 # prepare_dataが呼ばれていない場合 (DDPなどで発生する可能性)
                 logger.warning("時系列データがロードされていません。prepare_dataを呼び出します。")
                 self.prepare_data()
                 if self.timeseries_df is None: # それでもロードできない場合
                     raise RuntimeError("時系列データの準備に失敗しました。")

            # ImageFolderをMultimodalDatasetでラップ
            self.full_dataset = MultimodalDataset(
                image_folder_dataset=self.full_image_dataset, # ConcatDatasetを渡す
                timeseries_df=self.timeseries_df,
                window_size=self.ts_window_size,
                feature_columns=self.ts_feature_columns
            )
            logger.info(f"マルチモーダルデータセットを作成しました。有効データ数: {len(self.full_dataset)}")
        else:
            # シングルモーダルの場合は、ImageFolder (ConcatDataset) をそのまま使用
            self.full_dataset = self.full_image_dataset
            logger.info("シングルモーダルモードのため、画像データセットのみを使用します。")


        # --- 全データセットのラベルを取得 (層化分割のため) ---
        full_labels = []
        try:
            # full_dataset のタイプに応じてラベル取得方法を分岐
            if isinstance(self.full_dataset, MultimodalDataset):
                # マルチモーダルの場合は MultimodalDataset の __getitem__ 経由で取得
                for i in range(len(self.full_dataset)):
                    _, _, label = self.full_dataset[i]
                    full_labels.append(label)
            elif isinstance(self.full_dataset, ConcatDataset):
                # シングルモーダルの場合 (ネストしたConcatDatasetを想定)
                # self.full_dataset は self.full_image_dataset と同じ
                # self.full_image_dataset.datasets は各銘柄のConcatDatasetのリスト
                for symbol_concat_ds in self.full_dataset.datasets:
                    if isinstance(symbol_concat_ds, ConcatDataset):
                        # 各銘柄のConcatDataset内のImageFolder (train/test) を走査
                        for imagefolder_ds in symbol_concat_ds.datasets:
                            if isinstance(imagefolder_ds, ImageFolder):
                                full_labels.extend(imagefolder_ds.targets)
                            else:
                                logger.warning(f"銘柄データセット内に予期しないタイプ: {type(imagefolder_ds)}")
                    elif isinstance(symbol_concat_ds, ImageFolder):
                         # 万が一、ネストされていない場合 (以前の構造など)
                         logger.warning("予期しないデータ構造: ConcatDataset内に直接ImageFolderが見つかりました。")
                         full_labels.extend(symbol_concat_ds.targets)
                    else:
                        logger.warning(f"全データセット内に予期しないタイプ: {type(symbol_concat_ds)}")

            elif isinstance(self.full_dataset, ImageFolder):
                # 単一のImageFolderの場合 (通常は発生しないはず)
                full_labels = self.full_dataset.targets
            else:
                logger.warning(f"分割前のデータセットタイプが不明です: {type(self.full_dataset)}")

            if not full_labels:
                # ここでエラーが発生していた
                raise ValueError("全データセットのラベルを取得できませんでした。層化分割を実行できません。")

            full_labels = np.array(full_labels) # NumPy配列に変換
            unique_labels, counts = np.unique(full_labels, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))
            logger.info(f"データ分割前の全データセット (full_dataset) のラベル分布: {label_counts}")
            # ラベル2の存在チェックは継続
            required_label = 2 # チェックしたいラベル
            if required_label not in label_counts or label_counts[required_label] == 0:
                logger.warning(f"警告: データ分割前の全データセットにラベル{required_label}のデータが含まれていません！")

        except ValueError as ve: # 具体的な例外を捕捉
             logger.error(f"全データセットのラベル取得中にエラー: {ve}", exc_info=True)
             raise # ラベルがないと層化分割できないためエラーを再発生させる
        except Exception as e: # その他の予期せぬエラー
            logger.error(f"全データセットのラベル取得中に予期せぬエラー: {e}", exc_info=True)
            raise
        # --- ラベル取得ここまで ---

        # --- K-Fold または Train/Val/Test 分割 ---
        num_samples = len(self.full_dataset)
        indices = np.arange(num_samples) # インデックスをNumPy配列で取得

        if self.num_folds > 1:
            logger.info(f"{self.num_folds}-Fold 層化交差検証を実行します (Fold {self.fold})")
            # StratifiedKFold を使用
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            all_splits = list(skf.split(indices, full_labels)) # splitにラベルを渡す

            if self.fold >= self.num_folds:
                raise ValueError(f"foldインデックス({self.fold})が分割数({self.num_folds})を超えています。")

            train_val_indices, test_indices = all_splits[self.fold]

            # train_val_indices をさらに train/val に層化分割 (例: 80%/20%)
            val_split_ratio = 0.2
            # train_valセット内のラベルを取得
            train_val_labels = full_labels[train_val_indices]
            try:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=val_split_ratio,
                    random_state=self.seed,
                    stratify=train_val_labels # 層化を指定
                )
            except ValueError as e:
                 logger.warning(f"Train/Valの層化分割中にエラー: {e}。検証セットが小さすぎるか、特定のクラスのサンプルが少なすぎる可能性があります。層化なしで分割します。")
                 # 層化できない場合はランダムに分割 (元のKFoldに近い挙動)
                 train_indices, val_indices = train_test_split(
                     train_val_indices,
                     test_size=val_split_ratio,
                     random_state=self.seed
                 )

            logger.info(f"Fold {self.fold}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

            # Subsetを作成 (transformは後で適用)
            train_subset = Subset(self.full_dataset, train_indices)
            val_subset = Subset(self.full_dataset, val_indices)
            test_subset = Subset(self.full_dataset, test_indices)

        else:
            # 単純な Train/Val/Test 層化分割 (例: 70%/15%/15%)
            logger.info("単純な Train/Val/Test 層化分割を実行します。")
            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 1.0 - train_ratio - val_ratio

            # 1. 全データを Train+Val と Test に層化分割
            try:
                train_val_indices, test_indices = train_test_split(
                    indices,
                    test_size=test_ratio,
                    random_state=self.seed,
                    stratify=full_labels
                )
            except ValueError as e:
                 logger.warning(f"Train+Val/Testの層化分割中にエラー: {e}。テストセットが小さすぎるか、特定のクラスのサンプルが少なすぎる可能性があります。層化なしで分割します。")
                 train_val_indices, test_indices = train_test_split(
                     indices,
                     test_size=test_ratio,
                     random_state=self.seed
                 )

            # 2. Train+Val を Train と Val に層化分割
            # Train+Val 内での Val の割合を計算 (元の全体比率から)
            relative_val_ratio = val_ratio / (train_ratio + val_ratio)
            train_val_labels = full_labels[train_val_indices]
            try:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=relative_val_ratio,
                    random_state=self.seed,
                    stratify=train_val_labels
                )
            except ValueError as e:
                 logger.warning(f"Train/Valの層化分割中にエラー: {e}。検証セットが小さすぎるか、特定のクラスのサンプルが少なすぎる可能性があります。層化なしで分割します。")
                 train_indices, val_indices = train_test_split(
                     train_val_indices,
                     test_size=relative_val_ratio,
                     random_state=self.seed
                 )

            logger.info(f"分割結果: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

            # Subsetを作成
            train_subset = Subset(self.full_dataset, train_indices)
            val_subset = Subset(self.full_dataset, val_indices)
            test_subset = Subset(self.full_dataset, test_indices)

        # --- 各データセットに適切なTransformを適用 --- (DataLoader側で適用するため削除)

        # setup完了時にデータセットを保持
        self.train_dataset = train_subset
        self.val_dataset = val_subset
        self.test_dataset = test_subset

        # --- デバッグ用: 検証データセットのラベル分布を確認 ---
        if stage in ['fit', 'validate', None]: # fit または validate の場合に確認
            val_labels = []
            try:
                # Subsetの場合、dataset属性とindices属性を持つ
                if hasattr(self.val_dataset, 'dataset') and hasattr(self.val_dataset, 'indices'):
                    original_dataset = self.val_dataset.dataset
                    indices = self.val_dataset.indices

                    # --- 再帰的にラベルを取得するヘルパー関数 ---
                    def get_labels_recursive(dataset, idx_list):
                        labels = []
                        if isinstance(dataset, Subset): # Subsetの場合、さらに内部へ
                            # Subsetのインデックスを元のデータセットのインデックスに変換
                            original_indices = [dataset.indices[i] for i in idx_list]
                            labels.extend(get_labels_recursive(dataset.dataset, original_indices))
                        elif isinstance(dataset, ConcatDataset): # ConcatDatasetの場合
                            cumulative_sizes = dataset.cumulative_sizes
                            for idx in idx_list:
                                dataset_idx = np.searchsorted(cumulative_sizes, idx, side='right')
                                if dataset_idx == 0:
                                    sample_idx = idx
                                else:
                                    sample_idx = idx - cumulative_sizes[dataset_idx - 1]
                                # 再帰呼び出しで内部のデータセットからラベルを取得
                                # 渡すインデックスは、内部データセット内でのインデックス [sample_idx]
                                labels.extend(get_labels_recursive(dataset.datasets[dataset_idx], [sample_idx]))
                        elif isinstance(dataset, ImageFolder): # ImageFolderの場合、ラベルを取得
                            # idx_list に含まれるインデックスを使ってラベルを取得
                            labels.extend([dataset.targets[i] for i in idx_list])
                        elif isinstance(dataset, MultimodalDataset): # MultimodalDatasetの場合
                             # MultimodalDataset の __getitem__ を使ってラベルを取得
                             # idx_list は Subset が持つ、元の MultimodalDataset に対するインデックスリスト
                             for idx in idx_list:
                                 try:
                                     # Subset のインデックス idx は MultimodalDataset の 0 から len()-1 の範囲
                                     _, _, label = dataset[idx]
                                     labels.append(label)
                                 except IndexError:
                                     logger.warning(f"MultimodalDatasetでインデックスエラー (Subset経由): {idx}")
                                 except Exception as e_multi_subset:
                                     logger.warning(f"MultimodalDataset(Subset経由)からのラベル取得エラー: {e_multi_subset}")
                        else:
                            # 予期しないデータセットタイプの場合、警告を出す
                            logger.warning(f"get_labels_recursive内の予期しないデータセットタイプ: {type(dataset)}")
                        return labels
                    # --- ヘルパー関数ここまで ---

                    # 再帰関数を呼び出してラベルを取得
                    val_labels = get_labels_recursive(original_dataset, indices)

                # random_split で直接分割された場合 (Subsetではない) - 現在のコードでは発生しないはず
                elif hasattr(self.val_dataset, 'dataset') and not hasattr(self.val_dataset, 'indices'):
                     logger.warning("検証データセットが予期しない形式です (random_split?)。ラベル分布の確認をスキップします。")
                else:
                     # Subsetでもなく、上記形式でもない場合
                     logger.warning(f"検証データセットのタイプが不明または予期しない形式です: {type(self.val_dataset)}")


                if val_labels:
                    unique_labels, counts = np.unique(val_labels, return_counts=True)
                    label_counts = dict(zip(unique_labels, counts))
                    logger.info(f"検証データセット (val_dataset) のラベル分布: {label_counts}")
                    required_label = 2 # チェックしたいラベル
                    if required_label not in label_counts or label_counts[required_label] == 0:
                        # 警告メッセージを修正
                        logger.warning(f"警告: 検証データセットにラベル{required_label}のデータが含まれていないか、ゼロ件です！")
                else:
                    # ここで警告が出ていた箇所
                    logger.warning("検証データセットのラベルを取得できませんでした。get_labels_recursive の実装を確認してください。")

            except Exception as e:
                logger.error(f"検証データセットのラベル分布確認中にエラー: {e}", exc_info=True)
        # --- デバッグ用ログここまで ---


        logger.info("データセットのセットアップが完了しました。")


    # --- DataLoaderメソッド --- (TransformedDataset ラッパーを使用)
    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup(stage='fit')

        class TransformedDataset(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __getitem__(self, index):
                data = self.subset[index]
                if self.transform:
                    # dataが (img, label) または (img, ts_data, label) の形式
                    if len(data) == 2: # シングルモーダル
                        img, label = data
                        # ImageFolderがPIL Imageを返すことを想定
                        img = self.transform(img)
                        return img, label
                    elif len(data) == 3: # マルチモーダル
                        img, ts_data, label = data
                        # 画像にのみtransformを適用
                        img = self.transform(img)
                        return img, ts_data, label
                    else:
                        raise ValueError("予期しないデータ形式です。")
                else:
                    return data # transformがない場合はそのまま返す

            def __len__(self):
                return len(self.subset)

        train_ds = TransformedDataset(self.train_dataset, self.train_transform)

        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup(stage='validate')

        class TransformedDataset(Dataset): # train_dataloaderと同じものを再利用
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            def __getitem__(self, index):
                data = self.subset[index]
                if self.transform:
                    if len(data) == 2: img, label = data; img = self.transform(img); return img, label
                    elif len(data) == 3: img, ts_data, label = data; img = self.transform(img); return img, ts_data, label
                    else: raise ValueError("予期しないデータ形式です。")
                else: return data
            def __len__(self):
                return len(self.subset)

        val_ds = TransformedDataset(self.val_dataset, self.val_test_transform)

        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup(stage='test')

        class TransformedDataset(Dataset): # train_dataloaderと同じものを再利用
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            def __getitem__(self, index):
                data = self.subset[index]
                if self.transform:
                    if len(data) == 2: img, label = data; img = self.transform(img); return img, label
                    elif len(data) == 3: img, ts_data, label = data; img = self.transform(img); return img, ts_data, label
                    else: raise ValueError("予期しないデータ形式です。")
                else: return data
            def __len__(self): return len(self.subset)

        test_ds = TransformedDataset(self.test_dataset, self.val_test_transform)

        return DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0 and self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )