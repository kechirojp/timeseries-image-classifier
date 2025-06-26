# データディレクトリ構造

このディレクトリには、時系列画像分類プロジェクトで使用するデータファイルが格納さ### 2. 特徴量ファイル（例: `timeseries_15m_202412301431.csv`）
マルチモーダル学習で実際に使用される数値特徴量データ

**カラム構成:**
- `timestamp`: タイムスタンプ（YYYY-MM-DD HH:MM:SS形式）
- `feature_1` ～ `feature_6`: 6次元の数値特徴量
- `dataset_id`: データセット識別子

**命名規則:**
```
{dataset_name}_15m_{YYYYMMDD}{HHMM}.csv
```

**データ形式例:**
```csv
timestamp,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,dataset_id
2024-12-30 09:30:00,0.123,-0.456,0.789,-0.321,0.654,-0.987,dataset_a構造

```
data/
├── README.md                                    # このファイル
├── fix_labeled_data_timeseries_15m.csv         # 正解ラベルファイル（サンプル）
├── timeseries_15m_202412301431.csv             # 特徴量ファイル（サンプル）
├── dataset_a_15m_winsize40/                    # データセットA（画像データ）
│   ├── train/                                   # 訓練用データ
│   │   ├── class_0/                            # クラス0画像
│   │   ├── class_1/                            # クラス1画像
│   │   └── class_2/                            # クラス2画像
│   └── test/                                    # テスト用データ
│       ├── class_0/                            # クラス0画像
│       ├── class_1/                            # クラス1画像
│       └── class_2/                            # クラス2画像
├── dataset_b_15m_winsize40/                    # データセットB（同様の構造）
│   ├── train/
│   └── test/
└── dataset_c_15m_winsize40/                    # データセットC（同様の構造）
    ├── train/
    └── test/
```

## ファイル説明

### CSVファイル構造

### 1. 正解ラベルファイル（例: `fix_labeled_data_timeseries_15m.csv`）
マルチモーダル学習の参考データ

**注意: 現在の実装では、画像ディレクトリ構造（class_0/, class_1/, class_2/）からラベルを取得します。このCSVファイルは参考用またはカスタム実装用です。**

**カラム構成:**
- `timestamp`: タイムスタンプ（YYYY-MM-DD HH:MM:SS形式）
- `action`: アクション分類（0: class_0, 1: class_1, 2: class_2）
- `dataset_id`: データセット識別子

**命名規則:**
```
fix_labeled_data_{dataset_name}_15m.csv
```

**データ形式例:**
```csv
timestamp,action,dataset_id
2024-01-02 09:30:00,1,dataset_a
2024-01-02 09:45:00,2,dataset_a
...
```

### 2. 特徴量ファイル（例: `timeseries_15m_202412301431.csv`）
マルチモーダル学習で使用する数値特徴量データ（6次元）

**カラム構成:**
- `timestamp`: タイムスタンプ（YYYY-MM-DD HH:MM:SS形式）
- `feature_1` ~ `feature_6`: 6次元の数値特徴量
- `dataset_id`: データセット識別子

**命名規則:**
```
{dataset_name}_15m_{YYYYMMDDHHMI}.csv
```

**データ形式例:**
```csv
timestamp,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,dataset_id
2024-12-30 09:30:00,0.123,-0.456,0.789,-0.321,0.654,-0.987,dataset_a
...
```

### 画像データ

#### ディレクトリ命名規則
- `dataset_X_15m_winsize40`: データセット名_時間足_ウィンドウサイズ
- `train`/`test`: 訓練用/テスト用の分割
- `class_0`/`class_1`/`class_2`: 分類クラス（ディレクトリ名がそのままラベル値となります）

**重要**: ディレクトリ名`class_0`, `class_1`, `class_2`は、それぞれラベル値0, 1, 2に対応します。

#### 画像ファイル命名規則
```
{dataset_name}_15m_{YYYYMMDDHHMM}_label_{class_num}.png
```

**例:**
- `dataset_a_15m_202401020930_label_1.png`
- `dataset_b_15m_202401020945_label_2.png`

## データ準備手順

### マルチモーダル学習の動作フロー

1. **ラベル取得**: 画像ファイルのディレクトリ構造（`class_0/`, `class_1/`, `class_2/`）からラベルを取得
2. **タイムスタンプ抽出**: 画像ファイル名から日時情報を抽出（例: `dataset_a_15m_202401020930_label_1.png` → `2024-01-02 09:30:00`）
3. **特徴量マッチング**: 抽出したタイムスタンプで特徴量CSVから対応する時系列データを取得
4. **マルチモーダル入力**: 画像データ + 時系列特徴量データを組み合わせて学習

### 実データの準備手順

1. **時系列特徴量データ**: 上記形式のCSVファイルを`timeseries_15m_YYYYMMDDHHMM.csv`として配置
2. **画像データ**: 上記ディレクトリ構造で画像ファイルを配置
   - ファイル名は`{dataset_name}_15m_{YYYYMMDD}{HHMM}_label_{class_id}.png`形式
3. **設定確認**: `configs/config.yaml`でデータパスが正しく設定されていることを確認

**重要**: 画像ファイル名のタイムスタンプと特徴量CSVのタイムスタンプが一致している必要があります。

## 注意事項

- このリポジトリに含まれるCSVファイルはサンプル/ダミーデータです
- 実際のデータは適切なデータソースから取得してください
- 画像ファイル名のタイムスタンプと時系列データのタイムスタンプが対応している必要があります
- マルチモーダル学習では、画像とタイムスタンプが一致するデータのみが使用されます

## データサイズ目安

- **画像サイズ**: 380x380ピクセル（EfficientNet-B4推奨）
- **時系列ウィンドウ**: 40ステップ
- **特徴量次元**: 6次元
- **クラス数**: 3クラス（class_0, class_1, class_2）
