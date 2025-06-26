import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
from torchmetrics.classification import F1Score, Accuracy, ConfusionMatrix # ConfusionMatrix をインポート
import timm

# 他のモジュールからインポート
from src.components.optimizers import AGC
# componentsディレクトリからPositionalEncodingをインポート
from src.components.positional_encoding import PositionalEncoding # この行を追加

class MultimodalClassifier(pl.LightningModule):
    """
    画像(NFNet/ResNet)と時系列特徴量(Transformer)を組み合わせたマルチモーダル分類モデル。
    転移学習戦略をフラグで切り替え可能:
    - 段階的凍結解除 (Progressive Unfreezing) - 画像バックボーンのみ対象
    - ステージ毎差分学習率 (Differential Learning Rates)
    """
    def __init__(self, config, num_classes=3):
        """
        モデルの初期化。
        Args:
            config (dict): 設定ファイルから読み込まれた辞書。
            num_classes (int): 分類するクラス数。
        """
        super().__init__()
        # config全体を保存 (self.hparamsでアクセス可能)
        self.save_hyperparameters(config)
        self.num_classes = num_classes
        self.img_size = config.get("image_size", 224) # 画像サイズ

        # --- 転移学習戦略フラグ ---
        self.use_progressive_unfreezing = config.get("use_progressive_unfreezing", False)

        # --- 学習率などのハイパーパラメータ ---
        self.lr_head = float(config.get("lr_head", 3e-4)) # ヘッド部分の学習率
        self.lr_backbone = float(config.get("lr_backbone", 3e-5)) # バックボーン部分の基本学習率
        self.lr_decay_rate = float(config.get("lr_decay_rate", 0.1)) # バックボーンのステージごとの減衰率
        self.weight_decay = float(config.get("weight_decay", 0.01)) # 重み減衰

        # --- NFNet用AGC設定 ---
        # configからuse_agcを取得、デフォルトはFalse (NFNet以外では非推奨のため)
        self.use_agc = config.get("use_agc", False)
        self.agc_clip_factor = config.get("agc_clip_factor", 0.01)
        self.agc_eps = config.get("agc_eps", 1e-3)

        # --- Transformer関連のハイパーパラメータ ---
        # configの'data'セクションまたはルートから取得 (multimodal configを想定)
        ts_config = config.get("timeseries", {}) # 時系列固有の設定をまとめることを推奨
        self.feature_dim_ts = ts_config.get("feature_dim", config.get("feature_dim_ts", 6)) # 時系列指標の次元数
        self.window_size = ts_config.get("window_size", config.get("window_size", 40)) # 時系列データのウィンドウサイズ

        transformer_config = config.get("transformer", {}) # Transformer固有の設定
        self.transformer_dim = transformer_config.get("dim", 128) # Transformer内部の次元数
        self.transformer_layers = transformer_config.get("layers", 2) # Transformerエンコーダ層の数
        self.transformer_heads = transformer_config.get("heads", 4) # Multi-Head Attentionのヘッド数
        self.transformer_dropout = transformer_config.get("dropout", 0.1) # Transformer内のDropout率
        self.transformer_ff_dim = transformer_config.get("ff_dim", self.transformer_dim * 4) # FeedForward層の次元数

        # --- 画像特徴抽出器 (EfficientNet_B4をデフォルトに) ---
        # configの'model'セクションから画像モデルタイプを取得
        image_model_config = config.get("model", {})
        self.image_model_type = image_model_config.get("type", "efficientnet_b4") # デフォルトをEfficientNet_B4に変更
        drop_path_rate = image_model_config.get("drop_path_rate", 0.2)

        try:
            print(f"timmライブラリから画像バックボーン '{self.image_model_type}' を読み込みます...")
            # timmから画像バックボーンをロード
            self.image_backbone = timm.create_model(
                self.image_model_type,
                pretrained=True,
                drop_path_rate=drop_path_rate,
                num_classes=0 # 特徴抽出器として使用
            )
            print(f"画像バックボーン '{self.image_model_type}' の読み込みに成功しました。")

            # 特徴抽出器部分を取得 (通常はbackbone全体)
            self.feature_extractor = self.image_backbone

            # 特徴次元数を取得
            dummy_input_img = torch.randn(1, 3, self.img_size, self.img_size)
            with torch.no_grad():
                dummy_output_img = self.feature_extractor(dummy_input_img)
                # 出力がタプルやリストの場合に対応
                if isinstance(dummy_output_img, (list, tuple)):
                    dummy_output_img = dummy_output_img[0]
                # 空間次元が残っている場合はGlobal Pooling
                if len(dummy_output_img.shape) > 2:
                    dummy_output_img = F.adaptive_avg_pool2d(dummy_output_img, (1, 1))
            # 平坦化して次元数を取得
            self.feature_dim_img = dummy_output_img.view(1, -1).size(1)
            print(f"画像バックボーン '{self.image_model_type}' の特徴次元数: {self.feature_dim_img}")

            # NFNet系の場合のみAGCを有効にするか最終決定
            if "nfnet" not in self.image_model_type and self.use_agc:
                 print(f"警告: AGCはNFNet系モデル推奨ですが、'{self.image_model_type}' で有効化されています。")
            elif "nfnet" in self.image_model_type and not self.use_agc:
                 print(f"情報: NFNet系モデル '{self.image_model_type}' ですが、AGCは設定により無効化されています。")


        except Exception as e:
            print(f"エラー: 画像バックボーン '{self.image_model_type}' の読み込みに失敗しました: {e}")
            print("フォールバックとして ResNet18 を使用します。")
            # フォールバック処理
            self.image_backbone = torchvision.models.resnet18(pretrained=True)
            self.image_model_type = "resnet18_fallback"
            # ResNetの場合は最後のFC層を除いた部分を特徴抽出器とする
            modules = list(self.image_backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*modules)
            # ResNet18の特徴次元は512
            self.feature_dim_img = 512
            self.use_agc = False # ResNetではAGCは使わない
            print(f"ResNet18 (Fallback) 特徴抽出器出力次元: {self.feature_dim_img}")

        # --- 画像特徴抽出器の初期状態設定 (フラグに基づく) ---
        if self.use_progressive_unfreezing:
            # 段階的凍結解除の場合: 初期は凍結
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("画像バックボーンのパラメータを初期状態で凍結しました (段階的凍結解除モード)。")
        else:
            # ステージ毎差分学習率の場合: 初期から学習可能
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            print("画像バックボーンのパラメータを初期状態で学習可能に設定しました (差分学習率モード)。")
        # --- 画像特徴抽出器ここまで ---

        # --- 時系列特徴抽出器 (Transformer) ---
        # 入力次元 (feature_dim_ts) をTransformer次元 (transformer_dim) に変換する線形層
        self.ts_input_proj = nn.Linear(self.feature_dim_ts, self.transformer_dim)

        # 位置エンコーディング (インポートしたクラスを使用)
        self.pos_encoder = PositionalEncoding(self.transformer_dim, self.transformer_dropout, max_len=self.window_size + 10)

        # Transformerエンコーダ層の定義
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_ff_dim,
            dropout=self.transformer_dropout,
            batch_first=True, # 入力形式を (batch, seq, feature) に
            activation=F.gelu # GELU活性化関数を使用 (ReLUも可)
        )
        # Transformerエンコーダ本体
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.transformer_layers
        )
        print(f"Transformerエンコーダを定義しました (dim={self.transformer_dim}, layers={self.transformer_layers}, heads={self.transformer_heads})")
        # --- 時系列特徴抽出器ここまで ---

        # --- 最終分類器 (画像特徴量 + Transformer出力 を結合) ---
        # 結合後の特徴次元数
        combined_feature_dim = self.feature_dim_img + self.transformer_dim
        # 分類器のDropout率をconfigから取得
        classifier_config = config.get("classifier", {})
        dropout1_rate = classifier_config.get("dropout1", 0.3)
        dropout2_rate = classifier_config.get("dropout2", 0.2)

        self.classifier = nn.Sequential(
            # LayerNormを追加して安定化を図ることも検討可能
            # nn.LayerNorm(combined_feature_dim),
            nn.Linear(combined_feature_dim, 256),
            nn.SiLU(), # または nn.GELU()
            nn.Dropout(p=dropout1_rate),
            nn.Linear(256, 128),
            nn.SiLU(), # または nn.GELU()
            nn.Dropout(p=dropout2_rate),
            nn.Linear(128, self.num_classes)
        )
        print(f"最終分類器を定義しました (入力次元={combined_feature_dim}, 出力次元={self.num_classes})")
        # --- 最終分類器ここまで ---

        # --- 損失関数 ---
        self.ce_loss = nn.CrossEntropyLoss()

        # --- メトリクス ---
        task_type = "multiclass"
        self.train_f1 = F1Score(task=task_type, num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task=task_type, num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task=task_type, num_classes=num_classes, average="macro")
        self.train_acc = Accuracy(task=task_type, num_classes=num_classes)
        self.val_acc = Accuracy(task=task_type, num_classes=num_classes)
        self.test_acc = Accuracy(task=task_type, num_classes=num_classes)
        # 検証用混同行列メトリクスを追加
        self.val_cm = ConfusionMatrix(task=task_type, num_classes=num_classes)

        # --- 画像バックボーンのステージ設定 (凍結解除/差分学習率の両方で必要) ---
        self._setup_stages() # このメソッドは画像バックボーンにのみ関連

    def _setup_stages(self):
        """
        ロードされた画像バックボーンモデルの構造に基づいて、
        段階的ファインチューニングまたは差分学習率のためのステージを特定する。
        モデルタイプに応じて適切なステージ分割を行う。
        ステージリスト `self.stages` は深い層（出力に近い層）から浅い層（入力に近い層）の順になるように格納する。
        """
        self.stages = [] # ステージリストを初期化
        print(f"画像バックボーン '{self.image_model_type}' のステージ構造を設定します...")

        # --- NFNet系のステージ検出 ---
        if "nfnet" in self.image_model_type:
            try:
                if hasattr(self.image_backbone, 'stages') and isinstance(self.image_backbone.stages, nn.Sequential):
                    self.stages = list(self.image_backbone.stages.children())
                    print(f"  NFNetの 'stages' モジュールから {len(self.stages)} 個のステージを検出しました。")
                else:
                    potential_stages = []
                    for name, module in self.image_backbone.named_children():
                        if name.startswith('stage'): potential_stages.append(module)
                    if potential_stages:
                        self.stages = potential_stages
                        print(f"  NFNetの 'stageX' モジュールから {len(self.stages)} 個のステージを検出しました。")
                    else:
                        print("  NFNetのステージ構造を自動検出できませんでした。代替手段を試みます。")
                        self._detect_stages_heuristically()
            except Exception as e:
                print(f"  NFNetステージ構造の取得中にエラー: {e}")
                self._detect_stages_heuristically()

        # --- EfficientNet系のステージ検出 ---
        elif "efficientnet" in self.image_model_type:
            try:
                if hasattr(self.image_backbone, 'blocks') and isinstance(self.image_backbone.blocks, nn.Sequential):
                    self.stages = list(self.image_backbone.blocks.children())
                    print(f"  EfficientNetの 'blocks' モジュールから {len(self.stages)} 個のステージ (ブロックグループ) を検出しました。")
                else:
                    print("  EfficientNetの 'blocks' モジュールが見つかりません。代替手段を試みます。")
                    self._detect_stages_heuristically()
            except Exception as e:
                print(f"  EfficientNetステージ構造の取得中にエラー: {e}")
                self._detect_stages_heuristically()

        # --- ResNet系のステージ検出 (Fallback含む) ---
        elif "resnet" in self.image_model_type:
            try:
                resnet_layers = []
                # feature_extractor (ResNetのFC層抜き) から layer を探す
                for name, module in self.feature_extractor.named_children():
                    if name.startswith('layer'): resnet_layers.append(module)
                if len(resnet_layers) == 4:
                    self.stages = resnet_layers
                    print(f"  ResNet系の 'layer' モジュールから {len(self.stages)} 個のステージを検出しました。")
                else:
                    print(f"  ResNet系の 'layer' モジュールが期待通りに見つかりません ({len(resnet_layers)}個検出)。代替手段を試みます。")
                    self._detect_stages_heuristically()
            except Exception as e:
                print(f"  ResNetステージ構造の取得中にエラー: {e}")
                self._detect_stages_heuristically()

        # --- その他のモデルタイプや検出失敗時の汎用処理 ---
        else:
            print(f"画像バックボーン '{self.image_model_type}' のためのステージ自動検出を試みます。")
            self._detect_stages_heuristically()

        # --- ステージリストの最終調整 ---
        if not self.stages:
             print("警告: 画像バックボーンのステージ分割に失敗しました。バックボーン全体を単一ステージとして扱います。")
             self.stages = [self.feature_extractor]
        else:
             # ステージリストを逆順にする（深い層 -> 浅い層）
             self.stages.reverse()
             print(f"最終的な画像バックボーンステージ数: {len(self.stages)} (深い層から浅い層の順に格納)")

        # 各ステージのパラメータ数を表示 (逆順になっているのでreversedで元の順に戻して表示)
        print("画像バックボーン ステージごとのパラメータ数:")
        for i, stage in enumerate(reversed(self.stages)):
            # stageがnn.ModuleListの場合も考慮
            if isinstance(stage, nn.ModuleList):
                 params_frozen = sum(p.numel() for submodule in stage for p in submodule.parameters() if not p.requires_grad)
                 total_params = sum(p.numel() for submodule in stage for p in submodule.parameters())
            else:
                 params_frozen = sum(p.numel() for p in stage.parameters() if not p.requires_grad) # 凍結中のパラメータ数
                 total_params = sum(p.numel() for p in stage.parameters())
            status = "凍結中" if params_frozen > 0 else "学習可能"
            print(f"  Stage {i} (浅い層): {total_params - params_frozen:,} / {total_params:,} パラメータ ({status})")


    def _detect_stages_heuristically(self):
        """
        画像バックボーンのステージ構造が見つからない場合に、
        特徴抽出器のモジュール構造から経験的にステージを分割する代替手段。
        """
        print("  経験的なステージ検出を実行します...")
        potential_stages = [
            module for module in self.feature_extractor.children() if list(module.parameters())
        ]
        print(f"    特徴抽出器の主要な子モジュール数: {len(potential_stages)}")

        if len(potential_stages) >= 4:
            num_target_stages = 4
            modules_per_stage = len(potential_stages) // num_target_stages
            self.stages = []
            start_idx = 0
            for i in range(num_target_stages):
                end_idx = start_idx + modules_per_stage
                if i == num_target_stages - 1: end_idx = len(potential_stages)
                if start_idx < end_idx:
                    self.stages.append(nn.Sequential(*potential_stages[start_idx:end_idx]))
                start_idx = end_idx
            print(f"  特徴抽出器の子モジュールから{len(self.stages)}個のステージを構成しました。")
        elif potential_stages:
             self.stages = potential_stages
             print(f"  特徴抽出器の子モジュールをそのままステージとして使用します ({len(self.stages)}個)。")
        else:
            print("  ステージ分割可能な主要モジュールが見つかりませんでした。")


    def forward(self, x_img, x_ts):
        """
        フォワードパス。画像と時系列データの両方を受け取る。
        Args:
            x_img: 画像テンソル (batch, channels, height, width)
            x_ts: 時系列指標テンソル (batch, sequence_length, feature_dim_ts)
        Returns:
            logits: 分類器の出力 (batch, num_classes)
        """
        # 1. 画像特徴量の抽出
        img_features = self.feature_extractor(x_img) # (batch, C, H, W) or (batch, feature_dim_img)
        # Global Poolingが必要な場合 (出力が空間次元を持つ場合)
        if len(img_features.shape) > 2:
            img_features = F.adaptive_avg_pool2d(img_features, (1, 1))
        img_features = img_features.view(img_features.size(0), -1) # (batch, feature_dim_img)

        # 2. 時系列特徴量の抽出
        # 入力次元をTransformer次元に射影
        ts_projected = self.ts_input_proj(x_ts) # (batch, seq_len, transformer_dim)
        # 位置エンコーディングを追加
        ts_encoded = self.pos_encoder(ts_projected) # (batch, seq_len, transformer_dim)
        # Transformerエンコーダに入力
        # マスクは使用しない (オプションで追加可能: src_key_padding_mask)
        ts_output = self.transformer_encoder(ts_encoded) # (batch, seq_len, transformer_dim)
        # 最後のタイムステップの特徴量を使用 (CLSトークンを使う方法もある)
        ts_features = ts_output[:, -1, :] # (batch, transformer_dim)

        # 3. 特徴量の結合
        combined_features = torch.cat([img_features, ts_features], dim=1) # (batch, feature_dim_img + transformer_dim)

        # 4. 最終分類
        logits = self.classifier(combined_features) # (batch, num_classes)

        return logits

    def training_step(self, batch, batch_idx):
        """学習ステップ処理"""
        # バッチから画像、時系列データ、ラベルを取得
        # データモジュールの __getitem__ の返り値の順序に依存
        x_img, x_ts, y = batch
        # フォワードパス
        logits = self.forward(x_img, x_ts)

        # 損失計算
        loss = self.ce_loss(logits, y)

        # 予測とメトリクス更新
        preds = torch.argmax(logits, dim=1)
        self.train_f1.update(preds, y)
        self.train_acc.update(preds, y)

        # ログ記録 (batch_sizeを指定して平均損失を正しく計算)
        batch_size = x_img.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """検証ステップ処理"""
        x_img, x_ts, y = batch
        logits = self.forward(x_img, x_ts)
        loss = self.ce_loss(logits, y)

        # 予測とメトリクス更新
        preds = torch.argmax(logits, dim=1)
        self.val_f1.update(preds, y)
        self.val_acc.update(preds, y)
        # 混同行列を更新
        self.val_cm.update(preds, y)

        # ログ記録
        batch_size = x_img.size(0)
        # val_loss のログ記録のみ残す
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_end(self):
        """検証エポック終了時の処理"""
        # F1スコアとAccuracyを計算してログ記録
        val_f1 = self.val_f1.compute()
        val_acc = self.val_acc.compute()
        # ここでエポック終了時のメトリクスをログ記録する
        self.log('val_f1', val_f1, prog_bar=True) # メトリック名を 'val_f1' に統一 (PruningCallbackが監視するため)
        self.log('val_acc', val_acc, prog_bar=False) # メトリック名を 'val_acc' に統一

        # 混同行列を計算してターミナルに出力
        cm = self.val_cm.compute()
        print(f"Epoch {self.current_epoch} Validation Confusion Matrix:\n{cm}")

        # メトリクスをリセット (次のエポックのために重要)
        self.val_f1.reset()
        self.val_acc.reset()
        self.val_cm.reset()

    def test_step(self, batch, batch_idx):
        """テストステップ処理"""
        x_img, x_ts, y = batch
        logits = self.forward(x_img, x_ts)
        loss = self.ce_loss(logits, y)

        # 予測とメトリクス更新
        preds = torch.argmax(logits, dim=1)
        self.test_f1.update(preds, y)
        self.test_acc.update(preds, y)

        # ログ記録
        batch_size = x_img.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        return loss

    def on_test_epoch_end(self):
        """テストエポック終了時の処理 (オプション)"""
        test_f1 = self.test_f1.compute()
        test_acc = self.test_acc.compute()
        self.log('test_f1_epoch', test_f1) # エポック終了時の最終スコア
        self.log('test_acc_epoch', test_acc)
        # リセット
        self.test_f1.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        """オプティマイザとスケジューラを設定"""
        # パラメータグループを作成
        param_groups = [
            # 1. ヘッド部分 (最終分類器) - 高めの学習率
            {'params': self.classifier.parameters(), 'lr': self.lr_head, 'name': 'classifier'},
            # 2. Transformer関連 - バックボーン学習率 (常に学習対象)
            {'params': self.ts_input_proj.parameters(), 'lr': self.lr_backbone, 'name': 'ts_input_proj'},
            {'params': self.transformer_encoder.parameters(), 'lr': self.lr_backbone, 'name': 'transformer_encoder'},
            # PositionalEncoding は通常学習しない (パラメータを持たない)
        ]

        # 3. 画像バックボーンの各ステージ - 段階的な学習率
        # self.stages は入力に近い層（浅い層）から出力に近い層（深い層）の順
        num_stages = len(self.stages)
        for i, stage in enumerate(self.stages):
            # 浅い層 (i=0) ほど学習率が低く、深い層 (i=num_stages-1) ほど学習率が高い
            decay_power = num_stages - 1 - i  # 浅い層(入力側)ほど減衰が大きい
            stage_lr = self.lr_backbone * (self.lr_decay_rate ** decay_power)

            # stageがnn.ModuleListの場合も考慮
            stage_params = []
            modules_in_stage = stage if isinstance(stage, nn.ModuleList) else [stage]
            for module in modules_in_stage:
                 # 現在学習可能なパラメータのみを追加
                 stage_params.extend([p for p in module.parameters() if p.requires_grad])

            # 学習可能なパラメータが存在する場合のみグループを追加
            if stage_params:
                 param_groups.append({
                     'params': stage_params,
                     'lr': stage_lr,
                     # ステージ名はインデックス順
                     'name': f'img_stage_{i}'  # 入力側から0,1,2...
                 })
                 print(f"  Optimizer group: img_stage_{i} (Depth {num_stages - 1 - i}), lr: {stage_lr:.2e}, params: {len(stage_params)}")
            else:
                 # 段階的凍結解除モードの初期段階ではここに来ることがある
                 print(f"  情報: Img Stage {i} (Depth {num_stages - 1 - i}) に現在学習可能なパラメータがありません。スキップします。")


        # AdamWオプティマイザ
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        print(f"オプティマイザ: AdamW (weight_decay={self.weight_decay})")

        # AGCラッパー (有効かつNFNet系の場合)
        if self.use_agc and "nfnet" in self.image_model_type:
            try:
                optimizer = AGC(optimizer,
                                clip_factor=self.agc_clip_factor,
                                eps=self.agc_eps)
                print(f"Adaptive Gradient Clipping (AGC) を有効化しました (clip_factor={self.agc_clip_factor}, eps={self.agc_eps})")
            except Exception as agc_err:
                 print(f"警告: AGCの適用中にエラーが発生しました。AGCなしで続行します。エラー: {agc_err}")
                 # 元のオプティマイザに戻す
                 optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)


        # スケジューラの選択 (Cosine Annealingを推奨)
        scheduler_type = self.hparams.get("scheduler", "cosine")
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.get("max_epochs", 100),
                eta_min=self.hparams.get("eta_min", 1e-6)
            )
            print(f"CosineAnnealingLRスケジューラを使用します (T_max={self.hparams.get('max_epochs', 100)}, eta_min={self.hparams.get('eta_min', 1e-6)})")
        elif scheduler_type == "step":
             scheduler = torch.optim.lr_scheduler.StepLR(
                 optimizer,
                 step_size=self.hparams.get("scheduler_step_size", 30),
                 gamma=self.hparams.get("scheduler_gamma", 0.1)
             )
             print(f"StepLRスケジューラを使用します (step_size={self.hparams.get('scheduler_step_size', 30)}, gamma={self.hparams.get('scheduler_gamma', 0.1)})")
        else:
             print(f"警告: 未知のスケジューラタイプ '{scheduler_type}'。スケジューラは使用されません。")
             return [optimizer]

        # スケジューラの設定を辞書形式で返すのが推奨
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}]

    # --- 段階的ファインチューニングのための凍結解除メソッド (画像バックボーン用) ---
    # これらのメソッドは use_progressive_unfreezing = True の場合にコールバックから呼ばれる

    def unfreeze_stage(self, stage_index_from_deep):
        """
        指定されたインデックス（深い方から0）の画像バックボーンステージを凍結解除する。
        """
        if 0 <= stage_index_from_deep < len(self.stages):
            stage_to_unfreeze = self.stages[stage_index_from_deep]
            unfrozen_params_count = 0
            total_params_count = 0

            modules_to_unfreeze = stage_to_unfreeze if isinstance(stage_to_unfreeze, nn.ModuleList) else [stage_to_unfreeze]

            for module in modules_to_unfreeze:
                 for param in module.parameters():
                     if not param.requires_grad: # まだ凍結されているパラメータのみを対象
                          param.requires_grad = True
                          unfrozen_params_count += param.numel()
                     total_params_count += param.numel() # 総パラメータ数は常にカウント

            # ステージ名は浅い方から0とする (例: 全4ステージなら 3, 2, 1, 0)
            stage_name_from_shallow = len(self.stages) - 1 - stage_index_from_deep
            if unfrozen_params_count > 0:
                 print(f"Image Backbone Stage {stage_name_from_shallow} (Depth {stage_index_from_deep}) の凍結解除を実行。{unfrozen_params_count:,} 個のパラメータが新たに学習可能になりました (ステージ総数: {total_params_count:,})。")
                 # オプティマイザに新しいパラメータを追加する必要があるか確認 (通常は不要)
                 # self.trainer.strategy.setup_optimizers(self.trainer) # 必要に応じて
            else:
                 print(f"Image Backbone Stage {stage_name_from_shallow} (Depth {stage_index_from_deep}) は既に凍結解除済みか、パラメータがありません。")
        else:
            print(f"警告: 指定された画像バックボーンステージインデックス {stage_index_from_deep} は無効です (有効範囲: 0-{len(self.stages)-1})。")

    # コールバックから呼び出されることを想定したメソッド
    # モデルタイプによって層の名前が違うため、より汎用的な名前に変更
    def unfreeze_layer4(self): # 最も深いステージ (インデックス 0)
        print("Attempting to unfreeze Image Backbone Layer 4 (deepest stage)...")
        self.unfreeze_stage(0)

    def unfreeze_layer3(self): # 2番目に深いステージ (インデックス 1)
        print("Attempting to unfreeze Image Backbone Layer 3...")
        self.unfreeze_stage(1)

    def unfreeze_layer2(self): # 3番目に深いステージ (インデックス 2)
        print("Attempting to unfreeze Image Backbone Layer 2...")
        self.unfreeze_stage(2)

    # Layer 1 (最も浅いステージ) は通常、凍結解除の対象外とするか、
    # 非常に遅い段階で解除することが多い
