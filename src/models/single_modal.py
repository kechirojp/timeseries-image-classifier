import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning.pytorch as pl
from torchmetrics.classification import F1Score, Accuracy, ConfusionMatrix
import timm
from torch.utils.checkpoint import checkpoint_sequential # 勾配チェックポイント用
from src.components.optimizers import AGC # NFNETに使われる動的勾配クリッピング
import numpy as np # numpy をインポート


class SingleModalClassifier(pl.LightningModule):
    """
    シングルモーダル画像分類モデル (NFNet, EfficientNetなどに対応)。
    転移学習戦略をフラグで切り替え可能:
    - 段階的凍結解除 (Progressive Unfreezing)
    - ステージ毎差分学習率 (Differential Learning Rates)
    """
    def __init__(self, config, num_classes=3):
        super().__init__()
        # config全体を保存 (hparamsとしてアクセス可能)
        self.save_hyperparameters(config)
        self.num_classes = num_classes
        self.img_size = config.get("image_size", 224)

        # --- 転移学習戦略フラグ ---
        self.use_progressive_unfreezing = config.get("use_progressive_unfreezing", False)
        
        # --- メモリ効率化設定 ---
        self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)

        # --- ハイパーパラメータ ---
        self.lr_head = float(config.get("lr_head", 3e-4))
        self.lr_backbone = float(config.get("lr_backbone", 3e-5))
        self.lr_decay_rate = float(config.get("lr_decay_rate", 0.1))
        self.weight_decay = float(config.get("weight_decay", 0.01))

        # --- モデル固有設定 ---
        self.model_config = config.get("model", {})
        # 設定ファイルからモデルタイプを取得 (デフォルトをefficientnet_b4に変更)
        self.model_type = self.model_config.get("type", "efficientnet_b4")
        drop_path_rate = self.model_config.get("drop_path_rate", 0.2)

        # --- AGC設定 (NFNet系でのみ有効化を推奨) ---
        self.use_agc = config.get("use_agc", False) # デフォルトはFalse
        self.agc_clip_factor = config.get("agc_clip_factor", 0.01)
        self.agc_eps = config.get("agc_eps", 1e-3)
        # NFNet系モデルを使う場合のみAGCを有効にするか判定
        if "nfnet" in self.model_type and config.get("use_agc") is None:
             print(f"NFNet系モデル ({self.model_type}) を検出しました。設定ファイルでuse_agcが指定されていないため、デフォルトでAGCを有効にします。")
             self.use_agc = True
        elif "nfnet" not in self.model_type and self.use_agc:
             print(f"警告: モデルタイプ '{self.model_type}' ではAGCは推奨されませんが、設定が有効になっています。")


        # --- モデルのロード (timmを使用) ---
        try:
            print(f"timmライブラリからモデル '{self.model_type}' を読み込みます...")
            self.backbone = timm.create_model(
                self.model_type,
                pretrained=True,
                drop_path_rate=drop_path_rate,
                num_classes=0
            )
            print(f"モデル '{self.model_type}' の読み込みに成功しました。")

            # 勾配チェックポイントの有効化（メモリ消費を抑えるため）
            if self.use_gradient_checkpointing and hasattr(self.backbone, 'set_grad_checkpointing'):
                self.backbone.set_grad_checkpointing(enable=True)
                print(f"モデル '{self.model_type}' で勾配チェックポイントを有効化しました。")
            
            # 特徴抽出器部分を取得 (モデル構造によって異なる場合がある)
            # 一般的なケース: 最後の分類層を除いた部分
            # Note: timmの多くのモデルは num_classes=0 で特徴量を出力する
            self.feature_extractor = self.backbone
            # 例外的な処理が必要なモデルはここに追加 (例: features_only=Trueを使うモデル)
            # if self.model_type == 'some_specific_model':
            #     self.feature_extractor = timm.create_model(..., features_only=True)

            # 特徴次元の確認
            # ダミー入力を作成 (バッチサイズ1, 3チャンネル, 指定サイズ)
            dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
            with torch.no_grad():
                # 特徴抽出器の出力を得る
                dummy_output = self.feature_extractor(dummy_input)
                # 出力がタプルやリストの場合、最初の要素を使うことが多い
                if isinstance(dummy_output, (list, tuple)):
                    dummy_output = dummy_output[0]
                # 空間次元が残っている場合 (例: [1, C, H, W]) は Global Pooling が必要
                if len(dummy_output.shape) > 2:
                    # AdaptiveAvgPool2d を使って空間次元を潰す
                    dummy_output = F.adaptive_avg_pool2d(dummy_output, (1, 1))
                # バッチ次元を除いた特徴次元数を取得
                self.feature_dim = dummy_output.view(dummy_output.size(0), -1).shape[1]

            print(f"モデル '{self.model_type}' の特徴次元数: {self.feature_dim}")

        except Exception as e:
            print(f"エラー: モデル '{self.model_type}' の読み込みに失敗しました: {e}")
            print("フォールバックとして ResNet18 を使用します。")
            # フォールバック処理
            self.backbone = torchvision.models.resnet18(pretrained=True)
            self.model_type = "resnet18_fallback"
            # ResNetの場合は最後のFC層を除いた部分を特徴抽出器とする
            modules = list(self.backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*modules)
            # ResNet18の特徴次元は512
            self.feature_dim = 512
            self.use_agc = False # ResNetではAGCは使わない
            print(f"ResNet18 (Fallback) 特徴抽出器出力次元: {self.feature_dim}")

        # --- 特徴抽出器の初期状態設定 (フラグに基づく) ---
        if self.use_progressive_unfreezing:
            # 段階的凍結解除の場合: 初期は凍結
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("特徴抽出器の全パラメータを初期状態で凍結しました (段階的凍結解除モード)。")
        else:
            # ステージ毎差分学習率の場合: 初期から学習可能
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            print("特徴抽出器の全パラメータを初期状態で学習可能に設定しました (差分学習率モード)。")

        # --- 分類ヘッドの定義 ---
        # configからdropout率を取得できるように変更
        dropout1_rate = self.model_config.get("classifier_dropout1", 0.3)
        dropout2_rate = self.model_config.get("classifier_dropout2", 0.2)
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1), # 特徴抽出器側でPoolingしない場合に必要
            # nn.Flatten(), # 特徴抽出器側でFlattenしない場合に必要
            nn.Linear(self.feature_dim, 256),
            nn.SiLU(), # または nn.ReLU()
            nn.Dropout(p=dropout1_rate),
            nn.Linear(256, 128),
            nn.SiLU(), # または nn.ReLU()
            nn.Dropout(p=dropout2_rate),
            nn.Linear(128, self.num_classes)
        )
        print(f"分類ヘッドを定義しました (Dropout1: {dropout1_rate}, Dropout2: {dropout2_rate})")

        # --- 損失関数 ---
        self.ce_loss = nn.CrossEntropyLoss()

        # --- メトリクス ---
        # タスクタイプとクラス数を指定
        task_type = "multiclass"
        self.train_f1 = F1Score(task=task_type, num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task=task_type, num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task=task_type, num_classes=num_classes, average="macro")
        self.train_acc = Accuracy(task=task_type, num_classes=num_classes)
        self.val_acc = Accuracy(task=task_type, num_classes=num_classes)
        self.test_acc = Accuracy(task=task_type, num_classes=num_classes)
        # 検証用混同行列メトリクスを追加
        self.val_cm = ConfusionMatrix(task=task_type, num_classes=num_classes)

        # --- ステージ検出 (凍結解除/差分学習率の両方で必要) ---
        self._setup_stages()
        
    def _setup_stages(self):
        """
        ロードされたバックボーンモデルの構造に基づいて、
        段階的ファインチューニングまたは差分学習率のためのステージを特定する。
        モデルタイプに応じて適切なステージ分割を行う。
        ステージリスト `self.stages` は入力に近い層（浅い層）から出力に近い層（深い層）の順に並べる。
        """
        self.stages = [] # ステージリストを初期化
        print(f"モデルタイプ '{self.model_type}' のステージ構造を設定します...")

        # --- NFNet系のステージ検出 ---
        if "nfnet" in self.model_type:
            try:
                # timmのNFNetモデルは 'stages' という名前のSequentialモジュールを持つことが多い
                if hasattr(self.backbone, 'stages') and isinstance(self.backbone.stages, nn.Sequential):
                    # stagesモジュール内の各サブモジュールをステージとする
                    self.stages = list(self.backbone.stages.children())
                    print(f"  NFNetの 'stages' モジュールから {len(self.stages)} 個のステージを検出しました。")
                else:
                    # 'stages' が見つからない場合の代替検索 (例: stage0, stage1...)
                    potential_stages = []
                    for name, module in self.backbone.named_children():
                        if name.startswith('stage'):
                            potential_stages.append(module)
                    if potential_stages:
                        self.stages = potential_stages
                        print(f"  NFNetの 'stageX' モジュールから {len(self.stages)} 個のステージを検出しました。")
                    else:
                        print("  NFNetのステージ構造を自動検出できませんでした。代替手段を試みます。")
                        self._detect_stages_heuristically() # 汎用的な検出へ
            except Exception as e:
                print(f"  NFNetステージ構造の取得中にエラー: {e}")
                self._detect_stages_heuristically()

        # --- EfficientNet系のステージ検出 ---
        elif "efficientnet" in self.model_type:
            try:
                # timmのEfficientNetモデルは 'blocks' という名前のSequentialモジュールを持つことが多い
                if hasattr(self.backbone, 'blocks') and isinstance(self.backbone.blocks, nn.Sequential):
                    # 'blocks' モジュール内の各サブモジュール（通常はステージに対応）をリスト化
                    self.stages = list(self.backbone.blocks.children())
                    print(f"  EfficientNetの 'blocks' モジュールから {len(self.stages)} 個のステージ (ブロックグループ) を検出しました。")
                else:
                    print("  EfficientNetの 'blocks' モジュールが見つかりません。代替手段を試みます。")
                    self._detect_stages_heuristically() # 汎用的な検出へ
            except Exception as e:
                print(f"  EfficientNetステージ構造の取得中にエラー: {e}")
                self._detect_stages_heuristically()

        # --- ResNet系のステージ検出 (Fallback含む) ---
        elif "resnet" in self.model_type:
            try:
                # ResNet系は通常 layer1, layer2, layer3, layer4 を持つ
                resnet_layers = []
                for name, module in self.feature_extractor.named_children(): # feature_extractorから探す
                    if name.startswith('layer'):
                        resnet_layers.append(module)
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
            print(f"モデルタイプ '{self.model_type}' のためのステージ自動検出を試みます。")
            self._detect_stages_heuristically() # 汎用的な検出を呼び出す

        # --- ステージリストの最終調整 ---
        if not self.stages:
             print("警告: ステージ分割に失敗しました。特徴抽出器全体を単一ステージとして扱います。")
             # 特徴抽出器全体を一つのステージとする
             self.stages = [self.feature_extractor]
        # リバースしない - 入力に近い順（浅い→深い）を維持
        # else:
        #      # ステージリストを逆順にする（深い層 -> 浅い層）
        #      self.stages.reverse()
        print(f"最終的なステージ数: {len(self.stages)} (入力に近い層（浅い）から出力に近い層（深い）の順に格納)")

        # 各ステージのパラメータ数を表示
        print("ステージごとのパラメータ数:")
        for i, stage in enumerate(self.stages):
            # stageがnn.ModuleListの場合も考慮
            if isinstance(stage, nn.ModuleList):
                 params_frozen = sum(p.numel() for submodule in stage for p in submodule.parameters() if not p.requires_grad)
                 total_params = sum(p.numel() for submodule in stage for p in submodule.parameters())
            else:
                 params_frozen = sum(p.numel() for p in stage.parameters() if not p.requires_grad) # 凍結中のパラメータ数
                 total_params = sum(p.numel() for p in stage.parameters())
            status = "凍結中" if params_frozen > 0 else "学習可能"
            print(f"  Stage {i} (入力から第{i+1}層): {total_params - params_frozen:,} / {total_params:,} パラメータ ({status})")


    def _detect_stages_heuristically(self):
        """
        モデル固有のステージ構造が見つからない場合に、
        特徴抽出器のモジュール構造から経験的にステージを分割する代替手段。
        """
        print("  経験的なステージ検出を実行します...")
        # 特徴抽出器内の主要な子モジュール（パラメータを持つもの）を候補とする
        potential_stages = [
            module for module in self.feature_extractor.children() if list(module.parameters())
        ]
        print(f"    特徴抽出器の主要な子モジュール数: {len(potential_stages)}")

        if len(potential_stages) >= 4: # 十分な数の候補があれば採用
            # 主要なモジュールを4つのステージに均等に分割することを試みる
            num_target_stages = 4
            modules_per_stage = len(potential_stages) // num_target_stages
            self.stages = []
            start_idx = 0
            for i in range(num_target_stages):
                end_idx = start_idx + modules_per_stage
                # 最後のステージは残りすべてを含む
                if i == num_target_stages - 1:
                    end_idx = len(potential_stages)
                if start_idx < end_idx: # モジュールがある場合のみ追加
                    # 複数のモジュールを一つのステージとしてまとめる (nn.Sequentialでラップ)
                    self.stages.append(nn.Sequential(*potential_stages[start_idx:end_idx]))
                start_idx = end_idx
            print(f"  特徴抽出器の子モジュールから{len(self.stages)}個のステージを構成しました。")
        elif potential_stages: # 候補が少ない場合は、それらをそのままステージとする
             self.stages = potential_stages # 個々のモジュールをステージとする
             print(f"  特徴抽出器の子モジュールをそのままステージとして使用します ({len(self.stages)}個)。")
        else:
            print("  ステージ分割可能な主要モジュールが見つかりませんでした。")
            # self.stages は空のままとなり、呼び出し元で処理される

    def forward(self, x):
        """フォワードパス"""
        # 特徴抽出器を通して特徴量を取得
        features = self.feature_extractor(x)

        # モデルによっては出力がタプルやリストの場合がある (例: EfficientNetV2)
        if isinstance(features, (list, tuple)):
            features = features[0] # 最初の要素 (通常は最終特徴量) を使用

        # 空間次元が残っている場合 (例: [B, C, H, W]) は Global Pooling
        # timmの多くのモデルは num_classes=0 で [B, C] の出力を返すため、不要なことが多い
        if len(features.shape) > 2:
            # グローバル平均プーリングを適用し、[B, C, 1, 1] にする
            features = F.adaptive_avg_pool2d(features, (1, 1))
            # [B, C] に平坦化
            features = features.view(features.size(0), -1)

        # 分類ヘッドを通して最終的なロジットを得る
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        """学習ステップ処理"""
        x, y = batch
        logits = self.forward(x)
        loss = self.ce_loss(logits, y)

        # メトリクス計算とログ記録
        preds = torch.argmax(logits, dim=1)
        self.train_f1.update(preds, y)
        self.train_acc.update(preds, y)
        
        # ロス値のみをログ - メトリクスはエポック終了時にログする
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        """学習エポック終了時の処理"""
        # F1スコアとAccuracyを計算してログ記録
        train_f1 = self.train_f1.compute()
        train_acc = self.train_acc.compute()
        
        # エポック終了時のメトリクスをログ記録
        self.log('train_f1', train_f1, prog_bar=True)
        self.log('train_acc', train_acc, prog_bar=False)
        
        # メトリクスをリセット (次のエポックのために重要)
        self.train_f1.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        """検証ステップ処理"""
        x, y = batch

        # --- デバッグ用: バッチ内のラベル分布を確認 ---
        unique_labels, counts = np.unique(y.cpu().numpy(), return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        # logger.info(f"Validation Step Batch {batch_idx}: Label distribution = {label_counts}") # 必要に応じて有効化
        if 2 not in label_counts or label_counts[2] == 0:
             # logger.warning(f"警告: Validation Step Batch {batch_idx} にラベル2のデータが含まれていません！") # 必要に応じて有効化
             pass # ログが大量に出る可能性があるので注意
        # --- デバッグ用ログここまで ---

        logits = self.forward(x)
        loss = self.ce_loss(logits, y)

        # メトリクス計算とログ記録
        preds = torch.argmax(logits, dim=1)
        self.val_f1.update(preds, y)
        self.val_acc.update(preds, y)
        # 混同行列を更新
        self.val_cm.update(preds, y)

        # val_loss のログ記録のみ残す
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        x, y = batch
        logits = self.forward(x)
        loss = self.ce_loss(logits, y)

        # メトリクス計算とログ記録
        preds = torch.argmax(logits, dim=1)
        self.test_f1.update(preds, y)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
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
            # 分類ヘッドのパラメータグループ
            {'params': self.classifier.parameters(), 'lr': self.lr_head, 'name': 'classifier'},
        ]

        # バックボーンの各ステージのパラメータグループを追加
        # self.stages は浅い層(入力に近い層)から深い層(出力に近い層)の順
        num_stages = len(self.stages)
        for i, stage in enumerate(self.stages):
            # 浅い層 (i=0) ほど学習率が低く、深い層 (i=num_stages-1) ほど学習率が高い
            decay_power = num_stages - 1 - i  # 浅い層(入力側)ほど減衰が大きい
            stage_lr = self.lr_backbone * (self.lr_decay_rate ** decay_power)

            # stageがnn.ModuleListの場合も考慮
            stage_params = []
            modules_in_stage = stage if isinstance(stage, nn.ModuleList) else [stage]
            for module in modules_in_stage:
                 # 差分学習率モードでは常にTrue, 段階的解除モードでは解除されたもののみTrue
                 stage_params.extend([p for p in module.parameters() if p.requires_grad])

            # 学習可能なパラメータが存在する場合のみグループを追加
            if stage_params:
                 param_groups.append({
                     'params': stage_params,
                     'lr': stage_lr,
                     # ステージ名はインデックス順
                     'name': f'stage_{i}' # 入力側から0,1,2...
                 })
                 print(f"  Optimizer group: stage_{i} (Depth {num_stages - 1 - i}), lr: {stage_lr:.2e}, params: {len(stage_params)}")
            else:
                 # 段階的凍結解除モードの初期段階ではここに来ることがある
                 print(f"  情報: Stage {i} (Depth {num_stages - 1 - i}) に現在学習可能なパラメータがありません。スキップします。")


        # オプティマイザの選択 (AdamWを推奨)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)
        print(f"オプティマイザ: AdamW (weight_decay={self.weight_decay})")

        # AGCの適用 (設定が有効な場合)
        if self.use_agc:
            try:
                # AGCクラスのインスタンス化を試みる
                optimizer = AGC(optimizer,
                                clip_factor=self.agc_clip_factor,
                                eps=self.agc_eps)
                print(f"Adaptive Gradient Clipping (AGC) を有効化しました (clip_factor={self.agc_clip_factor}, eps={self.agc_eps})")
            except Exception as agc_err:
                 print(f"警告: AGCの適用中にエラーが発生しました。AGCなしで続行します。エラー: {agc_err}")
                 # AGCなしの元のオプティマイザに戻す（必要であれば）
                 optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)


        # スケジューラの選択 (Cosine Annealingを推奨)
        scheduler_type = self.hparams.get("scheduler", "cosine") # configから読み込み
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                # T_max は最大エポック数とするのが一般的
                T_max=self.hparams.get("max_epochs", 100), # configからmax_epochsを取得
                # eta_min は学習率の下限値
                eta_min=self.hparams.get("eta_min", 1e-6) # configからeta_minを取得
            )
            print(f"CosineAnnealingLRスケジューラを使用します (T_max={self.hparams.get('max_epochs', 100)}, eta_min={self.hparams.get('eta_min', 1e-6)})")
        elif scheduler_type == "step":
             scheduler = torch.optim.lr_scheduler.StepLR(
                 optimizer,
                 step_size=self.hparams.get("scheduler_step_size", 30), # configからstep_sizeを取得
                 gamma=self.hparams.get("scheduler_gamma", 0.1) # configからgammaを取得
             )
             print(f"StepLRスケジューラを使用します (step_size={self.hparams.get('scheduler_step_size', 30)}, gamma={self.hparams.get('scheduler_gamma', 0.1)})")
        else:
             print(f"警告: 未知のスケジューラタイプ '{scheduler_type}'。スケジューラは使用されません。")
             return [optimizer] # スケジューラなしでオプティマイザのみ返す

        # オプティマイザとスケジューラをリストで返す
        # スケジューラの設定を辞書形式で返すのが推奨
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}]


    # --- 段階的ファインチューニングのための凍結解除メソッド ---
    # これらのメソッドは use_progressive_unfreezing = True の場合にコールバックから呼ばれる

    def unfreeze_stage(self, stage_index_from_deep):
        """
        指定されたインデックス（出力に近い深い方から0）のバックボーンステージを凍結解除する。
        配列内では、入力に近い層（浅い層）から出力に近い層（深い層）の順に格納されているので、
        インデックスを反転させる必要がある。
        """
        # インデックスを反転 (深い層のインデックス → 配列内の実際のインデックス)
        actual_index = len(self.stages) - 1 - stage_index_from_deep
        if 0 <= actual_index < len(self.stages):
            stage_to_unfreeze = self.stages[actual_index]
            unfrozen_params_count = 0
            total_params_count = 0

            modules_to_unfreeze = stage_to_unfreeze if isinstance(stage_to_unfreeze, nn.ModuleList) else [stage_to_unfreeze]

            for module in modules_to_unfreeze:
                 for param in module.parameters():
                     if not param.requires_grad: # まだ凍結されているパラメータのみを対象
                          param.requires_grad = True
                          unfrozen_params_count += param.numel()
                     total_params_count += param.numel() # 総パラメータ数は常にカウント

            if unfrozen_params_count > 0:
                 print(f"Backbone Stage {actual_index} (出力から第{stage_index_from_deep+1}層) の凍結解除を実行。{unfrozen_params_count:,} 個のパラメータが新たに学習可能になりました (ステージ総数: {total_params_count:,})。")
            else:
                 print(f"Backbone Stage {actual_index} (出力から第{stage_index_from_deep+1}層) は既に凍結解除済みか、パラメータがありません。")
        else:
            print(f"警告: 指定されたバックボーンステージインデックス {stage_index_from_deep} は無効です (有効範囲: 0-{len(self.stages)-1})。")

    # コールバックから呼び出されることを想定したメソッド
    # モデルタイプによって層の名前が違うため、より汎用的な名前に変更
    def unfreeze_layer4(self): # 最も深いステージ (インデックス 0)
        print("Attempting to unfreeze Layer 4 (deepest stage)...")
        self.unfreeze_stage(0)

    def unfreeze_layer3(self): # 2番目に深いステージ (インデックス 1)
        print("Attempting to unfreeze Layer 3...")
        self.unfreeze_stage(1)

    def unfreeze_layer2(self): # 3番目に深いステージ (インデックス 2)
        print("Attempting to unfreeze Layer 2...")
        self.unfreeze_stage(2)

    # Layer 1 (最も浅いステージ) は通常、凍結解除の対象外とするか、
    # 非常に遅い段階で解除することが多い


# --- main.py でのモデルクラスの切り替えイメージ ---
# import argparse
# from src.models.single_modal import SingleModalClassifier
# from src.models.multi_modal import MultiModalClassifier # 仮
# from src.datamodules.single_modal import SingleDataModule # 仮
# from src.datamodules.multi_modal import MultiDataModule # 仮

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
#     args = parser.parse_args()
#     config = load_config(args.config)

#     model_type = config.get("model", {}).get("type", "unknown")
#     data_mode = config.get("data", {}).get("mode", "single") # configにdata.modeを追加想定

#     # データモジュールの選択
#     if data_mode == "multi":
#         data_module = MultiDataModule(config)
#         print("マルチモーダル用データモジュールを使用します。")
#     else: # デフォルトはシングルモーダル
#         data_module = SingleDataModule(config)
#         print("シングルモーダル用データモジュールを使用します。")

#     # モデルの選択
#     if data_mode == "multi":
#         # マルチモーダルモデルをインスタンス化 (仮のクラス名)
#         model = MultiModalClassifier(config)
#         print(f"マルチモーダルモデル ({model_type}) を使用します。")
#     else:
#         # シングルモーダルモデルをインスタンス化
#         model = SingleModalClassifier(config)
#         print(f"シングルモーダルモデル ({model_type}) を使用します。")

#     # ... (Trainerの設定と実行) ...