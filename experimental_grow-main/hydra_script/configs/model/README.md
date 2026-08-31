# Model Configuration (`model/`)

Defines the neural network architecture.

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | **required** | Model family identifier. See supported values below. |
| `activation` | str | `relu` | Activation function (`relu`, `leaky_relu`). |

## Supported Model Families

| `model` value | Class | Description |
|---------------|-------|-------------|
| `resnet` | `init_full_resnet_structure` | ResNet with growing support. Main architecture for growth experiments. |
| `true_resnet` | `get_resnet` | Standard torchvision-style ResNet (no growing support). |
| `perceptron` | `Perceptron` | Single-layer perceptron. |
| `mlp` | `GrowingMLP` | Multi-layer perceptron with growing support. |
| `residual_mlp` | `GrowingResidualMLP` | Residual MLP with growing support. |
| `mlp_mixer` | `GrowingMLPMixer` | MLP-Mixer with growing support. |
| `growing_transformer` | `GrowingTransformer` | Vision Transformer with growable feed-forward transformer blocks. |
| `growing_cct` | `GrowingCCT` | Compact Convolutional Transformer with growable feed-forward transformer blocks. |
| `growing_cvt` | `GrowingCVT` | Compact Vision Transformer with growable feed-forward transformer blocks. |
| `growing_vit_lite` | `GrowingViTLite` | ViT-Lite with growable feed-forward transformer blocks. |
| `eeg_model` | `EEGModel` | Specialized CNN for EEG data. |

## ResNet-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `number_of_blocks_per_stage` | int | varies | Number of residual blocks per stage. 1 = ResNet10-like, 2 = ResNet18-like. |
| `nb_stages` | int | (default from builder) | Number of residual stages. |
| `inplanes` | int | (default from builder) | Number of channels after the initial convolution. |
| `reduction_factor` | float | (default from builder) | Multiplier for channel widths. `1` = full width; values < 1 reduce the model size. |
| `normalization` | str | (default from builder) | Normalization layer type (e.g., `group` for GroupNorm). |
| `hidden_channels` | list | (auto) |  Explicit per-stage channel counts. Used in some sweep configs (e.g., `re-run.yaml`), overriding the default channel progression. Exact format depends on the model builder. |

## EEG Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_last_layer_size` | int | `256` | Size of the last fully-connected layer. |
| `activation` | str | `leaky_relu` | Activation function. |

## Growing Transformer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patch_size` | int or tuple | `4` | Image patch size. For CIFAR-100, `4` creates an 8x8 patch grid. |
| `d_model` | int | `128` | Transformer embedding width. |
| `num_heads` | int | `4` | Attention heads. Must divide `d_model`. |
| `d_ff` | int | `256` | Feed-forward hidden width; this is the growable branch. |
| `num_blocks` | int | `6` | Number of transformer blocks and growth positions. |
| `dropout` | float | `0.1` | Dropout probability. |
| `pooling` | str | `cls` | Classification pooling mode: `cls` or `mean`. |
| `use_cls_token` | bool | `true` | Whether to prepend a CLS token. Must be `true` for `pooling=cls`. |

## Growing CCT/CVT/ViT-Lite Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | varies | Token embedding width. |
| `num_layers` | int | varies | Transformer depth and number of growth positions. |
| `num_heads` | int | varies | Attention heads. Must divide `embedding_dim`. |
| `mlp_ratio` | float | varies | Feed-forward hidden width multiplier; this branch is growable. |
| `kernel_size` | int | varies | Tokenizer convolution or patch kernel size. |
| `n_conv_layers` | int | `1` | CCT tokenizer convolution depth. |
| `positional_embedding` | str | `learnable` | Positional embedding type: `learnable`, `sine`, or `none`. |
| `seq_pool` | bool | `true` for CCT/CVT | Whether to use sequence pooling instead of a CLS token. |

## Available Configs

| Config | Family | Description |
|--------|--------|-------------|
| `resnet_mini.yaml` | resnet | Small ResNet (1 block/stage, 3 stages, 16 inplanes) |
| `resnet10.yaml` | resnet | ResNet-10 (1 block/stage) |
| `resnet10_full.yaml` | resnet | ResNet-10 full width (reduction_factor=1) |
| `resnet18.yaml` | resnet | ResNet-18 (2 blocks/stage) |
| `resnet18_full.yaml` | resnet | ResNet-18 full width (reduction_factor=1) |
| `growing_vit.yaml` | growing_transformer | Small Vision Transformer for CIFAR-style images |
| `growing_cct_7_3x1_32.yaml` | growing_cct | CCT-7/3x1/32 architecture from Compact-Transformers |
| `eeg_cnn.yaml` | eeg_model | EEG CNN (256 last layer) |

## Adding a New Model

See the [main CONFIGURATION.md](../../CONFIGURATION.md#how-to-add-a-new-model) for step-by-step instructions.
