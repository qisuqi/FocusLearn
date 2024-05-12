import torch

from AttnAM_PyTorch.config.base import Config


def defaults() -> Config:
    config = Config(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=2021,

        # Data Path
        name="NAM",
        regression=True,

        # training
        num_epochs=10,
        batch_size=128,
        lr=5e-4,

        # Feature selection
        top_features=10,

        # RNN
        encoder_input_dim=1,
        encoder_hidden_unit=32,
        encoder_num_layers=1,
        encoder_dropout=0.5,

        # Attention
        num_attn_heads=1,
        attn_dropout=0.5,

        # Hidden size for featureNN
        feature_hidden_unit=[64, 32],
        feature_dropout=0.1,
        # Activation choice for NAM
        activation='exu',  # Either `ExU` or `Relu`

        # Output
        output_dropout=0.1,

        # Optimiser/learning rate
        warm_up=20,
        decay_rate=0,

        # Num units for FeatureNN
        num_basis_functions=10000,
        units_multiplier=2,
        shuffle=True,

        # for dataloaders
        num_workers=0,

        # for classification
        alpha=1,
        gamma=2,
        pos_weight=2,
        resample=True,
    )

    return config
