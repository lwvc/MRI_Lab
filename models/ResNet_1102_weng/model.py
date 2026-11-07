"""3D residual CNN for brain-age regression (temporary standard BN version)."""
from __future__ import annotations

import math

import torch
from torch import nn

from config import CFG, Config


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cfg: Config):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels, eps=cfg.norm_eps,
                                  momentum=cfg.norm_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels, eps=cfg.norm_eps,
                                  momentum=cfg.norm_momentum)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ELU(alpha=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class BrainAgeResNet(nn.Module):
    """Input: MRI [N,1,D,H,W] and optional sex/scanner [N,1]."""

    def __init__(self, cfg: Config = CFG):
        super().__init__()
        if cfg.norm_type != "bn":
            raise ValueError("This model currently supports norm_type='bn' only")
        if len(cfg.filters) != 6 or any(c <= 0 for c in cfg.filters):
            raise ValueError("Expected six positive channel counts")
        if cfg.metadata not in ("none", "sex", "scanner", "both"):
            raise ValueError("Invalid metadata selection")

        self.cfg = cfg
        channels = (1, *cfg.filters)
        self.blocks = nn.ModuleList(
            ResBlock(channels[i], channels[i + 1], cfg)
            for i in range(len(cfg.filters))
        )
        # For kernel=2/stride=2, ceil_mode reproduces Keras SAME pooling.
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.flatten = nn.Flatten()

        spatial = cfg.input_shape
        for _ in cfg.filters:
            spatial = tuple(math.ceil(v / 2) for v in spatial)
        in_features = cfg.filters[-1] * math.prod(spatial)
        metadata_width = {"none": 0, "sex": 1, "scanner": 1, "both": 2}[cfg.metadata]

        self.fc1 = nn.Linear(in_features, cfg.fc_units)
        self.activation = nn.ELU(alpha=1.0)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.output = nn.Linear(cfg.fc_units + metadata_width, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Match Keras: He uniform for convolution and first Dense;
        # Glorot uniform for the final Dense; all biases are zero.
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)

    def regularized_weights(self):
        """Keras kernel_regularizer targets (excludes shortcuts, BN and biases)."""
        for block in self.blocks:
            yield block.conv1.weight
            yield block.conv2.weight
        yield self.fc1.weight
        yield self.output.weight

    def forward(
        self,
        image: torch.Tensor,
        sex: torch.Tensor | None = None,
        scanner: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1 or tuple(image.shape[2:]) != self.cfg.input_shape:
            raise ValueError(f"Expected MRI shape [N, 1, {self.cfg.input_shape}]")

        x = image
        for block in self.blocks:
            x = self.pool(block(x))
        x = self.dropout(self.activation(self.fc1(self.flatten(x))))

        # Keep the previous PyTorch feature/sex/scanner order for new training.
        metadata = []
        for name, value in (("sex", sex), ("scanner", scanner)):
            if self.cfg.metadata in (name, "both"):
                if value is None:
                    raise ValueError(f"Missing {name} input")
                value = value.reshape(-1, 1)
                if value.shape[0] != image.shape[0]:
                    raise ValueError(f"{name} batch size does not match MRI")
                metadata.append(value.to(device=x.device, dtype=x.dtype))
        if metadata:
            x = torch.cat((x, *metadata), dim=1)
        return self.output(x)
