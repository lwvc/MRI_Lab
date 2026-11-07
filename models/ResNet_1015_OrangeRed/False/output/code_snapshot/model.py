from __future__ import annotations

import torch
from torch import nn


EXPECTED_IMAGE_SHAPE = (193, 229, 193)
FLATTENED_FEATURES = 16384
METADATA_WIDTH = {
    "none": 0,
    "sex": 1,
    "scanner": 1,
    "both": 2,
}


class ResBlock(nn.Module):
    """Legacy residual block matching the senior model."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, padding="same")
        # This intentionally matches the senior code: one BatchNorm module is
        # reused after both convolutions, with batch statistics used in eval.
        # self.batch_norm = nn.BatchNorm3d(
        #     out_channels,
        #     affine=True,
        #     track_running_stats=True, 
        # )
        self.bn1 = nn.BatchNorm3d(
            out_channels,
            track_running_stats=False
        )

        self.bn2 = nn.BatchNorm3d(
            out_channels,
            track_running_stats=True
        ) # 改 BN不同物件
        self.activation = nn.ELU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, padding="same")
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(inputs)
        features = self.conv1(inputs)
        features = self.bn1(features)  # 改 BN不同物件
        features = self.activation(features)
        features = self.conv2(features)
        features = self.bn2(features)  # 改 BN不同物件
        return self.activation(features + shortcut)


class ResNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblock1 = ResBlock(1, 8)
        self.resblock2 = ResBlock(8, 16)
        self.resblock3 = ResBlock(16, 32)
        self.resblock4 = ResBlock(32, 64)
        self.resblock5 = ResBlock(64, 128)
        self.resblock6 = ResBlock(128, 256)
        self.maxpool1 = nn.MaxPool3d(2, 2, 1, ceil_mode=False)
        self.maxpool2 = nn.MaxPool3d(2, 2, (1, 0, 1), ceil_mode=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.maxpool1(self.resblock1(inputs))
        features = self.maxpool1(self.resblock2(features))
        features = self.maxpool2(self.resblock3(features))
        features = self.maxpool1(self.resblock4(features))
        features = self.maxpool1(self.resblock5(features))
        return self.maxpool2(self.resblock6(features))


class BrainAgeResNet(nn.Module):
    def __init__(self, metadata: str = "both", dropout: float = 0.2) -> None:
        super().__init__()
        if metadata not in METADATA_WIDTH:
            raise ValueError(
                f"metadata must be one of {tuple(METADATA_WIDTH)}, got {metadata}"
            )
        self.metadata = metadata
        self.backbone = ResNetBackbone()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(FLATTENED_FEATURES, 256)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)
        self.output = nn.Linear(256 + METADATA_WIDTH[metadata], 1)

    def _metadata_tensor(
        self,
        sex: torch.Tensor,
        scanner: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.metadata == "none":
            return None
        if self.metadata == "sex":
            return sex
        if self.metadata == "scanner":
            return scanner
        return torch.cat((sex, scanner), dim=1)

    def forward(
        self,
        image: torch.Tensor,
        sex: torch.Tensor,
        scanner: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(image.shape[2:]) != EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Expected input shape (N, 1, {EXPECTED_IMAGE_SHAPE}), "
                f"got {tuple(image.shape)}"
            )

        features = self.backbone(image)
        features = self.flatten(features)
        if features.shape[1] != FLATTENED_FEATURES:
            raise RuntimeError(
                f"Expected {FLATTENED_FEATURES} flattened features, "
                f"got {features.shape[1]}"
            )
        features = self.dropout(self.activation(self.fc1(features)))

        metadata = self._metadata_tensor(sex, scanner)
        if metadata is not None:
            features = torch.cat((features, metadata), dim=1)
        return self.output(features)
