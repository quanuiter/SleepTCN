"""Kiến trúc chuẩn hóa cho E0–E3. Mọi forward trả logits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(16, stride=16, ceil_mode=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SleepCNN(nn.Module):
    """Một trong 15 CNN của ZleepAnlystNet, 2.495 tham số."""

    def __init__(self, n_classes: int = 5) -> None:
        super().__init__()
        self.cnn1 = ConvBlock(1, 10, kernel_size=55)
        self.cnn2 = ConvBlock(10, 5, kernel_size=25)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, n_classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn2(self.cnn1(x)))

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


class BiLSTMSleepNet(nn.Module):
    def __init__(
        self, input_dim: int = 75, hidden_dim: int = 128, n_classes: int = 5
    ) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                size = parameter.size(0)
                parameter.data[size // 4 : size // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or lengths.ndim != 1 or len(lengths) != len(x):
            raise ValueError("Expected x=(B,T,F), lengths=(B,)")
        if torch.any(lengths <= 0) or torch.any(lengths > x.shape[1]):
            raise ValueError("Invalid sequence length")
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.bilstm(packed)
        output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.shape[1]
        )
        logits = self.classifier(output)
        padding_mask = (
            torch.arange(x.shape[1], device=x.device)[None, :]
            >= lengths.to(x.device)[:, None]
        )
        return logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class ResNetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, stride, kernel_size) <= 0:
            raise ValueError("ResNet block dimensions must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("ResNet block kernel_size must be odd")
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


class EEGResNet1D(nn.Module):
    """Configurable one-dimensional ResNet extractor/classifier.

    The default configuration is byte-for-byte equivalent in architecture to
    the v2 model.  Explicit configuration is required for tuning so that the
    resolved architecture cannot silently differ from the experiment JSON.
    """

    DEFAULT_CONFIG: dict[str, object] = {
        "input_channels": 1,
        "stem": {
            "channels": 32,
            "kernel_size": 50,
            "stride": 5,
            "padding": 25,
            "max_pool_kernel": 3,
            "max_pool_stride": 2,
            "max_pool_padding": 1,
        },
        "residual_blocks": [
            {"in_channels": 32, "out_channels": 64, "stride": 1, "kernel_size": 7},
            {"in_channels": 64, "out_channels": 128, "stride": 2, "kernel_size": 7},
            {"in_channels": 128, "out_channels": 128, "stride": 2, "kernel_size": 7},
        ],
        "feature_dim": 128,
        "classifier_dropout": 0.5,
    }

    def __init__(
        self,
        feature_dim: int = 128,
        n_classes: int = 5,
        *,
        config: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            resolved = deepcopy(self.DEFAULT_CONFIG)
            resolved["feature_dim"] = feature_dim
        else:
            resolved = deepcopy(dict(config))
        self.resolved_config = self._validate_config(resolved)
        input_channels = int(self.resolved_config["input_channels"])
        stem = self.resolved_config["stem"]
        assert isinstance(stem, Mapping)
        stem_channels = int(stem["channels"])
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                stem_channels,
                int(stem["kernel_size"]),
                stride=int(stem["stride"]),
                padding=int(stem["padding"]),
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                int(stem["max_pool_kernel"]),
                stride=int(stem["max_pool_stride"]),
                padding=int(stem["max_pool_padding"]),
            ),
        )
        blocks = self.resolved_config["residual_blocks"]
        assert isinstance(blocks, Sequence)
        for index, block in enumerate(blocks, start=1):
            setattr(
                self,
                f"layer{index}",
                ResNetBlock1D(
                    int(block["in_channels"]),
                    int(block["out_channels"]),
                    stride=int(block["stride"]),
                    kernel_size=int(block["kernel_size"]),
                ),
            )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(float(self.resolved_config["classifier_dropout"]))
        self.classifier = nn.Linear(int(self.resolved_config["feature_dim"]), n_classes)

    @classmethod
    def from_config(
        cls, config: Mapping[str, object], n_classes: int = 5
    ) -> "EEGResNet1D":
        return cls(config=config, n_classes=n_classes)

    @property
    def residual_layers(self) -> tuple[nn.Module, ...]:
        blocks = self.resolved_config["residual_blocks"]
        assert isinstance(blocks, Sequence)
        return tuple(
            getattr(self, f"layer{index}")
            for index in range(1, len(blocks) + 1)
        )

    @classmethod
    def _validate_config(cls, config: dict[str, object]) -> dict[str, object]:
        required = {
            "input_channels",
            "stem",
            "residual_blocks",
            "feature_dim",
            "classifier_dropout",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(f"ResNet config is missing fields: {missing}")
        if int(config["input_channels"]) != 1:
            raise ValueError("SleepTCN EEG extractor currently requires one input channel")
        if int(config["feature_dim"]) <= 0:
            raise ValueError("feature_dim must be positive")
        dropout = float(config["classifier_dropout"])
        if not 0.0 <= dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1)")
        stem = config["stem"]
        if not isinstance(stem, Mapping):
            raise ValueError("stem must be a mapping")
        for field in (
            "channels",
            "kernel_size",
            "stride",
            "max_pool_kernel",
            "max_pool_stride",
        ):
            if field not in stem or int(stem[field]) <= 0:
                raise ValueError(f"stem.{field} must be positive")
        for field in ("padding", "max_pool_padding"):
            if field not in stem or int(stem[field]) < 0:
                raise ValueError(f"stem.{field} must be nonnegative")
        blocks = config["residual_blocks"]
        if not isinstance(blocks, Sequence) or not blocks:
            raise ValueError("residual_blocks must be a non-empty sequence")
        previous_channels = int(stem["channels"])
        normalized_blocks: list[dict[str, int]] = []
        for index, block in enumerate(blocks):
            if not isinstance(block, Mapping):
                raise ValueError(f"residual_blocks[{index}] must be a mapping")
            values = {
                field: int(block[field])
                for field in ("in_channels", "out_channels", "stride", "kernel_size")
                if field in block
            }
            if set(values) != {"in_channels", "out_channels", "stride", "kernel_size"}:
                raise ValueError(f"residual_blocks[{index}] is incomplete")
            if values["in_channels"] != previous_channels:
                raise ValueError(
                    f"residual_blocks[{index}].in_channels does not match previous output"
                )
            if min(values.values()) <= 0 or values["kernel_size"] % 2 == 0:
                raise ValueError(f"invalid residual_blocks[{index}] dimensions")
            normalized_blocks.append(values)
            previous_channels = values["out_channels"]
        if previous_channels != int(config["feature_dim"]):
            raise ValueError("last residual block output must equal feature_dim")
        config["input_channels"] = int(config["input_channels"])
        config["feature_dim"] = int(config["feature_dim"])
        config["classifier_dropout"] = dropout
        config["stem"] = {field: int(stem[field]) for field in stem}
        config["residual_blocks"] = normalized_blocks
        return config

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        output = self.stem(x)
        for layer in self.residual_layers:
            output = layer(output)
        return self.pool(output).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.extract_features(x)))


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("TCN kernel_size must be odd to preserve sequence length")
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool or padding_mask.shape != (
                x.shape[0],
                1,
                x.shape[2],
            ):
                raise ValueError("TemporalBlock padding_mask must have shape (B,1,T)")
        residual = self.residual(x)
        if padding_mask is not None:
            residual = residual.masked_fill(padding_mask, 0.0)
        output = self.conv1(x)
        output = self.norm1(output.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(self.activation(output))
        if padding_mask is not None:
            output = output.masked_fill(padding_mask, 0.0)
        output = self.conv2(output)
        output = self.norm2(output.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(self.activation(output))
        if padding_mask is not None:
            output = output.masked_fill(padding_mask, 0.0)
        result = output + residual
        return result if padding_mask is None else result.masked_fill(padding_mask, 0.0)


class SleepTCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        n_blocks: int = 6,
        dropout: float = 0.2,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
                for index in range(n_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @property
    def receptive_field(self) -> int:
        kernel_size = self.blocks[0].conv1.kernel_size[0]
        return 1 + 2 * (kernel_size - 1) * sum(
            block.conv1.dilation[0] for block in self.blocks
        )

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x=(B,T,F)")
        if padding_mask is not None:
            if padding_mask.dtype != torch.bool or padding_mask.shape != x.shape[:2]:
                raise ValueError("padding_mask must be bool with shape (B,T)")
        output = self.input_projection(x)
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        output = output.transpose(1, 2)
        channel_mask = None if padding_mask is None else padding_mask.unsqueeze(1)
        for block in self.blocks:
            output = block(output, channel_mask)
        output = output.transpose(1, 2)
        logits = self.classifier(self.dropout(output))
        if padding_mask is not None:
            logits = logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return logits
