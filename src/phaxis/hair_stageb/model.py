"""Multi-head U-Net used by the locked RHAxiscc Stage B checkpoints."""

from __future__ import annotations

import timm
import torch
from torch import nn
import torch.nn.functional as functional


HEADS = {
    "base_hm": 1,
    "base_off": 2,
    "base_dir": 2,
    "base_len": 1,
    "tip_hm": 1,
    "tip_off": 2,
    "line": 1,
    "flow": 2,
    "root": 1,
}


class ConvBNAct(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, kernel: int = 3):
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel,
                padding=kernel // 2,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int):
        super().__init__()
        # Keep the original c1/c2 attribute names: they are part of the
        # checkpoint state_dict contract, even though an anonymous Sequential
        # would be functionally equivalent.
        self.c1 = ConvBNAct(input_channels + skip_channels, output_channels)
        self.c2 = ConvBNAct(output_channels, output_channels)

    def forward(self, tensor: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        tensor = functional.interpolate(tensor, scale_factor=2.0, mode="nearest")
        if skip is not None:
            if tensor.shape[-2:] != skip.shape[-2:]:
                tensor = functional.interpolate(tensor, size=skip.shape[-2:], mode="nearest")
            tensor = torch.cat([tensor, skip], dim=1)
        return self.c2(self.c1(tensor))


class DilatedContext(nn.Module):
    def __init__(self, channels: int, rates: tuple[int, ...] = (1, 3, 6, 9)):
        super().__init__()
        middle = max(64, channels // 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        middle,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(middle),
                    nn.SiLU(inplace=True),
                )
                for rate in rates
            ]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, middle, 1, bias=False),
            nn.BatchNorm2d(middle),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(middle * (len(rates) + 1) + channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        global_context = self.pool(tensor).expand(-1, -1, tensor.shape[-2], tensor.shape[-1])
        return self.fuse(
            torch.cat([*(branch(tensor) for branch in self.branches), global_context, tensor], dim=1)
        )


class Head(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, bias: float | None):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(input_channels, 64), nn.Conv2d(64, output_channels, 1)
        )
        if bias is not None:
            nn.init.constant_(self.body[-1].bias, bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.body(tensor)


class MultiHeadUNet(nn.Module):
    HEATMAP_HEADS = ("base_hm", "tip_hm", "line", "root")

    def __init__(
        self,
        heads: dict[str, int] = HEADS,
        *,
        encoder: str = "resnet34",
        in_channels: int = 3,
        out_stride: int = 2,
        decoder_channels: tuple[int, ...] = (256, 128, 96, 64),
        pretrained: bool = False,
        context: bool = True,
        stem_stride1: bool = False,
    ):
        super().__init__()
        self.out_stride = out_stride
        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, features_only=True, in_chans=in_channels
        )
        encoder_channels = self.encoder.feature_info.channels()
        reductions = self.encoder.feature_info.reduction()
        self.context = DilatedContext(encoder_channels[-1]) if context else nn.Identity()
        decoder: list[nn.Module] = []
        current_channels = encoder_channels[-1]
        skips = list(encoder_channels[:-1])[::-1]
        skip_reductions = list(reductions[:-1])[::-1]
        widths = list(decoder_channels)
        while len(widths) < len(skips):
            widths.append(widths[-1])
        for index, (skip_channels, reduction) in enumerate(zip(skips, skip_reductions)):
            decoder.append(DecoderBlock(current_channels, skip_channels, widths[index]))
            current_channels = widths[index]
            if reduction <= out_stride:
                break
        self.decoder = nn.ModuleList(decoder)
        self.final_stride = skip_reductions[len(decoder) - 1] if decoder else reductions[-1]
        self.refine = nn.Sequential(
            ConvBNAct(current_channels, current_channels),
            ConvBNAct(current_channels, current_channels),
        )
        self.stem = (
            nn.Sequential(ConvBNAct(in_channels, 32), ConvBNAct(32, 32))
            if stem_stride1
            else None
        )
        head_channels = current_channels + (32 if self.stem is not None else 0)
        self.heads = nn.ModuleDict(
            {
                name: Head(
                    head_channels,
                    channels,
                    -4.0 if name in self.HEATMAP_HEADS else None,
                )
                for name, channels in heads.items()
            }
        )

    def forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(tensor)
        output = self.context(features[-1])
        skips = features[:-1][::-1]
        for index, block in enumerate(self.decoder):
            output = block(output, skips[index] if index < len(skips) else None)
        output = self.refine(output)
        if self.final_stride != self.out_stride:
            output = functional.interpolate(
                output,
                scale_factor=self.final_stride / self.out_stride,
                mode="bilinear",
                align_corners=False,
            )
        if self.stem is not None:
            stem = self.stem(tensor)
            if stem.shape[-2:] != output.shape[-2:]:
                stem = functional.interpolate(
                    stem, size=output.shape[-2:], mode="bilinear", align_corners=False
                )
            output = torch.cat([output, stem], dim=1)
        return {name: head(output) for name, head in self.heads.items()}
