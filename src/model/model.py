import torch
import torch.nn as nn
from typing import Tuple, Optional

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: Tuple[int, int],
        pool: Optional[Tuple[int, int]] = None,
        padding: str = "same",
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool) if pool and pool != (1, 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale

class ResidualBlock(nn.Module):
    """Residual block with two convolutions, optional SE, and dropout."""
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: Tuple[int, int],
        pool: Optional[Tuple[int, int]] = None,
        padding: str = "same",
        use_batchnorm: bool = True,
        dropout: float = 0.0,
        use_se: bool = False,
    ):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, kernel, pool=None, padding=padding,
                               use_batchnorm=use_batchnorm)
        self.conv2 = ConvBlock(out_ch, out_ch, kernel, pool=None, padding=padding,
                               use_batchnorm=use_batchnorm)
        self.pool = nn.MaxPool2d(kernel_size=pool) if pool and pool != (1, 1) else nn.Identity()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        out = self.se(out)
        out = self.dropout(out)
        out = self.pool(out)
        return out

class ChannelTemporalAttention(nn.Module):
    def __init__(self, feature_channels: int, num_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(feature_channels // reduction, 4)
        self.channel_pool = nn.AdaptiveAvgPool2d((num_channels, 1))
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_mlp = nn.Sequential(
            nn.Conv2d(feature_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.combine = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        B, C, H, T = x.shape
        channel_logits = self.channel_mlp(self.channel_pool(x))
        temporal_logits = self.temporal_mlp(x.mean(dim=2, keepdim=True))
        channel_map = channel_logits.expand(-1, -1, -1, T)
        temporal_map = temporal_logits.expand(-1, -1, H, -1)
        combined = torch.cat([channel_map, temporal_map], dim=1)
        attention = self.combine(combined)
        return x * (1.0 + attention), attention, torch.sigmoid(channel_logits), torch.sigmoid(temporal_logits)

class ChannelDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full((x.size(0), 1, x.size(2), 1), keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob

class SpeechNet(nn.Module):
    """
    Args:
        num_channels (int): number of EMG channels.
        num_classes (int): number of output classes.
        use_batchnorm (bool): if True, use BatchNorm in conv layers.
        use_attention (bool): if True, apply channel-temporal attention.
        channel_dropout (float): dropout probability for channel-wise dropout.
        dropout (float): dropout probability for spatial dropout (after residual blocks, spatial convs).
        width_factor (float): multiplier for all filter counts.
        n_res_blocks (int): number of residual blocks (default 2).
        use_se (bool): if True, add SE blocks after each residual block.
    """
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        use_batchnorm: bool = True,
        use_attention: bool = True,
        channel_dropout: float = 0.2,
        dropout: float = 0.3,
        width_factor: float = 1.0,
        n_res_blocks: int = 2,
        use_se: bool = False,
    ):
        super().__init__()

        # Base channel counts (scaled by width_factor)
        base_stem = int(32 * width_factor)
        base_res = [int(64 * width_factor), int(128 * width_factor), int(256 * width_factor), int(256 * width_factor)]
        # If n_res_blocks > 2, we need more channel levels; we'll reuse the last one.
        # For simplicity, we use a list of channels for each block.
        # We'll create a list of (in_ch, out_ch) pairs for residual blocks.
        # The first block's in_ch = base_stem, then subsequent blocks use the previous out_ch.
        res_channels = []
        for i in range(n_res_blocks):
            if i == 0:
                in_ch = base_stem
                out_ch = base_res[0]
            else:
                in_ch = base_res[i-1] if i-1 < len(base_res) else base_res[-1]
                out_ch = base_res[i] if i < len(base_res) else base_res[-1]
            res_channels.append((in_ch, out_ch))

        # Pooling sizes for each block: we want total temporal downsampling = 32.
        # With n_res_blocks, we distribute the downsampling.
        # Original: res1 pool=(1,4), res2 pool=(1,2) => 4*2=8, plus stem 4 => 32.
        # If more blocks, we can split: e.g., 3 blocks: 4,2,2 = 16? Actually we want product=8 after stem.
        # We'll set pool sizes: first block (1,4), second (1,2), rest (1,1) if any.
        pool_sizes = [(1,4)] + [(1,2)] + [(1,1)] * (n_res_blocks - 2)
        # Ensure we have exactly n_res_blocks pool sizes.
        if len(pool_sizes) < n_res_blocks:
            pool_sizes += [(1,1)] * (n_res_blocks - len(pool_sizes))
        pool_sizes = pool_sizes[:n_res_blocks]

        # Stem
        self.stem = ConvBlock(1, base_stem, kernel=(1, 7), pool=(1, 4),
                               use_batchnorm=use_batchnorm)

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i, ((in_ch, out_ch), pool) in enumerate(zip(res_channels, pool_sizes)):
            self.res_blocks.append(
                ResidualBlock(in_ch, out_ch, kernel=(1, 5), pool=pool,
                              use_batchnorm=use_batchnorm, dropout=dropout,
                              use_se=use_se)
            )

        # After all residual blocks, the channel dimension is the out_ch of the last block.
        last_out = res_channels[-1][1]

        # Attention (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelTemporalAttention(feature_channels=last_out,
                                                      num_channels=num_channels)
        else:
            self.attention = None

        self.channel_dropout = ChannelDropout(p=channel_dropout)

        spatial_in = last_out
        spatial_out = int(256 * width_factor)
        if num_channels >= 14:
            k1, k2 = (7,1), (5,1)
        elif num_channels >= 8:
            k1, k2 = (5,1), (3,1)
        else:
            k1, k2 = (3,1), (3,1)

        self.spatial1 = ConvBlock(spatial_in, spatial_out, kernel=k1, pool=(1, 1),
                                   padding="valid", use_batchnorm=use_batchnorm)
        self.spatial2 = ConvBlock(spatial_out, spatial_out, kernel=k2, pool=(1, 1),
                                   padding="valid", use_batchnorm=use_batchnorm)

        self.spatial_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(spatial_out, num_classes)

    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_features: bool = False,
    ):
        x = self.stem(x)

        for block in self.res_blocks:
            x = block(x)

        attention = channel_att = temporal_att = None

        if self.attention is not None:
            x, attention, channel_att, temporal_att = self.attention(x)

        x = self.channel_dropout(x)
        x = self.spatial1(x)
        x = self.spatial_dropout(x)
        x = self.spatial2(x)
        x = self.spatial_dropout(x)

        x = self.global_pool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)

        if return_features and return_attention:
            return logits, features, attention, channel_att, temporal_att

        if return_features:
            return logits, features

        if return_attention:
            return logits, attention, channel_att, temporal_att

        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)