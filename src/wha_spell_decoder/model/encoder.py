import torch
from torch import nn


def make_cnn_block(
    *, in_channels, out_channels, kernel_size, stride, padding, num_groups=8
):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.GroupNorm(num_groups, num_channels=out_channels),
        nn.GELU(),
    )

# Returns [dim, h, w] of sincos positionings
def sincos_2d(h, w, dim, device=None, dtype=None):
    assert dim % 4 == 0, "dim must be divisible by 4"
    n = dim // 4
    i = torch.arange(n, device=device, dtype=torch.float32)
    freqs = 1.0 / (100 ** (i / n))

    row_angles = torch.arange(h, device=device, dtype=torch.float32)[None, :] * freqs[:, None]
    row_positions = torch.stack([torch.sin(row_angles), torch.cos(row_angles)], dim=1).flatten(0, 1)

    col_angles = torch.arange(w, device=device, dtype=torch.float32)[None, :] * freqs[:, None]
    col_positions = torch.stack([torch.sin(col_angles), torch.cos(col_angles)], dim=1).flatten(0, 1)

    positions = torch.cat([row_positions[:, :, None].expand(-1, h, w), col_positions[:, None, :].expand(-1, h, w)], dim=0)
    return positions.to(dtype=dtype)

# Forward contains x + block(x) instead of just block(x)
class ResidualBlock(nn.Module):
    def __init__(self, *, channels, kernel_size, num_groups=8):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.activation = nn.GELU()

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        return self.activation(x + y)


class ImageEncoder(nn.Module):
    def __init__(self, *, d_model):
        super().__init__()

        # WxH -> W/2xH/2 i.e. 1920x1080 -> 960x540
        self.stem = make_cnn_block(
            in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3
        )
        # Maintain resolution
        self.stage1 = nn.Sequential(
            *[ResidualBlock(channels=32, kernel_size=3) for _ in range(3)]
        )
        # W/2xH/2 -> W/4xH/4 i.e. 960x540 -> 480x270
        self.downsample2 = make_cnn_block(
            in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2
        )
        # Maintain resolution
        self.stage2 = nn.Sequential(
            *[ResidualBlock(channels=64, kernel_size=3) for _ in range(3)]
        )
        # W/4xH/4 -> W/8xH/8 i.e. 480x270 -> 240x135
        self.downsample3 = make_cnn_block(
            in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2
        )
        # Maintain resolution
        self.stage3 = nn.Sequential(
            *[ResidualBlock(channels=128, kernel_size=3) for _ in range(3)]
        )
        # W/8xH/8 -> W/16xH/16 i.e 240x135 -> 120x68
        self.downsample4 = make_cnn_block(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1
        )
        # Maintain resolution
        self.stage4 = nn.Sequential(
            *[ResidualBlock(channels=256, kernel_size=3) for _ in range(5)]
        )
        # W/16xH/16 -> W/32xH/32 i.e 120x68 -> 60x34
        self.downsample5 = make_cnn_block(
            in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        # Maintain resolution
        self.stage5 = nn.Sequential(
            *[ResidualBlock(channels=512, kernel_size=3) for _ in range(5)]
        )
        # W/32xH/32 -> W/64xH/64 i.e 60x34 -> 30x17
        self.downsample6 = make_cnn_block(
            in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1
        )
        # Maintain resolution
        self.stage6 = nn.Sequential(
            *[ResidualBlock(channels=512, kernel_size=3) for _ in range(3)]
        )
        # remap to same dimension as transformer tokens
        self.proj = nn.Conv2d(in_channels=512, out_channels=d_model, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample2(x)
        x = self.stage2(x)
        x = self.downsample3(x)
        x = self.stage3(x)
        x = self.downsample4(x)
        x = self.stage4(x)
        x = self.downsample5(x)
        x = self.stage5(x)
        x = self.downsample6(x)
        x = self.stage6(x)
        # Add position embeddings
        _, C, H, W = x.shape
        x = x + sincos_2d(H, W, C, device=x.device, dtype=x.dtype)
        x = self.proj(x)
        # Flatten x row major, reshape to be Bx(H*W)xC
        x = x.flatten(2).transpose(1, 2)
        
        return x