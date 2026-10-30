import torch.nn as nn


def make_cnn_block(
    *, in_channels, out_channels, kernel_size, stride, padding, num_groups=8
):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.GroupNorm(num_groups, num_channels=out_channels),
        nn.GELU(),
    )


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
    def __init__(self):
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
        # TODO remap to same dimension as transformer tokens
    
    def forward(self):
        pass # TODO