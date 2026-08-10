import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_small


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, channels=512, scales=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvBNReLU(in_channels, channels, 1))
            for scale in scales
        ])
        self.bottleneck = ConvBNReLU(in_channels + len(scales) * channels, channels)

    def forward(self, x):
        pooled = [F.interpolate(branch(x), x.shape[-2:], mode="bilinear", align_corners=False)
                  for branch in self.branches]
        return self.bottleneck(torch.cat([x, *pooled], 1))


class UPerHead(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), channels=512, dropout=0.1):
        super().__init__()
        self.ppm = PyramidPooling(in_channels[-1], channels)
        self.lateral = nn.ModuleList([ConvBNReLU(value, channels, 1) for value in in_channels[:-1]])
        self.fpn_convs = nn.ModuleList([ConvBNReLU(channels, channels) for _ in in_channels[:-1]])
        self.fpn_bottleneck = ConvBNReLU(len(in_channels) * channels, channels)
        self.classifier = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(channels, 1, 1))

    def forward(self, features, output_size):
        c1, c2, c3, c4 = features
        laterals = [self.lateral[0](c1), self.lateral[1](c2), self.lateral[2](c3), self.ppm(c4)]
        for index in range(3, 0, -1):
            laterals[index - 1] += F.interpolate(
                laterals[index], laterals[index - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        outputs = [self.fpn_convs[index](laterals[index]) for index in range(3)] + [laterals[3]]
        size = outputs[0].shape[-2:]
        outputs = [outputs[0]] + [F.interpolate(x, size, mode="bilinear", align_corners=False)
                                  for x in outputs[1:]]
        logits = self.classifier(self.fpn_bottleneck(torch.cat(outputs, 1)))
        return F.interpolate(logits, output_size, mode="bilinear", align_corners=False)


class HomographyHead(nn.Module):
    def __init__(self, in_channels=768):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(4096, 256), nn.GELU(), nn.Linear(256, 8))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        a = self.mlp(self.pool(self.reduce(x)))
        one = torch.ones_like(a[:, :1])
        return torch.cat([
            one + a[:, 0:1], a[:, 1:2], a[:, 2:3],
            a[:, 3:4], one + a[:, 4:5], a[:, 5:6],
            a[:, 6:7], a[:, 7:8], one,
        ], 1).view(-1, 3, 3)


def warp_homography(x, homography):
    batch, _, height, width = x.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, height, device=x.device),
        torch.linspace(-1, 1, width, device=x.device), indexing="ij",
    )
    points = torch.stack((xs, ys, torch.ones_like(xs))).reshape(1, 3, -1).expand(batch, -1, -1)
    source = torch.linalg.solve(homography.float(), points)
    z = source[:, 2:3]
    eps = torch.finfo(source.dtype).eps
    z = torch.where(z.abs() < eps, torch.where(z >= 0, eps, -eps), z)
    grid = (source[:, :2] / z).transpose(1, 2).reshape(batch, height, width, 2).to(x.dtype)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class WallProjectionNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        backbone = convnext_small(weights=None)
        old_stem = backbone.features[0][0]
        stem = nn.Conv2d(
            in_channels, old_stem.out_channels, old_stem.kernel_size,
            old_stem.stride, old_stem.padding, bias=old_stem.bias is not None,
        )
        backbone.features[0][0] = stem
        self.features = backbone.features
        self.top_head = UPerHead()
        self.homography_head = HomographyHead()

    def forward_backbone(self, x):
        x = self.features[0](x)
        c1 = self.features[1](x)
        c2 = self.features[3](self.features[2](c1))
        c3 = self.features[5](self.features[4](c2))
        c4 = self.features[7](self.features[6](c3))
        return c1, c2, c3, c4

    def forward(self, x):
        features = self.forward_backbone(x)
        top_logits = self.top_head(features, x.shape[-2:])
        top = torch.sigmoid(top_logits)
        homography = self.homography_head(features[-1])
        return {"top_logits": top_logits, "top": top, "H": homography,
                "bottom": warp_homography(top, homography)}
