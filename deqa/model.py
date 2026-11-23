import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .features import NAMES

_CHANNELS = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
}


class ScaleBlock(nn.Module):
    """Разность признаков на одном масштабе и её свёртка в вектор."""

    def __init__(self, channels, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(2 * channels, dim, 1), nn.GELU())
        self.attn = nn.Conv2d(dim, 1, 1)

    def forward(self, f_blur, f_res):
        d = f_res - f_blur
        # сама разность и её квадрат. квадрат не зависит от знака
        # и реагирует на любое изменение текстуры
        x = self.proj(torch.cat([d, d * d], dim=1))
        # веса по пространству. там где резкость не поменялась, разность
        # около нуля, и такие области не должны тянуть оценку к среднему
        w = F.softmax(self.attn(x).flatten(2), dim=-1)
        return (x.flatten(2) * w).sum(-1)


class DeQA(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, dim=128, hidden=256):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.blocks = nn.ModuleList([ScaleBlock(c, dim) for c in _CHANNELS[backbone]])

        n = len(NAMES)
        self.classic = nn.Sequential(nn.Linear(n, 32), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(4 * dim + 32, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1)
        )

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # статистики считаются по обучающей выборке в train.py и едут вместе с весами
        self.register_buffer("feat_mean", torch.zeros(n))
        self.register_buffer("feat_std", torch.ones(n))
        self.register_buffer("score_mean", torch.zeros(()))
        self.register_buffer("score_std", torch.ones(()))

    def train(self, mode=True):
        super().train(mode)
        # батчи маленькие, поэтому статистики BatchNorm в кодировщике не трогаем
        for m in list(self.stem.modules()) + list(self.layers.modules()):
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def encode(self, x):
        x = self.stem((x - self.img_mean) / self.img_std)
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats

    def forward(self, blur, res, classic):
        """blur, res: RGB в [0, 1], B x 3 x H x W. classic: B x 9 после squash.

        Возвращает оценку в нормированной шкале, в которой считается лосс.
        """
        fb, fr = self.encode(blur), self.encode(res)
        parts = [block(a, b) for block, a, b in zip(self.blocks, fb, fr)]
        parts.append(self.classic((classic - self.feat_mean) / self.feat_std))
        return self.head(torch.cat(parts, dim=1)).squeeze(1)

    def denorm(self, pred):
        # обратно в шкалу субъективных оценок
        return pred * self.score_std + self.score_mean

    def encoder_params(self):
        return list(self.stem.parameters()) + list(self.layers.parameters())

    def new_params(self):
        return (list(self.blocks.parameters()) + list(self.classic.parameters())
                + list(self.head.parameters()))
