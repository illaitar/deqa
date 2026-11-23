import torch

from . import features
from .data import to_tensor
from .model import DeQA


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DeQAMetric:
    """Обёртка для подсчёта метрики на парах картинок.

        metric = DeQAMetric("deqa.pth")
        score = metric(blur, result)    # uint8 BGR, больше значит лучше
    """

    def __init__(self, weights, device=None):
        self.device = device or pick_device()
        ckpt = torch.load(weights, map_location="cpu")
        self.model = DeQA(backbone=ckpt["backbone"], pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, blur, res, classic=None):
        if blur.shape != res.shape:
            raise ValueError(f"размеры не совпадают: {blur.shape} и {res.shape}")
        if classic is None:
            classic = features.squash(features.compute(blur, res))
        f = torch.as_tensor(classic, dtype=torch.float32)[None].to(self.device)
        b = to_tensor(blur)[None].to(self.device)
        r = to_tensor(res)[None].to(self.device)
        return self.model.denorm(self.model(b, r, f)).item()
