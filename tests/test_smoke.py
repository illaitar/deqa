import numpy as np
import torch

from deqa import features
from deqa.losses import rank_loss
from deqa.model import DeQA


def _pair(size=128):
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return np.ascontiguousarray(sharp), np.ascontiguousarray(sharp[:, ::-1])


def test_features():
    blur, res = _pair()
    f = features.compute(blur, res)
    assert f.shape == (len(features.NAMES),)
    assert np.isfinite(f).all()


def test_features_flat_image():
    # на одноцветном кропе раньше вылезал log(0)
    flat = np.full((128, 128, 3), 127, np.uint8)
    assert np.isfinite(features.compute(flat, flat)).all()


def test_forward():
    model = DeQA("resnet18", pretrained=False).eval()
    x = torch.rand(2, 3, 128, 128)
    out = model(x, torch.rand_like(x), torch.zeros(2, len(features.NAMES)))
    assert out.shape == (2,)


def test_rank_loss_order():
    target = torch.tensor([[0.0, 1.0, 2.0]])
    assert rank_loss(target, target) < rank_loss(-target, target)
    # одни повторы: пар нет, лосс ноль
    assert rank_loss(torch.ones(1, 3), torch.ones(1, 3)) == 0
