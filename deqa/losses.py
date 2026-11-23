import torch
import torch.nn.functional as F


def rank_loss(pred, target, eps=1e-3):
    """Попарная ранговая потеря внутри сцены.

    pred, target: S x K, в строке результаты разных методов на одном фрагменте.
    Сравнивать оценки между сценами нельзя: модель Брэдли-Терри строится для
    каждой сцены отдельно, и общий ноль у шкал разный.
    """
    dp = pred.unsqueeze(2) - pred.unsqueeze(1)
    dt = target.unsqueeze(2) - target.unsqueeze(1)
    # пары с одинаковой оценкой пропускаем, сюда же попадают повторы из добивки батча
    mask = dt.abs() > eps
    if not mask.any():
        return pred.sum() * 0
    return F.softplus(-torch.sign(dt[mask]) * dp[mask]).mean()


def total_loss(pred, target, rank_weight=1.0):
    return F.mse_loss(pred, target) + rank_weight * rank_loss(pred, target)
