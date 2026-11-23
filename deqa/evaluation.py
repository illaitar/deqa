import numpy as np
import torch
from scipy import stats

from .data import read_pair, to_tensor


@torch.no_grad()
def predict_table(model, root, df, feats, device):
    """Оценки для всех строк таблицы. Фрагменты идут целиком, по одному."""
    model.eval()
    out = []
    for i, row in enumerate(df.itertuples()):
        blur, res = read_pair(root, row.video, row.method)
        f = torch.from_numpy(feats[i:i + 1]).float().to(device)
        p = model(to_tensor(blur)[None].to(device), to_tensor(res)[None].to(device), f)
        out.append(model.denorm(p).item())
    return np.array(out)


def correlations(df, pred):
    """PLCC, SRCC и KRCC: по всем парам сразу и в среднем по сценам.

    Среднее по сценам честнее, потому что шкалы Брэдли-Терри у сцен свои.
    """
    funcs = {"plcc": stats.pearsonr, "srcc": stats.spearmanr, "krcc": stats.kendalltau}
    res = {name: float(f(pred, df.value)[0]) for name, f in funcs.items()}
    for name, f in funcs.items():
        per_scene = [
            f(pred[g.index], g.value)[0]
            for _, g in df.groupby("video") if len(g) > 2 and g.value.nunique() > 1
        ]
        res[name + "_scene"] = float(np.nanmean(per_scene)) if per_scene else float("nan")
    return res
