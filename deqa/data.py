"""Данные лежат так же, как в BASED:

    crops/<сцена>/real_blur.png      размытый вход
    crops/<сцена>/<метод>.png        результат метода
    subj.csv                         колонки video, method, value

value это оценка Брэдли-Терри из субъективного сравнения.
"""

import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from . import features

BLUR_NAME = "real_blur"


def read_table(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df.method != BLUR_NAME]
    return df[["video", "method", "value"]].reset_index(drop=True)


def read_pair(root, video, method):
    blur = cv2.imread(os.path.join(root, video, BLUR_NAME + ".png"))
    res = cv2.imread(os.path.join(root, video, method + ".png"))
    if blur is None or res is None:
        raise FileNotFoundError(f"нет картинок для {video}/{method} в {root}")
    return blur, res


def to_tensor(img):
    return torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255


def classic_features(root, df, cache=None):
    """Признаки считаются по целому фрагменту и долго (секунды на пару), поэтому кэш."""
    if cache and os.path.exists(cache):
        saved = np.load(cache, allow_pickle=True)
        if list(saved["keys"]) == list(df.video + "/" + df.method):
            return saved["feats"]
    feats = np.stack([
        features.compute(*read_pair(root, r.video, r.method))
        for r in tqdm(df.itertuples(), total=len(df), desc="классические признаки")
    ])
    feats = features.squash(feats)
    if cache:
        np.savez(cache, feats=feats, keys=np.array(df.video + "/" + df.method))
    return feats


class Scenes(Dataset):
    """Один элемент это сцена: k результатов разных методов на одном фрагменте.

    Так ранговую потерю можно считать прямо внутри элемента батча.
    """

    def __init__(self, root, df, feats, k=6, crop=128, train=True):
        self.root, self.df, self.feats = root, df, feats
        self.k, self.crop, self.train = k, crop, train
        self.groups = [g.index.to_numpy() for _, g in df.groupby("video", sort=True)]

    def __len__(self):
        return len(self.groups)

    def _crop(self, blur, res):
        # фрагменты бывают 128, 256 и 512, режем всё до наименьшего размера
        h, w = blur.shape[:2]
        c = self.crop
        if h < c or w < c:
            blur, res = (cv2.resize(x, (max(w, c), max(h, c))) for x in (blur, res))
            h, w = blur.shape[:2]
        y = random.randint(0, h - c) if self.train else (h - c) // 2
        x = random.randint(0, w - c) if self.train else (w - c) // 2
        blur, res = blur[y:y + c, x:x + c], res[y:y + c, x:x + c]
        if self.train and random.random() < 0.5:
            blur, res = blur[:, ::-1], res[:, ::-1]
        return blur, res

    def __getitem__(self, i):
        idx = self.groups[i]
        if self.train:
            # если методов меньше k, добиваем повторами, rank_loss их отбросит
            idx = np.random.choice(idx, self.k, replace=len(idx) < self.k)
        blurs, ress = [], []
        for j in idx:
            row = self.df.iloc[j]
            b, r = self._crop(*read_pair(self.root, row.video, row.method))
            blurs.append(to_tensor(b))
            ress.append(to_tensor(r))
        return (torch.stack(blurs), torch.stack(ress),
                torch.from_numpy(self.feats[idx]).float(),
                torch.tensor(self.df.value.to_numpy()[idx], dtype=torch.float32))
