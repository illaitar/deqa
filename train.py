import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from deqa.data import Scenes, classic_features, read_table
from deqa.evaluation import correlations, predict_table
from deqa.inference import pick_device
from deqa.losses import total_loss
from deqa.model import DeQA


def parse():
    p = argparse.ArgumentParser(description="Обучение DeQA")
    p.add_argument("--crops", required=True, help="папка с фрагментами")
    p.add_argument("--subj", required=True, help="csv с субъективными оценками")
    p.add_argument("--out", default="runs/deqa")
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--scenes", type=int, default=8, help="сцен в батче")
    p.add_argument("--k", type=int, default=6, help="методов на сцену")
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr", type=float, default=3e-5)
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device()

    df = read_table(args.subj)
    feats = classic_features(args.crops, df, cache=os.path.join(args.out, "features.npz"))

    # делим по сценам, иначе один и тот же размытый вход попадёт в обе части
    videos = sorted(df.video.unique())
    random.Random(args.seed).shuffle(videos)
    n_val = max(1, round(len(videos) * args.val_frac))
    is_val = df.video.isin(videos[:n_val]).to_numpy()
    tr, va = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
    f_tr, f_va = feats[~is_val], feats[is_val]
    print(f"сцен: {len(videos) - n_val} обучение, {n_val} проверка, устройство {device}")

    model = DeQA(args.backbone, pretrained=not args.no_pretrained)
    model.feat_mean.copy_(torch.from_numpy(f_tr.mean(0)))
    model.feat_std.copy_(torch.from_numpy(f_tr.std(0) + 1e-6))
    model.score_mean.fill_(float(tr.value.mean()))
    model.score_std.fill_(float(tr.value.std()))
    model.to(device)

    loader = DataLoader(
        Scenes(args.crops, tr, f_tr, k=args.k, crop=args.crop),
        batch_size=args.scenes, shuffle=True, num_workers=args.workers, drop_last=True,
    )
    # кодировщик уже обучен на ImageNet, ему хватает шага на порядок меньше
    opt = torch.optim.AdamW([
        {"params": model.encoder_params(), "lr": args.encoder_lr},
        {"params": model.new_params(), "lr": args.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.encoder_lr, args.lr],
        total_steps=args.epochs * max(1, len(loader)), pct_start=0.1,
    )

    best = -2.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for blur, res, f, y in loader:
            s, k = y.shape
            blur, res, f = (x.flatten(0, 1).to(device) for x in (blur, res, f))
            y = ((y - model.score_mean.cpu()) / model.score_std.cpu()).to(device)
            pred = model(blur, res, f).view(s, k)
            loss = total_loss(pred, y, args.rank_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())

        c = correlations(va, predict_table(model, args.crops, va, f_va, device))
        print(f"{epoch + 1:3d}  loss {np.mean(losses):.4f}  "
              f"srcc {c['srcc']:.3f}  srcc по сценам {c['srcc_scene']:.3f}")
        if c["srcc_scene"] > best:
            best = c["srcc_scene"]
            torch.save({"model": model.state_dict(), "backbone": args.backbone},
                       os.path.join(args.out, "deqa.pth"))
    print("лучший srcc по сценам на проверке:", round(best, 4))


if __name__ == "__main__":
    main()
