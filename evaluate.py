import argparse

from deqa.data import classic_features, read_table
from deqa.evaluation import correlations, predict_table
from deqa.inference import DeQAMetric


def main():
    p = argparse.ArgumentParser(description="Корреляция DeQA с субъективными оценками")
    p.add_argument("--weights", required=True)
    p.add_argument("--crops", required=True)
    p.add_argument("--subj", required=True)
    p.add_argument("--cache", help="куда положить кэш классических признаков (.npz)")
    p.add_argument("--save", help="csv с оценками по каждой паре")
    args = p.parse_args()

    metric = DeQAMetric(args.weights)
    df = read_table(args.subj)
    feats = classic_features(args.crops, df, cache=args.cache)
    pred = predict_table(metric.model, args.crops, df, feats, metric.device)

    for name, value in correlations(df, pred).items():
        print(f"{name:11s} {value:.4f}")
    if args.save:
        df.assign(deqa=pred).to_csv(args.save, index=False)


if __name__ == "__main__":
    main()
