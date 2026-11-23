import argparse

import cv2

from deqa import DeQAMetric


def main():
    p = argparse.ArgumentParser(description="DeQA для одной пары картинок")
    p.add_argument("blur", help="размытый вход")
    p.add_argument("result", help="результат метода")
    p.add_argument("--weights", required=True)
    args = p.parse_args()

    blur, res = cv2.imread(args.blur), cv2.imread(args.result)
    if blur is None or res is None:
        raise SystemExit("не удалось прочитать картинки")
    print(f"{DeQAMetric(args.weights)(blur, res):.4f}")


if __name__ == "__main__":
    main()
