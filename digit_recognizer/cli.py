"""Command-line entry point: `python -m digit_recognizer {train,predict}`."""
import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from PIL import Image

from .config import DEFAULT_MODEL_PATH, TrainingConfig
from .errors import DigitRecognizerError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="digit_recognizer")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train the CNN on MNIST")
    train.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    train.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)

    predict = sub.add_parser("predict", help="classify an image of a digit")
    predict.add_argument("image", help="path to a PNG/JPG")
    predict.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "train":
            from .training import train
            config = dataclasses.replace(
                TrainingConfig(), epochs=args.epochs, batch_size=args.batch_size)
            result = train(config)
            print(result.report.classification_report)
        else:
            from .predictor import DigitPredictor
            result = DigitPredictor.from_path(Path(args.model)).predict(Image.open(args.image))
            print(f"Predicted digit: {result.digit} ({result.confidence:.1%} confidence)")
    except DigitRecognizerError as exc:       # expected, user-facing failures
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
