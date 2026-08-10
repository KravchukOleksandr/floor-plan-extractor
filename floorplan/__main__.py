import argparse
from pathlib import Path

from .pipeline import FloorPlanExtractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("weak", "conservative"), default="weak")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    extractor = FloorPlanExtractor(args.weights, args.device)
    result = extractor.extract(args.input, args.preset)
    result.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
