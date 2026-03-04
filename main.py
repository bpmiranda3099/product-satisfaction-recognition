import argparse
import os
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

from src.app import AppConfig, ProductSatisfactionApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam or path to video stream/file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = AppConfig(source=args.source)
    app = ProductSatisfactionApp(config=config, project_root=PROJECT_ROOT)
    app.run()


if __name__ == "__main__":
    main()
