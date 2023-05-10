import argparse

from .preprocessor.preprocessor import Preprocessor
from ..utils import load_yaml_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config.yaml")
    args = parser.parse_args()

    config = load_yaml_file(args.config)
    preprocessor = Preprocessor(config['synthesizer']['preprocess'])
    preprocessor.build_from_path()
