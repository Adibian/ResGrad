import argparse

from preprocessor import persian, ljspeech
from utils.tools import load_yaml_file


def main(config):
    if "Persian" in config["main"]["dataset"]:
        persian.prepare_align(config['synthesizer']['preprocess'])
    elif config["main"]["dataset"] == "LJSpeech":
        ljspeech.prepare_align(config['synthesizer']['preprocess'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config.yaml")
    args = parser.parse_args()

    config = load_yaml_file(args.config)
    main(config)
