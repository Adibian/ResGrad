from utils import load_yaml_file

import argparse

from resgrad.train import resgrad_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("-c", "--config", type=str, default='config/Persian/config.yaml', required=False, help="path to config.yaml")
    args = parser.parse_args()
    
    # Read Config
    config = load_yaml_file(args.config)
    resgrad_train(args, config)