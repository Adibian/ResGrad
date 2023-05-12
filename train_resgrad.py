from utils import load_yaml_file
from resgrad.train import resgrad_train

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("--config", type=str, default='config/Persian/config.yaml', required=False, help="path to config.yaml")
    args = parser.parse_args()
    
    # Read Config
    config = load_yaml_file(args.config)
    resgrad_config = config['resgrad']
    resgrad_config['main'].update(config['main'])
    print(resgrad_config['main'])
    resgrad_train(args, resgrad_config)